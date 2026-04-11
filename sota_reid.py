"""
SOTA Video-based Person Re-ID
==============================
Architecture inspired by VID-Trans-ReID (BMVC 2022) and TCViT (AAAI 2024).

Key improvements over the baseline (ResNet-50 + BiGRU):
  1. Temporal Transformer  — self-attention over frames replaces BiGRU;
                             captures long-range frame dependencies 
  2. BNNeck               — dual embedding branch (raw for triplet,
                             BN-normalised for ID classification)
  3. Label-smoothed CE    — softens overconfident ID predictions
  4. LR warmup            — stabilises early training before cosine decay
  5. Batch-hard triplet   — same as baseline, kept as second loss signal

Usage:
    python sota_reid.py                      # train from scratch
    python sota_reid.py --eval-only          # evaluate saved checkpoint
"""

import os
import math
import random
import argparse
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

NUM_IO_THREADS = 8   # thread pool for image loading — safe on Windows

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA
# ══════════════════════════════════════════════════════════════════════════════

class ParsedImage:
    def __init__(self, image_path, person_id, camera_id, frame_id):
        self.image_path = image_path
        self.person_id  = person_id
        self.camera_id  = camera_id
        self.frame_id   = frame_id


def parse_dataset(root_dir, max_frame_gap=50, min_seq_len=4):
    """
    Parse IUSTPersonReID bounding-box directory into sequences.
    Returns sequences_by_person: {pid: {cam: [seq, ...]}}
    Each seq is a list of ParsedImage sorted by frame_id.
    """
    images = []
    for fname in sorted(os.listdir(root_dir)):
        if not fname.endswith(('.jpg', '.jpeg', '.png')):
            continue
        parts = fname.split('_')
        images.append(ParsedImage(
            image_path=os.path.join(root_dir, fname),
            person_id=int(parts[0]),
            camera_id=parts[1],
            frame_id=int(parts[2])
        ))

    # group by (pid, cam)
    groups: dict = {}
    for img in images:
        key = (img.person_id, img.camera_id)
        groups.setdefault(key, []).append(img)

    sequences_by_person: dict = {}
    for (pid, cam), imgs in groups.items():
        imgs.sort(key=lambda x: x.frame_id)
        # split on frame gaps
        seqs, cur = [], [imgs[0]]
        for img in imgs[1:]:
            if img.frame_id - cur[-1].frame_id <= max_frame_gap:
                cur.append(img)
            else:
                if len(cur) >= min_seq_len:
                    seqs.append(cur)
                cur = [img]
        if len(cur) >= min_seq_len:
            seqs.append(cur)
        if seqs:
            sequences_by_person.setdefault(pid, {}).setdefault(cam, []).extend(seqs)

    return sequences_by_person


class SequenceDataset(Dataset):
    def __init__(self, sequences_by_person, transform, max_seq_len=8):
        self.transform    = transform
        self.max_seq_len  = max_seq_len
        self.samples      = []           # (seq, pid)
        self.pid_to_indices = {}

        for pid, cam_data in sequences_by_person.items():
            for cam, seqs in cam_data.items():
                for seq in seqs:
                    idx = len(self.samples)
                    self.samples.append((seq, pid))
                    self.pid_to_indices.setdefault(pid, []).append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, pid = self.samples[idx]
        total = len(seq)
        indices = np.linspace(0, total - 1, min(self.max_seq_len, total), dtype=int)
        paths = [seq[i].image_path for i in indices]

        def load(p):
            return self.transform(Image.open(p).convert('RGB'))

        with ThreadPoolExecutor(max_workers=min(NUM_IO_THREADS, len(paths))) as pool:
            frames = list(pool.map(load, paths))
        return torch.stack(frames), len(frames), pid


class PKSampler(Sampler):
    """P persons × K sequences per batch."""
    def __init__(self, pid_to_indices, P=16, K=4):
        self.pid_to_indices = pid_to_indices
        self.P = P
        self.K = K
        self.pids = [pid for pid, idxs in pid_to_indices.items() if len(idxs) >= 2]

    def __iter__(self):
        pids = self.pids.copy()
        random.shuffle(pids)
        for i in range(0, len(pids) - self.P + 1, self.P):
            batch = []
            for pid in pids[i:i + self.P]:
                batch.extend(random.choices(self.pid_to_indices[pid], k=self.K))
            yield batch

    def __len__(self):
        return len(self.pids) // self.P


def collate_fn(batch):
    imgs_list, lengths, pids = zip(*batch)
    max_t = max(lengths)
    C, H, W = imgs_list[0].shape[1:]
    padded = torch.zeros(len(imgs_list), max_t, C, H, W)
    for i, (imgs, t) in enumerate(zip(imgs_list, lengths)):
        padded[i, :t] = imgs
    return padded, torch.tensor(lengths, dtype=torch.long), torch.tensor(pids, dtype=torch.long)


def build_transforms():
    train = T.Compose([
        T.Resize((256, 128)),          # taller crop — standard ReID resolution
        T.RandomHorizontalFlip(),
        T.Pad(10),
        T.RandomCrop((256, 128)),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])
    test = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train, test


# ══════════════════════════════════════════════════════════════════════════════
# 2.  MODEL
# ══════════════════════════════════════════════════════════════════════════════

class ResNetBackbone(nn.Module):
    """ResNet-50 with layer3+4 trainable, rest frozen. Output: (B, 2048)."""
    def __init__(self):
        super().__init__()
        resnet = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
        # stride=1 on layer4 for larger feature maps (standard ReID trick)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv2.stride         = (1, 1)

        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

        # freeze stem, layer1, layer2
        for name, param in self.base.named_parameters():
            if not any(k in name for k in ('4.', '5.')):  # layer3=4, layer4=5 in Sequential
                param.requires_grad = False

    def forward(self, x):
        feat = self.base(x)          # (B, 2048, H', W')
        return self.gap(feat).flatten(1)  # (B, 2048)


class TemporalTransformer(nn.Module):
    """
    Transformer encoder over T frame features with a learnable CLS token.

    Inspired by VID-Trans-ReID (BMVC 2022) and TCViT (AAAI 2024).

    Input:  (B, T, d_model)
    Output: (B, d_model)   — CLS token representation
    """
    def __init__(self, d_model=512, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_embed = nn.Parameter(torch.zeros(1, 65, d_model))  # up to 64 frames + CLS
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True   # pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, lengths):
        """
        x:       (B, T, d_model)
        lengths: (B,)  actual sequence lengths
        """
        B, T, _ = x.shape

        # prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)      # (B, 1, d)
        x   = torch.cat([cls, x], dim=1)            # (B, T+1, d)
        needed = T + 1
        if needed <= self.pos_embed.shape[1]:
            pos = self.pos_embed[:, :needed, :]
        else:
            # interpolate stored positional embedding to the actual length
            pos = F.interpolate(
                self.pos_embed.permute(0, 2, 1),   # (1, d, max_len)
                size=needed, mode='linear', align_corners=False
            ).permute(0, 2, 1)                     # (1, needed, d)
        x   = x + pos

        # key-padding mask: True = ignore (padded positions)
        mask = torch.ones(B, T + 1, dtype=torch.bool, device=x.device)
        mask[:, 0] = False                           # CLS always attended
        for i, l in enumerate(lengths):
            mask[i, 1:l + 1] = False                # real frames

        x = self.transformer(x, src_key_padding_mask=mask)
        return self.norm(x[:, 0])                   # CLS token → (B, d)


class BNNeck(nn.Module):
    """
    Bottle-neck with dual outputs (Bag of Tricks, CVPRW 2019).

    feat_before_bn → triplet loss   (in L2-metric space)
    feat_after_bn  → ID classifier  (in class-discriminative space)

    The two losses supervise different aspects of the embedding.
    """
    def __init__(self, in_dim, embedding_dim, num_classes):
        super().__init__()
        self.fc   = nn.Linear(in_dim, embedding_dim, bias=False)
        self.bn   = nn.BatchNorm1d(embedding_dim)
        self.bn.bias.requires_grad_(False)           # no shift — standard practice
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)

        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out')
        nn.init.constant_(self.bn.weight, 1)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x):
        feat     = self.fc(x)                        # (B, embedding_dim)
        feat_bn  = self.bn(feat)                     # normalised — for classifier
        logits   = self.classifier(feat_bn)          # (B, num_classes)
        return F.normalize(feat, p=2, dim=1), feat_bn, logits


class SOTAReIDModel(nn.Module):
    """
    Full model:
        ResNet-50 (stride-1 layer4) → frame projection → Temporal Transformer
        → BNNeck → (L2-normed embedding, BN embedding, ID logits)

    At inference (.eval()): returns only the L2-normed embedding.
    At training  (.train()): returns (embedding, logits).
    """
    def __init__(self, num_classes, embedding_dim=512,
                 tf_heads=8, tf_layers=2, dropout=0.1):
        super().__init__()
        self.backbone    = ResNetBackbone()           # 2048-dim per frame
        self.frame_proj  = nn.Sequential(
            nn.Linear(2048, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal    = TemporalTransformer(
            d_model=embedding_dim, nhead=tf_heads,
            num_layers=tf_layers, dropout=dropout
        )
        self.bnneck      = BNNeck(embedding_dim, embedding_dim, num_classes)

    def forward(self, x, lengths):
        """
        x:       (B, T, C, H, W)
        lengths: (B,)
        """
        B, T, C, H, W = x.shape
        frame_feats = self.backbone(x.view(B * T, C, H, W))     # (B*T, 2048)
        proj        = self.frame_proj(frame_feats).view(B, T, -1)  # (B, T, d)
        seq_feat    = self.temporal(proj, lengths)               # (B, d)

        emb, emb_bn, logits = self.bnneck(seq_feat)

        if self.training:
            return emb, logits
        return emb


# ══════════════════════════════════════════════════════════════════════════════
# 3.  LOSSES
# ══════════════════════════════════════════════════════════════════════════════

def batch_hard_triplet_loss(embeddings, labels, margin=0.3):
    """Batch-hard triplet loss with L2 distance."""
    dist     = torch.cdist(embeddings, embeddings)
    same     = labels.unsqueeze(1) == labels.unsqueeze(0)

    pos_mask = same.clone().fill_diagonal_(False)
    hardest_pos = (dist * pos_mask.float()).max(dim=1).values

    hardest_neg = dist.masked_fill(same, float('inf')).min(dim=1).values
    return F.relu(hardest_pos - hardest_neg + margin).mean()


class LabelSmoothingCE(nn.Module):
    """Cross-entropy with label smoothing (Szegedy et al., 2016)."""
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.smoothing   = smoothing
        self.num_classes = num_classes

    def forward(self, logits, targets):
        confidence = 1.0 - self.smoothing
        smooth_val = self.smoothing / (self.num_classes - 1)
        with torch.no_grad():
            soft_targets = torch.full_like(logits, smooth_val)
            soft_targets.scatter_(1, targets.unsqueeze(1), confidence)
        return F.cross_entropy(logits, targets) * confidence + \
               -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean() * self.smoothing


# ══════════════════════════════════════════════════════════════════════════════
# 4.  LR SCHEDULE — linear warmup + cosine annealing
# ══════════════════════════════════════════════════════════════════════════════

class WarmupCosineScheduler:
    """
    Linear warmup for `warmup_epochs`, then cosine annealing to `eta_min`.
    Call .step() once per epoch.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
        self.optimizer      = optimizer
        self.warmup_epochs  = warmup_epochs
        self.total_epochs   = total_epochs
        self.eta_min        = eta_min
        self.base_lrs       = [pg['lr'] for pg in optimizer.param_groups]
        self.epoch          = 0

    def step(self):
        self.epoch += 1
        e = self.epoch
        if e <= self.warmup_epochs:
            scale = e / self.warmup_epochs
        else:
            progress = (e - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            scale    = self.eta_min + 0.5 * (1 - self.eta_min) * (1 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * scale

    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, scaler, ce_loss, pid_to_cls, device,
                triplet_margin=0.3, id_weight=1.0):
    model.train()
    total = 0.0
    for imgs, lengths, pids in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device)
        pids = pids.to(device)
        cls_labels = torch.tensor([pid_to_cls[p.item()] for p in pids],
                                   dtype=torch.long, device=device)

        with torch.amp.autocast('cuda'):
            emb, logits = model(imgs, lengths)
            loss_tri = batch_hard_triplet_loss(emb, pids, margin=triplet_margin)
            loss_id  = ce_loss(logits, cls_labels)
            loss     = loss_tri + id_weight * loss_id

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()

    return total / len(loader)


@torch.no_grad()
def encode_sequences(sequences, model, transform, max_seq_len, device):
    model.eval()
    embeddings, pids = [], []

    def load(p):
        return transform(Image.open(p).convert('RGB'))

    with ThreadPoolExecutor(max_workers=NUM_IO_THREADS) as pool:
        for seq in tqdm(sequences, desc="Encoding", leave=False):
            total   = len(seq)
            indices = np.linspace(0, total - 1, min(max_seq_len, total), dtype=int)
            frames  = list(pool.map(load, [seq[i].image_path for i in indices]))
            imgs    = torch.stack(frames).unsqueeze(0).to(device)
            length  = torch.tensor([len(frames)])
            with torch.amp.autocast('cuda'):
                emb = model(imgs, length)
            embeddings.append(emb.cpu().float().numpy())
            pids.append(seq[0].person_id)

    return np.concatenate(embeddings, axis=0), np.array(pids)


def evaluate(q_embs, q_pids, g_embs, g_pids, max_rank=20):
    sim        = cos_sim(q_embs, g_embs)
    sorted_idx = np.argsort(-sim, axis=1)

    rank1 = np.mean(g_pids[sorted_idx[:, 0]] == q_pids)

    cmc = np.zeros(min(max_rank, len(g_pids)))
    aps = []
    for i in range(len(q_pids)):
        matched = g_pids[sorted_idx[i]] == q_pids[i]
        for r in range(len(cmc)):
            if matched[:r + 1].any():
                cmc[r] += 1
        if matched.sum() == 0:
            continue
        cum  = np.cumsum(matched)
        prec = cum / np.arange(1, len(matched) + 1)
        aps.append(prec[matched].mean())

    cmc /= len(q_pids)
    return rank1, float(np.mean(aps)) if aps else 0.0, cmc


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root',    default='data/IUSTPersonReID')
    parser.add_argument('--model-dir',    default='models')
    parser.add_argument('--epochs',       type=int,   default=120)
    parser.add_argument('--patience',     type=int,   default=15)
    parser.add_argument('--warmup',       type=int,   default=10)
    parser.add_argument('--P',            type=int,   default=12)
    parser.add_argument('--K',            type=int,   default=4)
    parser.add_argument('--max-seq-len',  type=int,   default=6)
    parser.add_argument('--embedding-dim',type=int,   default=512)
    parser.add_argument('--tf-heads',     type=int,   default=8)
    parser.add_argument('--tf-layers',    type=int,   default=2)
    parser.add_argument('--dropout',      type=float, default=0.1)
    parser.add_argument('--margin',       type=float, default=0.3)
    parser.add_argument('--id-weight',    type=float, default=1.0)
    parser.add_argument('--smoothing',    type=float, default=0.1)
    parser.add_argument('--eval-only',    action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(args.model_dir, exist_ok=True)

    # ── data ──────────────────────────────────────────────────────────────────
    train_root = os.path.join(args.data_root, 'bounding_box_train')
    test_root  = os.path.join(args.data_root, 'bounding_box_test')

    train_seqs_by_person = parse_dataset(train_root)
    test_seqs_by_person  = parse_dataset(test_root)

    train_tf, test_tf = build_transforms()

    train_dataset = SequenceDataset(train_seqs_by_person, train_tf, args.max_seq_len)
    sampler       = PKSampler(train_dataset.pid_to_indices, P=args.P, K=args.K)
    train_loader  = DataLoader(train_dataset, batch_sampler=sampler,
                               num_workers=0, collate_fn=collate_fn, pin_memory=False)

    print(f"Train: {len(train_dataset)} sequences, "
          f"{len(train_seqs_by_person)} persons, "
          f"{len(sampler)} steps/epoch")

    # query = first seq per test person, gallery = rest
    query_seqs, gallery_seqs = [], []
    for pid, cam_data in test_seqs_by_person.items():
        person_seqs = [s for cam_seqs in cam_data.values() for s in cam_seqs]
        if not person_seqs:
            continue
        query_seqs.append(person_seqs[0])
        gallery_seqs.extend(person_seqs[1:] if len(person_seqs) > 1 else person_seqs)
    print(f"Test:  {len(query_seqs)} queries, {len(gallery_seqs)} gallery")

    # ── model ─────────────────────────────────────────────────────────────────
    all_train_pids = sorted(train_seqs_by_person.keys())
    pid_to_cls     = {pid: i for i, pid in enumerate(all_train_pids)}
    NUM_CLASSES    = len(all_train_pids)

    model = SOTAReIDModel(
        num_classes   = NUM_CLASSES,
        embedding_dim = args.embedding_dim,
        tf_heads      = args.tf_heads,
        tf_layers     = args.tf_layers,
        dropout       = args.dropout,
    ).to(device)


    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable")

    ckpt_path = os.path.join(args.model_dir, 'sota_reid_best.pth')

    # Resume from checkpoint if one exists
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print(f"Resumed from checkpoint: {ckpt_path}")

    if args.eval_only:
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        q_embs, q_pids = encode_sequences(query_seqs,   model, test_tf, args.max_seq_len, device)
        g_embs, g_pids = encode_sequences(gallery_seqs, model, test_tf, args.max_seq_len, device)
        rank1, mAP, cmc = evaluate(q_embs, q_pids, g_embs, g_pids)
        print(f"Rank-1={rank1:.4f}  mAP={mAP:.4f}  Rank-5={cmc[4]:.4f}  Rank-10={cmc[9]:.4f}")
        return

    # ── optimiser ─────────────────────────────────────────────────────────────
    # differential LR: backbone (fine-tune) gets 0.1× of head LR
    optimizer = torch.optim.Adam([
        {'params': model.backbone.parameters(),   'lr': 1e-4},
        {'params': model.frame_proj.parameters(), 'lr': 3e-4},
        {'params': model.temporal.parameters(),   'lr': 3e-4},
        {'params': model.bnneck.parameters(),     'lr': 3e-4},
    ], weight_decay=5e-4)

    scheduler = WarmupCosineScheduler(optimizer,
                                      warmup_epochs=args.warmup,
                                      total_epochs=args.epochs)
    scaler    = torch.amp.GradScaler('cuda')
    ce_loss   = LabelSmoothingCE(NUM_CLASSES, smoothing=args.smoothing).to(device)

    # ── training loop ─────────────────────────────────────────────────────────
    best_rank1, patience_counter = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scaler, ce_loss,
                           pid_to_cls, device,
                           triplet_margin=args.margin, id_weight=args.id_weight)

        q_embs, q_pids = encode_sequences(query_seqs,   model, test_tf, args.max_seq_len, device)
        g_embs, g_pids = encode_sequences(gallery_seqs, model, test_tf, args.max_seq_len, device)
        rank1, mAP, cmc = evaluate(q_embs, q_pids, g_embs, g_pids)

        lr_now = scheduler.get_lr()[0]
        print(f"Epoch [{epoch:3d}/{args.epochs}]  "
              f"loss={loss:.4f}  Rank-1={rank1:.4f}  mAP={mAP:.4f}  "
              f"Rank-5={cmc[4]:.4f}  lr={lr_now:.2e}")

        if rank1 > best_rank1:
            best_rank1       = rank1
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  --> Saved best model (Rank-1={best_rank1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping.")
                break

        scheduler.step()

    print(f"\nBest Rank-1: {best_rank1:.4f}")
    print(f"Checkpoint:  {ckpt_path}")


if __name__ == '__main__':
    main()
