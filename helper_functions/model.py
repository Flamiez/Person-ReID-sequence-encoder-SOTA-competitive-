import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class TemporalTransformer(nn.Module):
    def __init__(self, dim, heads=8, layers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, x, pad_mask=None):
        return self.encoder(x, src_key_padding_mask=pad_mask)


class ImprovedSequenceEncoder(nn.Module):
    def __init__(self, rnn_hidden=512, embedding_dim=512, dropout=0.3):
        super().__init__()

        # DINOv2-B/14: self-supervised ViT trained on 142M images.
        # patch grid: (256/14) × (128/14) ≈ 18 × 9 patch tokens (timm interpolates).
        self.vit = timm.create_model(
            'vit_base_patch14_dinov2.lvd142m',
            pretrained=True,
            img_size=(256, 128),
            num_classes=0,
        )
        self.patch_h = 256 // 14   # 18 patch rows
        self.patch_w = 128 // 14   # 9  patch cols
        vit_dim = 768

        # All blocks trainable — layer-wise LR decay applied in the optimizer.
        for param in self.vit.parameters():
            param.requires_grad = True

        # Recompute activations during backward instead of storing them (~40% VRAM saved)
        self.vit.set_grad_checkpointing(enable=True)

        # Global projection
        self.global_proj  = nn.Linear(vit_dim, rnn_hidden)          # 768 → 512

        # Multi-granularity local projectors — dims sum to rnn_hidden per granularity:
        #   2 stripes × (rnn_hidden//2=256) = 512
        #   4 stripes × (rnn_hidden//4=128) = 512
        #   8 stripes × (rnn_hidden//8=64)  = 512
        # combined_dim = 512 + 512 + 512 + 512 = rnn_hidden * 4 = 2048
        self.local_proj_2 = nn.Linear(vit_dim, rnn_hidden // 2)
        self.local_proj_4 = nn.Linear(vit_dim, rnn_hidden // 4)
        self.local_proj_8 = nn.Linear(vit_dim, rnn_hidden // 8)

        combined_dim = rnn_hidden * 4   # 2048

        # Temporal Transformer over the per-frame combined features
        self.temporal_tf = TemporalTransformer(combined_dim)

        # Bidirectional GRU
        self.rnn = nn.GRU(
            input_size=combined_dim,
            hidden_size=rnn_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Attention pooling over time steps
        self.attention = TemporalAttention(rnn_hidden * 2)

        # Final embedding head
        self.embed_head = nn.Sequential(
            nn.Linear(rnn_hidden * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x, lengths):
        B, T, C, H, W = x.shape
        frames = x.view(B * T, C, H, W)

        # ViT forward — CLS token at index 0, patch tokens at 1:
        tokens       = self.vit.forward_features(frames)
        cls_token    = tokens[:, 0]                        # (B*T, 768)
        patch_tokens = tokens[:, 1:]                       # (B*T, 162, 768)

        # Reshape patches to spatial grid (18 × 9), then crop to 16 rows.
        # 18 is not divisible by 8; cropping to 16 (divisible by 2, 4, 8) ensures
        # torch.chunk returns exactly 2/4/8 equal-sized stripes.
        patch_tokens = patch_tokens.view(B * T, self.patch_h, self.patch_w, -1)
        patch_tokens = patch_tokens[:, :16, :, :]         # (B*T, 16, 9, 768)

        # Multi-granularity stripe pooling
        stripes_2 = torch.chunk(patch_tokens, 2, dim=1)   # 2 × (B*T, 8, 9, 768)
        stripes_4 = torch.chunk(patch_tokens, 4, dim=1)   # 4 × (B*T, 4, 9, 768)
        stripes_8 = torch.chunk(patch_tokens, 8, dim=1)   # 8 × (B*T, 2, 9, 768)

        local_2 = [s.mean(dim=[1, 2]) for s in stripes_2]  # 2 × (B*T, 768)
        local_4 = [s.mean(dim=[1, 2]) for s in stripes_4]  # 4 × (B*T, 768)
        local_8 = [s.mean(dim=[1, 2]) for s in stripes_8]  # 8 × (B*T, 768)

        # Project each granularity
        g        = self.global_proj(cls_token)                     # (B*T, 512)
        locals_2 = [self.local_proj_2(f) for f in local_2]        # 2 × (B*T, 256)
        locals_4 = [self.local_proj_4(f) for f in local_4]        # 4 × (B*T, 128)
        locals_8 = [self.local_proj_8(f) for f in local_8]        # 8 × (B*T,  64)

        # Concatenate → (B*T, 2048)
        feat = torch.cat([g] + locals_2 + locals_4 + locals_8, dim=1)
        feat = feat.view(B, T, -1)

        # Stripe features for temporal stripe consistency loss (4-stripe, 128-dim)
        stripe_feats = torch.stack(locals_4, dim=1).view(B, T, 4, -1)

        # Temporal Transformer (mask padded frames)
        pad_mask = torch.arange(T, device=x.device).unsqueeze(0) >= \
                   lengths.to(x.device).unsqueeze(1)
        feat = self.temporal_tf(feat, pad_mask=pad_mask)

        # Bidirectional GRU
        packed  = nn.utils.rnn.pack_padded_sequence(
            feat, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        rnn_out, _ = self.rnn(packed)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)

        # Mask padding positions
        max_len = rnn_out.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < \
               lengths.unsqueeze(1).to(x.device)

        # Attention pooling → embedding
        pooled    = self.attention(rnn_out, mask)
        embedding = self.embed_head(pooled)
        return F.normalize(embedding, dim=1), stripe_feats


class TemporalAttention(nn.Module):
    """Scalar dot-product attention over time steps to pool RNN outputs."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, rnn_out, mask=None):
        scores = self.attn(rnn_out).squeeze(-1)           # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        weights = F.softmax(scores, dim=-1).unsqueeze(-1) # (B, T, 1)
        return (weights * rnn_out).sum(dim=1)             # (B, H)


class ImprovedSeqToSeqReIDModel(nn.Module):
    def __init__(self, embedding_dim=512, rnn_hidden=512, num_classes=None):
        super().__init__()

        self.encoder = ImprovedSequenceEncoder(rnn_hidden=rnn_hidden,
                                               embedding_dim=embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False) \
            if num_classes is not None else None

    def forward(self, x, lengths):
        embedding, stripe_feats = self.encoder(x, lengths)

        if self.training:
            if self.classifier is not None:
                # AMSoftmax: cosine similarity between normalized embedding and
                # normalized classifier weights. Margin applied in the loss.
                normed_w = F.normalize(self.classifier.weight, dim=1)
                logits   = F.linear(embedding, normed_w)   # cosine sims in [-1, 1]
            else:
                logits = None
            return embedding, logits, stripe_feats

        return embedding


class DetectThenReIDModel(nn.Module):
    """
    Single-model wrapper: YOLOv8 person detector → crop → ImprovedSeqToSeqReIDModel.

    Designed for full-frame inputs (surveillance footage, IUSTPersonReID, etc.).
    The detector is frozen and excluded from parameters() / state_dict().
    Only the ReID model is trained / saved.

    Args:
        detector      : ultralytics YOLO model (frozen, not registered as submodule)
        reid_model    : ImprovedSeqToSeqReIDModel
        reid_transform: torchvision transform applied to each cropped PIL frame
        conf_thresh   : YOLO confidence threshold for person detections
        padding       : fractional bounding-box padding passed to detect_and_crop

    Forward input:
        pil_sequences : list[list[PIL.Image]] — outer list = batch, inner = frames
        lengths       : (B,) LongTensor of actual sequence lengths

    Forward output:
        Same as ImprovedSeqToSeqReIDModel.forward — embeddings (B, embedding_dim)
        or (embeddings, logits, stripe_feats) during training when num_classes is set.
    """

    def __init__(self, detector, reid_model, reid_transform,
                 conf_thresh=0.4, padding=0.05):
        super().__init__()
        self.reid_model     = reid_model
        self.reid_transform = reid_transform
        self.conf_thresh    = conf_thresh
        self.padding        = padding
        # Bypass nn.Module.__setattr__ so the detector is NOT registered as a
        # submodule — its weights stay out of parameters() and state_dict().
        object.__setattr__(self, '_detector', detector)

    @property
    def detector(self):
        return self._detector

    def _crop(self, pil_image):
        from helper_functions.utils import detect_and_crop
        return detect_and_crop(pil_image, self._detector,
                               conf_thresh=self.conf_thresh,
                               padding=self.padding)

    def train(self, mode=True):
        """Keep detector permanently in eval mode."""
        super().train(mode)
        if hasattr(self._detector, 'model'):
            self._detector.model.eval()
        return self

    def forward(self, pil_sequences, lengths):
        device = next(self.reid_model.parameters()).device

        batch = []
        for seq_pils in pil_sequences:
            frames = [self.reid_transform(self._crop(f)) for f in seq_pils]
            batch.append(torch.stack(frames))

        max_t   = max(t.shape[0] for t in batch)
        C, H, W = batch[0].shape[1:]
        padded  = torch.zeros(len(batch), max_t, C, H, W, device=device)
        for i, t in enumerate(batch):
            padded[i, :t.shape[0]] = t.to(device)

        return self.reid_model(padded, lengths)


class TemporalStripeConsistencyLoss(nn.Module):
    def __init__(self, margin=0.3, lambda_inter=0.5, margin_intra=0.5):
        super().__init__()
        self.margin       = margin
        self.lambda_inter = lambda_inter
        # Only penalise frame pairs whose stripe similarity drops BELOW margin_intra.
        # Pairs above the margin (natural temporal variation) are ignored, making
        # the loss adaptive to challenging datasets like MARS.
        self.margin_intra = margin_intra

    def forward(self, stripe_feats, lengths):
        # stripe_feats: (B, T, 4, D)
        B, T, S, D = stripe_feats.shape
        stripe_feats = F.normalize(stripe_feats, dim=-1)

        intra_loss = stripe_feats.new_zeros(())
        inter_loss = stripe_feats.new_zeros(())
        n_valid = 0

        for b in range(B):
            L = min(int(lengths[b].item()), T)
            if L < 2:
                continue
            feats = stripe_feats[b, :L]   # (L, S, D)

            # Intra-stripe: only penalise frame pairs below margin_intra.
            for s in range(S):
                sf  = feats[:, s, :]
                sim = sf @ sf.T
                off = ~torch.eye(L, dtype=torch.bool, device=sim.device)
                intra_loss = intra_loss + F.relu(self.margin_intra - sim[off]).mean()

            # Inter-stripe: different stripes in same frame should be dissimilar
            sim_t = torch.bmm(feats, feats.transpose(1, 2))              # (L, S, S)
            off_s = ~torch.eye(S, dtype=torch.bool, device=feats.device)
            inter_mean = sim_t[:, off_s].mean()
            inter_loss = inter_loss + F.relu(inter_mean - self.margin)

            n_valid += 1

        if n_valid == 0:
            return stripe_feats.sum() * 0.0

        return intra_loss / (n_valid * S) + self.lambda_inter * inter_loss / n_valid


class CombinedReIDLoss(nn.Module):
    def __init__(self, margin=0.3, circle_m=0.25, circle_gamma=80, lambda_circle=0.2,
                 lambda_stripe=0.3, am_scale=30.0, am_margin=0.35):
        super().__init__()

        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.circle_m      = circle_m
        self.circle_gamma  = circle_gamma
        self.lambda_circle = lambda_circle
        self.lambda_stripe = lambda_stripe
        self.stripe_loss   = TemporalStripeConsistencyLoss(margin=0.3, lambda_inter=0.5,
                                                           margin_intra=0.5)
        # AMSoftmax parameters
        self.am_scale  = am_scale
        self.am_margin = am_margin

    def forward(self, embeddings, labels, logits=None, stripe_feats=None, lengths=None):
        loss = 0.0

        # 1. ID loss (AMSoftmax)
        if logits is not None:
            # Subtract margin from the correct-class cosine similarity, then scale.
            # Equivalent to CosFace: encourages a compact, well-separated hypersphere.
            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(1, labels.unsqueeze(1), self.am_margin)
            logits_am = self.am_scale * (logits - one_hot)
            id_loss = self.ce(logits_am, labels)
            loss += id_loss
        else:
            id_loss = torch.tensor(0.0)

        # 2. Triplet loss (batch-hard)
        dist = torch.cdist(embeddings, embeddings)

        eye      = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        mask_pos = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~eye
        mask_neg = ~(labels.unsqueeze(1) == labels.unsqueeze(0))

        hardest_pos = (dist * mask_pos.float()).max(dim=1)[0]
        hardest_neg = (dist + 1e5 * mask_pos.float()).min(dim=1)[0]

        triplet_loss = F.relu(hardest_pos - hardest_neg + 0.3).mean()
        loss += triplet_loss

        # 3. Circle loss
        sim = F.linear(F.normalize(embeddings), F.normalize(embeddings))

        pos_mask = mask_pos & ~eye
        neg_mask = mask_neg

        delta_p = 1 - self.circle_m
        delta_n = self.circle_m

        NEG_INF = float('-inf')
        logit_p_mat = torch.full_like(sim, NEG_INF)
        logit_n_mat = torch.full_like(sim, NEG_INF)

        sp = sim[pos_mask]
        sn = sim[neg_mask]
        ap = torch.clamp_min(-sp.detach() + 1 + self.circle_m, min=0.)
        an = torch.clamp_min( sn.detach()     + self.circle_m, min=0.)

        logit_p_mat[pos_mask] = -self.circle_gamma * ap * (sp - delta_p)
        logit_n_mat[neg_mask] =  self.circle_gamma * an * (sn - delta_n)

        lse_p = torch.logsumexp(logit_p_mat, dim=1)
        lse_n = torch.logsumexp(logit_n_mat, dim=1)

        valid = (lse_p > NEG_INF) & (lse_n > NEG_INF)
        circle_loss = F.softplus(lse_n[valid] + lse_p[valid]).mean() \
                      if valid.any() else embeddings.sum() * 0.0

        loss += self.lambda_circle * circle_loss

        # 4. Temporal stripe consistency loss
        if stripe_feats is not None and lengths is not None:
            stripe_loss = self.stripe_loss(stripe_feats, lengths)
        else:
            stripe_loss = torch.tensor(0.0)
        loss += self.lambda_stripe * stripe_loss

        return loss, {
            "id_loss":      id_loss,
            "triplet_loss": triplet_loss,
            "circle_loss":  circle_loss,
            "stripe_loss":  stripe_loss,
        }
