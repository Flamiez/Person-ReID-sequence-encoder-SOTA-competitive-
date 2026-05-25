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

        self.vit = timm.create_model(
            'vit_base_patch14_dinov2.lvd142m',
            pretrained=True,
            img_size=(256, 128),
            num_classes=0,
        )
        self.patch_h = 256 // 14
        self.patch_w = 128 // 14
        vit_dim = 768

        for param in self.vit.parameters():
            param.requires_grad = True

        self.vit.set_grad_checkpointing(enable=True)

        self.global_proj  = nn.Linear(vit_dim, rnn_hidden)

        self.local_proj_2 = nn.Linear(vit_dim, rnn_hidden // 2)
        self.local_proj_4 = nn.Linear(vit_dim, rnn_hidden // 4)
        self.local_proj_8 = nn.Linear(vit_dim, rnn_hidden // 8)

        combined_dim = rnn_hidden * 4

        self.temporal_tf = TemporalTransformer(combined_dim)

        self.rnn = nn.GRU(
            input_size=combined_dim,
            hidden_size=rnn_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.attention = TemporalAttention(rnn_hidden * 2)

        self.embed_head = nn.Sequential(
            nn.Linear(rnn_hidden * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x, lengths):
        B, T, C, H, W = x.shape
        frames = x.view(B * T, C, H, W)

        tokens       = self.vit.forward_features(frames)
        cls_token    = tokens[:, 0]
        patch_tokens = tokens[:, 1:]

        patch_tokens = patch_tokens.view(B * T, self.patch_h, self.patch_w, -1)
        patch_tokens = patch_tokens[:, :16, :, :]

        stripes_2 = torch.chunk(patch_tokens, 2, dim=1)
        stripes_4 = torch.chunk(patch_tokens, 4, dim=1)
        stripes_8 = torch.chunk(patch_tokens, 8, dim=1)

        local_2 = [s.mean(dim=[1, 2]) for s in stripes_2]
        local_4 = [s.mean(dim=[1, 2]) for s in stripes_4]
        local_8 = [s.mean(dim=[1, 2]) for s in stripes_8]

        g        = self.global_proj(cls_token)
        locals_2 = [self.local_proj_2(f) for f in local_2]
        locals_4 = [self.local_proj_4(f) for f in local_4]
        locals_8 = [self.local_proj_8(f) for f in local_8]

        feat = torch.cat([g] + locals_2 + locals_4 + locals_8, dim=1)
        feat = feat.view(B, T, -1)

        stripe_feats = torch.stack(locals_4, dim=1).view(B, T, 4, -1)

        pad_mask = torch.arange(T, device=x.device).unsqueeze(0) >= \
                   lengths.to(x.device).unsqueeze(1)
        feat = self.temporal_tf(feat, pad_mask=pad_mask)

        packed  = nn.utils.rnn.pack_padded_sequence(
            feat, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        rnn_out, _ = self.rnn(packed)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)

        max_len = rnn_out.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < \
               lengths.unsqueeze(1).to(x.device)

        pooled    = self.attention(rnn_out, mask)
        embedding = self.embed_head(pooled)
        return F.normalize(embedding, dim=1), stripe_feats


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, rnn_out, mask=None):
        scores = self.attn(rnn_out).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        return (weights * rnn_out).sum(dim=1)


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
                normed_w = F.normalize(self.classifier.weight, dim=1)
                logits   = F.linear(embedding, normed_w)
            else:
                logits = None
            return embedding, logits, stripe_feats

        return embedding


class DetectThenReIDModel(nn.Module):

    def __init__(self, detector, reid_model, reid_transform,
                 conf_thresh=0.4, padding=0.05):
        super().__init__()
        self.reid_model     = reid_model
        self.reid_transform = reid_transform
        self.conf_thresh    = conf_thresh
        self.padding        = padding
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
        self.margin_intra = margin_intra

    def forward(self, stripe_feats, lengths):
        B, T, S, D = stripe_feats.shape
        stripe_feats = F.normalize(stripe_feats, dim=-1)

        intra_loss = stripe_feats.new_zeros(())
        inter_loss = stripe_feats.new_zeros(())
        n_valid = 0

        for b in range(B):
            L = min(int(lengths[b].item()), T)
            if L < 2:
                continue
            feats = stripe_feats[b, :L]

            for s in range(S):
                sf  = feats[:, s, :]
                sim = sf @ sf.T
                off = ~torch.eye(L, dtype=torch.bool, device=sim.device)
                intra_loss = intra_loss + F.relu(self.margin_intra - sim[off]).mean()

            sim_t = torch.bmm(feats, feats.transpose(1, 2))
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
        self.am_scale  = am_scale
        self.am_margin = am_margin

    def forward(self, embeddings, labels, logits=None, stripe_feats=None, lengths=None):
        loss = 0.0

        if logits is not None:
            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(1, labels.unsqueeze(1), self.am_margin)
            logits_am = self.am_scale * (logits - one_hot)
            id_loss = self.ce(logits_am, labels)
            loss += id_loss
        else:
            id_loss = torch.tensor(0.0)

        dist = torch.cdist(embeddings, embeddings)

        eye      = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        mask_pos = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~eye
        mask_neg = ~(labels.unsqueeze(1) == labels.unsqueeze(0))

        hardest_pos = (dist * mask_pos.float()).max(dim=1)[0]
        hardest_neg = (dist + 1e5 * mask_pos.float()).min(dim=1)[0]

        triplet_loss = F.relu(hardest_pos - hardest_neg + 0.3).mean()
        loss += triplet_loss

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
