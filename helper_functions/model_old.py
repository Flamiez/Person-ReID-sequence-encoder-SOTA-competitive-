import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class FeatureExtractModel(nn.Module):
    def __init__(self, embedding_dim=1024):
        super(FeatureExtractModel, self).__init__()
        googlenet = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
        
        self.base = nn.Sequential(
            googlenet.conv1,
            googlenet.maxpool1,
            googlenet.conv2,
            googlenet.conv3,
            googlenet.maxpool2,
            googlenet.inception3a,
            googlenet.inception3b,
            googlenet.maxpool3,
            googlenet.inception4a,
            googlenet.inception4b,
            googlenet.inception4c,
            googlenet.inception4d,
            googlenet.inception4e,
            googlenet.maxpool4,
            googlenet.inception5a,
            googlenet.inception5b,
            googlenet.avgpool
        )
        
        self._add_bn_layers()
        self.embedding = nn.Sequential(
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True)
        )
        self._initialize_weights()
    
    def _add_bn_layers(self):
        for name, module in self.base.named_children():
            if 'inception' in name:
                module.add_module('bn', nn.BatchNorm2d(module.output_channels))
    
    def _initialize_weights(self):
        for m in self.embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        embedding = self.embedding(x)
        norm_embedding = nn.functional.normalize(embedding, p=2, dim=1)
        
        return norm_embedding 
    
class TemporalAttention(nn.Module):
    """Scalar dot-product attention over time steps to pool RNN outputs."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, rnn_out, mask=None):
        # rnn_out: (B, T, H)
        scores = self.attn(rnn_out).squeeze(-1)          # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        weights = F.softmax(scores, dim=-1).unsqueeze(-1) # (B, T, 1)
        return (weights * rnn_out).sum(dim=1)             # (B, H)


class SequenceEncoder(nn.Module):
    """
    Encodes a variable-length image sequence into a fixed-dim embedding.

    Per-frame:  GoogLeNet backbone → 1024-dim features → projection
    Temporal:   Bidirectional GRU → temporal attention pooling → L2-norm
    """
    def __init__(self, frame_feat_dim=1024, rnn_hidden=512, rnn_layers=2,
                 embedding_dim=512, dropout=0.3):
        super().__init__()

        googlenet = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(
            googlenet.conv1, googlenet.maxpool1,
            googlenet.conv2, googlenet.conv3, googlenet.maxpool2,
            googlenet.inception3a, googlenet.inception3b, googlenet.maxpool3,
            googlenet.inception4a, googlenet.inception4b,
            googlenet.inception4c, googlenet.inception4d, googlenet.inception4e,
            googlenet.maxpool4,
            googlenet.inception5a, googlenet.inception5b,
            googlenet.avgpool
        )

        self.frame_proj = nn.Sequential(
            nn.Linear(frame_feat_dim, rnn_hidden),
            nn.LayerNorm(rnn_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.rnn = nn.GRU(
            input_size=rnn_hidden,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0
        )

        self.attention = TemporalAttention(rnn_hidden * 2)

        self.embed_head = nn.Sequential(
            nn.Linear(rnn_hidden * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in list(self.frame_proj.modules()) + list(self.embed_head.modules()):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, lengths):
        """
        Args:
            x:       (B, T, C, H, W)  padded image sequences
            lengths: (B,)              actual sequence lengths (LongTensor)
        Returns:
            (B, embedding_dim)  L2-normalized sequence embeddings
        """
        B, T, C, H, W = x.shape

        # Per-frame CNN features
        frame_feats = self.cnn(x.view(B * T, C, H, W))   # (B*T, 1024, 1, 1)
        frame_feats = frame_feats.view(B * T, -1)          # (B*T, 1024)
        frame_feats = self.frame_proj(frame_feats)          # (B*T, rnn_hidden)
        frame_feats = frame_feats.view(B, T, -1)            # (B, T, rnn_hidden)

        # Pack → bidir GRU → unpack
        packed = pack_padded_sequence(frame_feats, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        rnn_out, _ = self.rnn(packed)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)  # (B, T', 2H)

        # Padding mask for attention
        max_len = rnn_out.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < \
               lengths.to(x.device).unsqueeze(1)              # (B, T')

        pooled = self.attention(rnn_out, mask)                # (B, 2H)
        embedding = self.embed_head(pooled)                   # (B, embedding_dim)
        return F.normalize(embedding, p=2, dim=1)


class SequenceEncoderRNN(nn.Module):
    """
    RNN-only sequence encoder — expects pre-extracted CNN frame features.

    Input:  (B, T, frame_feat_dim)  pre-computed per-frame feature vectors
    Output: (B, embedding_dim)      L2-normalised sequence embeddings
    """
    def __init__(self, frame_feat_dim=1024, rnn_hidden=512, rnn_layers=2,
                 embedding_dim=512, dropout=0.3):
        super().__init__()

        self.frame_proj = nn.Sequential(
            nn.Linear(frame_feat_dim, rnn_hidden),
            nn.LayerNorm(rnn_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.rnn = nn.GRU(
            input_size=rnn_hidden,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0
        )

        self.attention = TemporalAttention(rnn_hidden * 2)

        self.embed_head = nn.Sequential(
            nn.Linear(rnn_hidden * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in list(self.frame_proj.modules()) + list(self.embed_head.modules()):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, lengths):
        """
        Args:
            x:       (B, T, frame_feat_dim)  pre-extracted frame features
            lengths: (B,)                    actual sequence lengths
        Returns:
            (B, embedding_dim)  L2-normalized embeddings
        """
        B, T, _ = x.shape

        proj = self.frame_proj(x.view(B * T, -1)).view(B, T, -1)  # (B, T, rnn_hidden)

        packed = pack_padded_sequence(proj, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        rnn_out, _ = self.rnn(packed)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)  # (B, T', 2H)

        max_len = rnn_out.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < \
               lengths.to(x.device).unsqueeze(1)

        pooled = self.attention(rnn_out, mask)
        embedding = self.embed_head(pooled)
        return F.normalize(embedding, p=2, dim=1)


class SeqToSeqReIDModelRNN(nn.Module):
    """SeqToSeqReIDModel that operates on pre-extracted CNN features (no CNN forward pass)."""
    def __init__(self, frame_feat_dim=1024, rnn_hidden=512, rnn_layers=2,
                 embedding_dim=512, dropout=0.3):
        super().__init__()
        self.encoder = SequenceEncoderRNN(
            frame_feat_dim=frame_feat_dim,
            rnn_hidden=rnn_hidden,
            rnn_layers=rnn_layers,
            embedding_dim=embedding_dim,
            dropout=dropout
        )

    def encode(self, x, lengths):
        return self.encoder(x, lengths)

    def forward(self, query_feats, query_lengths, gallery_feats=None, gallery_lengths=None):
        query_emb = self.encoder(query_feats, query_lengths)
        gallery_emb = self.encoder(gallery_feats, gallery_lengths) \
            if gallery_feats is not None else None
        return query_emb, gallery_emb


class SequenceEncoderE2E(nn.Module):
    """
    End-to-end sequence encoder with fine-tunable ResNet-50 backbone.

    Early layers (layer1, layer2) are frozen; layer3, layer4, and the RNN head
    are trained. This lets the backbone adapt to ReID while keeping training stable.

    Input:  (B, T, C, H, W)  raw image sequences
    Output: (B, embedding_dim)  L2-normalised embeddings
    """
    def __init__(self, rnn_hidden=512, rnn_layers=2, embedding_dim=512, dropout=0.3):
        super().__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Freeze stem + layer1 + layer2; train layer3 + layer4
        for name, param in resnet.named_parameters():
            freeze = not any(k in name for k in ('layer3', 'layer4'))
            param.requires_grad = not freeze

        self.cnn = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            resnet.avgpool
        )
        frame_feat_dim = 2048  # ResNet-50 avgpool output

        self.frame_proj = nn.Sequential(
            nn.Linear(frame_feat_dim, rnn_hidden),
            nn.LayerNorm(rnn_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.rnn = nn.GRU(
            input_size=rnn_hidden,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0
        )

        self.attention = TemporalAttention(rnn_hidden * 2)

        self.embed_head = nn.Sequential(
            nn.Linear(rnn_hidden * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in list(self.frame_proj.modules()) + list(self.embed_head.modules()):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, lengths):
        """
        Args:
            x:       (B, T, C, H, W)  padded image sequences
            lengths: (B,)              actual sequence lengths
        Returns:
            (B, embedding_dim)  L2-normalized embeddings
        """
        B, T, C, H, W = x.shape

        frame_feats = self.cnn(x.view(B * T, C, H, W)).view(B * T, -1)  # (B*T, 2048)
        proj = self.frame_proj(frame_feats).view(B, T, -1)               # (B, T, H)

        packed = pack_padded_sequence(proj, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        rnn_out, _ = self.rnn(packed)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)

        max_len = rnn_out.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < \
               lengths.to(x.device).unsqueeze(1)

        pooled = self.attention(rnn_out, mask)
        embedding = self.embed_head(pooled)
        return F.normalize(embedding, p=2, dim=1)


class SeqToSeqReIDModelE2E(nn.Module):
    """
    End-to-end ReID model: ResNet-50 backbone + GRU + attention.

    When num_classes is provided, adds an ID classification head used only
    during training (triplet loss + cross-entropy ID loss).  At inference,
    call forward() without labels — only the L2-normalised embedding is returned.
    """
    def __init__(self, rnn_hidden=512, rnn_layers=2, embedding_dim=512,
                 dropout=0.3, num_classes=None):
        super().__init__()
        self.encoder = SequenceEncoderE2E(
            rnn_hidden=rnn_hidden,
            rnn_layers=rnn_layers,
            embedding_dim=embedding_dim,
            dropout=dropout
        )
        # Classification head — only used during training
        self.classifier = nn.Linear(embedding_dim, num_classes) \
            if num_classes is not None else None

    def forward(self, x, lengths):
        embedding = self.encoder(x, lengths)          # (B, embedding_dim), L2-normed
        if self.classifier is not None and self.training:
            logits = self.classifier(embedding)       # (B, num_classes)
            return embedding, logits
        return embedding


class SequenceEncoderViT(nn.Module):
    """
    Sequence encoder with ViT-B/16 backbone.

    All transformer blocks frozen except the last 2 (blocks 10 & 11) and
    the final LayerNorm, which are fine-tuned at a low LR alongside the RNN head.

    Input:  (B, T, C, H, W)  raw image sequences (224×224)
    Output: (B, embedding_dim)  L2-normalised embeddings
    """
    def __init__(self, rnn_hidden=512, rnn_layers=2, embedding_dim=512, dropout=0.3):
        super().__init__()

        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        vit.heads = nn.Identity()          # drop classification head → 768-dim output

        # Freeze all, then unfreeze last 2 encoder blocks + final LN
        for param in vit.parameters():
            param.requires_grad = False
        for i in [10, 11]:
            for param in vit.encoder.layers[i].parameters():
                param.requires_grad = True
        for param in vit.encoder.ln.parameters():
            param.requires_grad = True

        self.vit = vit
        frame_feat_dim = 768  # ViT-B/16 CLS token dim

        self.frame_proj = nn.Sequential(
            nn.Linear(frame_feat_dim, rnn_hidden),
            nn.LayerNorm(rnn_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.rnn = nn.GRU(
            input_size=rnn_hidden,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0
        )

        self.attention = TemporalAttention(rnn_hidden * 2)

        self.embed_head = nn.Sequential(
            nn.Linear(rnn_hidden * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in list(self.frame_proj.modules()) + list(self.embed_head.modules()):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, lengths):
        B, T, C, H, W = x.shape

        frame_feats = self.vit(x.view(B * T, C, H, W))          # (B*T, 768)
        proj = self.frame_proj(frame_feats).view(B, T, -1)       # (B, T, rnn_hidden)

        packed = pack_padded_sequence(proj, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        rnn_out, _ = self.rnn(packed)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)

        max_len = rnn_out.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < \
               lengths.to(x.device).unsqueeze(1)

        pooled = self.attention(rnn_out, mask)
        embedding = self.embed_head(pooled)
        return F.normalize(embedding, p=2, dim=1)


class SeqToSeqReIDModelViT(nn.Module):
    """ReID model with ViT-B/16 backbone + BiGRU + temporal attention."""
    def __init__(self, rnn_hidden=512, rnn_layers=2, embedding_dim=512, dropout=0.3):
        super().__init__()
        self.encoder = SequenceEncoderViT(
            rnn_hidden=rnn_hidden,
            rnn_layers=rnn_layers,
            embedding_dim=embedding_dim,
            dropout=dropout
        )

    def forward(self, x, lengths):
        return self.encoder(x, lengths)


class SeqToSeqReIDModel(nn.Module):
    """
    End-to-end sequence-to-sequence person re-identification model.

    Encodes variable-length image sequences (query and gallery) into a shared
    L2-normalised embedding space where matching persons have similar embeddings.

    Typical training usage (triplet / contrastive loss on returned embeddings):
        q_emb, g_emb = model(query_seqs, query_lengths, gallery_seqs, gallery_lengths)

    Inference (encode one batch of sequences):
        emb = model.encode(seqs, lengths)
    """
    def __init__(self, frame_feat_dim=1024, rnn_hidden=512, rnn_layers=2,
                 embedding_dim=512, dropout=0.3):
        super().__init__()
        self.encoder = SequenceEncoder(
            frame_feat_dim=frame_feat_dim,
            rnn_hidden=rnn_hidden,
            rnn_layers=rnn_layers,
            embedding_dim=embedding_dim,
            dropout=dropout
        )

    def encode(self, x, lengths):
        """Encode a batch of padded image sequences → normalised embeddings."""
        return self.encoder(x, lengths)

    def forward(self, query_seqs, query_lengths, gallery_seqs=None, gallery_lengths=None):
        """
        Args:
            query_seqs:      (B, T_q, C, H, W)
            query_lengths:   (B,)
            gallery_seqs:    (G, T_g, C, H, W)  — optional at inference time
            gallery_lengths: (G,)
        Returns:
            query_emb:   (B, embedding_dim)
            gallery_emb: (G, embedding_dim) or None
        """
        query_emb = self.encoder(query_seqs, query_lengths)
        gallery_emb = self.encoder(gallery_seqs, gallery_lengths) \
            if gallery_seqs is not None else None
        return query_emb, gallery_emb
    

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


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

    def forward(self, x, mask=None):
        return self.encoder(x)


class ImprovedSequenceEncoder(nn.Module):
    def __init__(self, rnn_hidden=512, embedding_dim=512, dropout=0.3):
        super().__init__()

        # 🔷 1. Backbone (stronger than plain ResNet)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        for name, param in resnet.named_parameters():
            param.requires_grad = "layer3" in name or "layer4" in name

        self.cnn = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )

        self.pool = nn.AdaptiveAvgPool2d((8, 4))  # keep spatial info

        # 🔷 2. Local + global projection
        self.global_proj = nn.Linear(2048, rnn_hidden)
        self.local_proj = nn.Linear(2048, rnn_hidden)

        # 🔷 3. Temporal Transformer
        self.temporal_tf = TemporalTransformer(rnn_hidden * 5)

        # 🔷 4. GRU (same as yours)
        self.rnn = nn.GRU(
            input_size=rnn_hidden * 5,
            hidden_size=rnn_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # 🔷 5. Attention pooling
        self.attention = TemporalAttention(rnn_hidden * 2)

        # 🔷 6. Embedding head
        self.embed_head = nn.Sequential(
            nn.Linear(rnn_hidden * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x, lengths):
        B, T, C, H, W = x.shape

        # 🔷 Frame feature maps
        fmap = self.cnn(x.view(B * T, C, H, W))   # (B*T, 2048, H, W)
        fmap = self.pool(fmap)                    # (B*T, 2048, 8, 4)

        # 🔷 Global feature
        global_feat = fmap.mean(dim=[2, 3])       # (B*T, 2048)

        # 🔷 Local features (4 horizontal stripes)
        stripes = torch.chunk(fmap, 4, dim=2)
        local_feats = [s.mean(dim=[2, 3]) for s in stripes]

        # 🔷 Project
        g = self.global_proj(global_feat)
        locals_ = [self.local_proj(f) for f in local_feats]

        # 🔷 Combine
        feat = torch.cat([g] + locals_, dim=1)    # (B*T, 5*hidden)
        feat = feat.view(B, T, -1)

        # 🔷 Temporal Transformer
        feat = self.temporal_tf(feat)

        # 🔷 GRU
        packed = nn.utils.rnn.pack_padded_sequence(
            feat, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        rnn_out, _ = self.rnn(packed)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)

        # 🔷 Mask
        max_len = rnn_out.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)

        # 🔷 Attention pooling
        pooled = self.attention(rnn_out, mask)

        # 🔷 Embedding
        embedding = self.embed_head(pooled)

        return F.normalize(embedding, dim=1)


class ImprovedSeqToSeqReIDModel(nn.Module):
    def __init__(self, embedding_dim=512, num_classes=None):
        super().__init__()

        self.encoder = ImprovedSequenceEncoder(embedding_dim=embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes) \
            if num_classes is not None else None

    def forward(self, x, lengths):
        embedding = self.encoder(x, lengths)

        if self.classifier is not None and self.training:
            logits = self.classifier(embedding)
            return embedding, logits

        return embedding


class CombinedReIDLoss(nn.Module):
    def __init__(self, margin=0.3, circle_m=0.25, circle_gamma=256, lambda_circle=0.5):
        super().__init__()

        self.triplet = nn.TripletMarginLoss(margin=margin)
        self.ce = nn.CrossEntropyLoss()

        self.circle_m = circle_m
        self.circle_gamma = circle_gamma
        self.lambda_circle = lambda_circle

    def forward(self, embeddings, labels, logits=None):
        loss = 0.0

        # 1. ID loss
        if logits is not None:
            id_loss = self.ce(logits, labels)
            loss += id_loss
        else:
            id_loss = torch.tensor(0.0)

        # 2. Triplet loss (batch-hard)
        dist = torch.cdist(embeddings, embeddings)

        mask_pos = labels.unsqueeze(1) == labels.unsqueeze(0)
        mask_neg = ~mask_pos

        hardest_pos = (dist * mask_pos.float()).max(dim=1)[0]
        hardest_neg = (dist + 1e5 * mask_pos.float()).min(dim=1)[0]

        triplet_loss = F.relu(hardest_pos - hardest_neg + 0.3).mean()
        loss += triplet_loss

        # 3. Circle loss
        sim = F.linear(F.normalize(embeddings), F.normalize(embeddings))

        pos_mask = mask_pos.float() - torch.eye(len(labels), device=labels.device)
        neg_mask = mask_neg.float()

        sp = sim * pos_mask
        sn = sim * neg_mask

        ap = torch.clamp_min(-sp.detach() + 1 + self.circle_m, min=0.)
        an = torch.clamp_min(sn.detach() + self.circle_m, min=0.)

        delta_p = 1 - self.circle_m
        delta_n = self.circle_m

        logit_p = -self.circle_gamma * ap * (sp - delta_p)
        logit_n = self.circle_gamma * an * (sn - delta_n)

        circle_loss = (torch.logsumexp(logit_n, dim=1) +
                       torch.logsumexp(logit_p, dim=1)).mean()

        loss += self.lambda_circle * circle_loss

        return loss, {
            "id_loss": id_loss,
            "triplet_loss": triplet_loss,
            "circle_loss": circle_loss,
        }
    

####################


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.checkpoint import checkpoint_sequential


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

        # 🔷 1. Backbone (stronger than plain ResNet)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        for name, param in resnet.named_parameters():
            param.requires_grad = "layer3" in name or "layer4" in name

        self.cnn = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )

        self.pool = nn.AdaptiveAvgPool2d((8, 4))  # keep spatial info

        # 🔷 2. Local + global projection
        self.global_proj = nn.Linear(2048, rnn_hidden)
        self.local_proj = nn.Linear(2048, rnn_hidden)

        # 🔷 3. Temporal Transformer
        self.temporal_tf = TemporalTransformer(rnn_hidden * 5)

        # 🔷 4. GRU (same as yours)
        self.rnn = nn.GRU(
            input_size=rnn_hidden * 5,
            hidden_size=rnn_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # 🔷 5. Attention pooling
        self.attention = TemporalAttention(rnn_hidden * 2)

        # 🔷 6. Embedding head
        self.embed_head = nn.Sequential(
            nn.Linear(rnn_hidden * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x, lengths):
        B, T, C, H, W = x.shape

        # 🔷 Frame feature maps — checkpoint_sequential recomputes activations
        # during backward instead of storing them, saving ~60% activation memory.
        frames = x.view(B * T, C, H, W)
        fmap = checkpoint_sequential(self.cnn, segments=4, input=frames,
                                     use_reentrant=False)   # (B*T, 2048, H, W)
        fmap = self.pool(fmap)                    # (B*T, 2048, 8, 4)

        # 🔷 Global feature
        global_feat = fmap.mean(dim=[2, 3])       # (B*T, 2048)

        # 🔷 Local features (4 horizontal stripes)
        stripes = torch.chunk(fmap, 4, dim=2)
        local_feats = [s.mean(dim=[2, 3]) for s in stripes]

        # 🔷 Project
        g = self.global_proj(global_feat)
        locals_ = [self.local_proj(f) for f in local_feats]

        # 🔷 Combine
        feat = torch.cat([g] + locals_, dim=1)    # (B*T, 5*hidden)
        feat = feat.view(B, T, -1)

        # 🔷 Temporal Transformer (mask out padded frames)
        pad_mask = torch.arange(T, device=x.device).unsqueeze(0) >= \
                   lengths.to(x.device).unsqueeze(1)   # (B, T), True = padded
        feat = self.temporal_tf(feat, pad_mask=pad_mask)

        # 🔷 GRU
        packed = nn.utils.rnn.pack_padded_sequence(
            feat, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        rnn_out, _ = self.rnn(packed)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)

        # 🔷 Mask
        max_len = rnn_out.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1).to(x.device)

        # 🔷 Attention pooling
        pooled = self.attention(rnn_out, mask)

        # 🔷 Embedding
        embedding = self.embed_head(pooled)

        return F.normalize(embedding, dim=1)

class TemporalAttention(nn.Module):
    """Scalar dot-product attention over time steps to pool RNN outputs."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, rnn_out, mask=None):
        # rnn_out: (B, T, H)
        scores = self.attn(rnn_out).squeeze(-1)          # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        weights = F.softmax(scores, dim=-1).unsqueeze(-1) # (B, T, 1)
        return (weights * rnn_out).sum(dim=1)             # (B, H)


class ImprovedSeqToSeqReIDModel(nn.Module):
    def __init__(self, embedding_dim=512, rnn_hidden=256, num_classes=None):
        super().__init__()

        self.encoder = ImprovedSequenceEncoder(rnn_hidden=rnn_hidden,
                                               embedding_dim=embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes) \
            if num_classes is not None else None

    def forward(self, x, lengths):
        embedding = self.encoder(x, lengths)

        if self.classifier is not None and self.training:
            logits = self.classifier(embedding)
            return embedding, logits

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
        or (embeddings, logits) during training when num_classes is set.
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

    # ------------------------------------------------------------------ helpers
    @property
    def detector(self):
        return self._detector

    def _crop(self, pil_image):
        from helper_functions.utils import detect_and_crop
        return detect_and_crop(pil_image, self._detector,
                               conf_thresh=self.conf_thresh,
                               padding=self.padding)

    # --------------------------------------------------------------- nn.Module
    def train(self, mode=True):
        """Keep detector permanently in eval mode."""
        super().train(mode)
        if hasattr(self._detector, 'model'):
            self._detector.model.eval()
        return self

    def forward(self, pil_sequences, lengths):
        device = next(self.reid_model.parameters()).device

        # Detect → crop → transform every frame in the batch
        batch = []
        for seq_pils in pil_sequences:
            frames = [self.reid_transform(self._crop(f)) for f in seq_pils]
            batch.append(torch.stack(frames))          # (T_i, C, H, W)

        # Pad to longest sequence in this batch
        max_t      = max(t.shape[0] for t in batch)
        C, H, W    = batch[0].shape[1:]
        padded     = torch.zeros(len(batch), max_t, C, H, W, device=device)
        for i, t in enumerate(batch):
            padded[i, :t.shape[0]] = t.to(device)

        return self.reid_model(padded, lengths)


class CombinedReIDLoss(nn.Module):
    def __init__(self, margin=0.3, circle_m=0.25, circle_gamma=256, lambda_circle=0.5):
        super().__init__()

        self.triplet = nn.TripletMarginLoss(margin=margin)
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.circle_m = circle_m
        self.circle_gamma = circle_gamma
        self.lambda_circle = lambda_circle

    def forward(self, embeddings, labels, logits=None):
        loss = 0.0

        # 1. ID loss
        if logits is not None:
            id_loss = self.ce(logits, labels)
            loss += id_loss
        else:
            id_loss = torch.tensor(0.0)

        # 2. Triplet loss (batch-hard)
        dist = torch.cdist(embeddings, embeddings)

        eye      = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        mask_pos = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~eye  # exclude self
        mask_neg = ~(labels.unsqueeze(1) == labels.unsqueeze(0))        # exclude all same-class

        hardest_pos = (dist * mask_pos.float()).max(dim=1)[0]
        hardest_neg = (dist + 1e5 * mask_pos.float()).min(dim=1)[0]

        triplet_loss = F.relu(hardest_pos - hardest_neg + 0.3).mean()
        loss += triplet_loss

        # 3. Circle loss
        sim = F.linear(F.normalize(embeddings), F.normalize(embeddings))

        # pos_mask: same-class pairs excluding diagonal; neg_mask: different-class pairs
        eye      = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        pos_mask = mask_pos & ~eye
        neg_mask = mask_neg

        delta_p = 1 - self.circle_m
        delta_n = self.circle_m

        # Build (B, B) logit matrices; non-relevant entries stay at -inf so they
        # don't contribute to logsumexp (no masked-pair contamination).
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

        # Only average over anchors that have at least one positive and one negative
        valid = (lse_p > NEG_INF) & (lse_n > NEG_INF)
        circle_loss = F.softplus(lse_n[valid] + lse_p[valid]).mean() \
                      if valid.any() else embeddings.sum() * 0.0

        loss += self.lambda_circle * circle_loss

        return loss, {
            "id_loss": id_loss,
            "triplet_loss": triplet_loss,
            "circle_loss": circle_loss,
        }