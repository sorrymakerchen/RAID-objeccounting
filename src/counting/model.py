"""RAID's expert blocks with text-guided, per-image retrieval."""

import torch
import torch.nn.functional as F
from torch import nn

from src.expert import Expert_layer1, Expert_layer2


class TextGuidedRetrieval(nn.Module):
    def __init__(self, k=150, temperature=0.1):
        super().__init__()
        if k < 1 or temperature <= 0:
            raise ValueError('Retrieval k and temperature must be positive')
        self.k, self.temperature = k, temperature

    def forward(self, features, text):
        b, c, h, w = features.shape
        if text.shape != (b, c) or self.k > h * w:
            raise ValueError(f'Retrieval dimensions incompatible: features={features.shape}, '
                             f'text={text.shape}, k={self.k}')
        query = F.normalize(features.flatten(2).transpose(1, 2).float(), dim=-1)
        text = F.normalize(text.float(), dim=-1)
        similarity = torch.einsum('bnc,bc->bn', query, text)
        # Stable ordering makes tied scores reproducible across repeated evaluation.
        indices = similarity.argsort(dim=1, descending=True, stable=True)[:, :self.k]
        candidates = query.gather(1, indices.unsqueeze(-1).expand(-1, -1, c))
        matching = query @ candidates.transpose(1, 2)
        reconstructed = (matching / self.temperature).softmax(-1) @ candidates
        return {'query': query.transpose(1, 2).reshape(b, c, h, w),
                'reconstructed': reconstructed.transpose(1, 2).reshape(b, c, h, w),
                'matching': matching.transpose(1, 2).reshape(b, self.k, h, w),
                'similarity': similarity.reshape(b, 1, h, w), 'indices': indices}


class CountingRouter(nn.Module):
    def __init__(self, channels, experts):
        super().__init__()
        self.conv = nn.Conv2d(channels, experts, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        scores = self.conv(x).mean((2, 3))
        if self.training:
            scores = scores + torch.rand_like(scores) * 0.01
        return scores.softmax(-1)


class CountingMoE(nn.Module):
    """Reuse RAID experts without changing legacy anomaly-detection routing."""

    def __init__(self, input_dim=769, expert_dim=384, k=150, experts=3, warmup_epochs=30):
        super().__init__()
        if experts < 2 or k < 10 or k % 10 or warmup_epochs < 0:
            raise ValueError('Counting MoE requires experts>=2, k divisible by 10, warmup>=0')
        self.experts, self.warmup_epochs = experts, warmup_epochs
        self.router1 = CountingRouter(input_dim, experts)
        self.router2 = CountingRouter(input_dim + expert_dim + k, experts)
        self.models1 = nn.ModuleList([Expert_layer1(input_dim, expert_dim)
                                     for _ in range(experts)])
        self.models2 = nn.ModuleList([Expert_layer2(0, input_dim + expert_dim, k,
                                                   use_sigmoid=False)
                                     for _ in range(experts)])
        for expert in self.models2:
            # Start with modest density per cell, avoiding an initial ~700-object bias.
            nn.init.constant_(expert.head_conv[-1].bias, -3.0)

    @staticmethod
    def top2_weights(probabilities):
        indices = probabilities.argsort(dim=-1, descending=True, stable=True)[:, :2]
        mask = torch.zeros_like(probabilities).scatter_(1, indices, 1)
        selected = probabilities * mask
        return selected / selected.sum(-1, keepdim=True).clamp_min(1e-8)

    def forward(self, guidance, matching, epoch=0):
        probabilities = self.router1(guidance)
        weights = self.top2_weights(probabilities)
        outputs = []
        for index, expert in enumerate(self.models1):
            # Dispatch whole selected samples; zero-weight experts must not add biases.
            selected = (weights[:, index] > 0).nonzero(as_tuple=True)[0]
            routed = guidance.new_zeros((guidance.shape[0],
                                         guidance.shape[1] + expert.out_dim,
                                         *guidance.shape[-2:]))
            if selected.numel():
                routed = routed.index_copy(0, selected, expert(guidance[selected]))
            outputs.append(routed * weights[:, index, None, None, None])
        combined = torch.stack(outputs).sum(0)
        usage = (weights.detach() > 0).float().mean(0) / 2
        balance = self.experts * (usage * probabilities.mean(0)).sum()
        if epoch < self.warmup_epochs:
            weights2 = guidance.new_full((guidance.shape[0], self.experts), 1 / self.experts)
        else:
            weights2 = self.router2(torch.cat([combined, matching], dim=1))
            balance = balance + self.experts * weights2.mean(0).square().sum()
        logits = torch.stack([expert(combined, matching)[0] for expert in self.models2], dim=1)
        logits = (logits * weights2.unsqueeze(-1)).sum(1)
        return logits, balance


class RAIDCounter(nn.Module):
    def __init__(self, encoder, feature_dim=768, reduced_dim=384, k=150,
                 expert_dim=384, warmup_epochs=30):
        super().__init__()
        self.encoder = encoder.requires_grad_(False).eval()
        self.retrieval = TextGuidedRetrieval(k)
        self.projection = nn.Conv2d(feature_dim, reduced_dim, 1)
        self.moe = CountingMoE(reduced_dim * 2 + 1, expert_dim, k,
                               warmup_epochs=warmup_epochs)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, images, texts, epoch=0):
        if isinstance(texts, str) or len(texts) != len(images) or any(
                not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError('Each image requires one non-empty text prompt')
        with torch.no_grad():
            features, text = self.encoder(images, texts)
            retrieved = self.retrieval(features, text)
        guidance = torch.cat([self.projection(retrieved['query']),
                              self.projection(retrieved['reconstructed']),
                              retrieved['similarity']], dim=1)
        logits, balance = self.moe(guidance, retrieved['matching'], epoch)
        density = F.softplus(logits).reshape(images.shape[0], 1, *features.shape[-2:])
        return {'density': density, 'count': density.sum((1, 2, 3)),
                'similarity': retrieved['similarity'], 'indices': retrieved['indices'],
                'balance_loss': balance}
