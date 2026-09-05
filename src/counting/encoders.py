"""Frozen DINOv2 + CLIP/Talk2DINO, independent of MMSegmentation."""

import logging
from pathlib import Path

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)
DINO_REPOSITORY = 'facebookresearch/dinov2:85a24602099d397264d5b30461ad7f3bfd726ca1'


class CLIPTextEncoder(nn.Module):
    """Keep CLIP's text computation without its dtype dependency on visual.conv1."""

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, tokens):
        dtype = self.token_embedding.weight.dtype
        x = self.token_embedding(tokens)
        x = x + self.positional_embedding.to(dtype)
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.ln_final(x).to(dtype)
        # CLIP encodes the end-of-text token, whose vocabulary ID is the largest.
        return x[torch.arange(x.shape[0], device=x.device), tokens.argmax(-1)] @ self.text_projection


class Talk2DINOProjection(nn.Module):
    """ViT-B projection layout compatible with the official released checkpoint.

    Based on Talk2DINO ProjectionLayer (Apache-2.0); see THIRD_PARTY_NOTICES.md.
    """

    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(512, 768)
        self.hidden_layers = nn.ModuleList([nn.Linear(768, 768)])

    def forward(self, text):
        return self.hidden_layers[0](self.linear_layer(text.float()).tanh())

    def load_weights(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'Talk2DINO ViT-B projection checkpoint missing: {path}')
        state = torch.load(path, map_location='cpu', weights_only=True)
        if 'linear_layer2.weight' in state:
            state['hidden_layers.0.weight'] = state.pop('linear_layer2.weight')
            state['hidden_layers.0.bias'] = state.pop('linear_layer2.bias')
        try:
            self.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise ValueError(f'Incompatible Talk2DINO checkpoint {path}; '
                             'expected vitb_mlp_infonce (512 -> 768 -> 768)') from error


class FrozenTextImageEncoder(nn.Module):
    def __init__(self, projection_weights, dino_repo=None, dino_weights=None,
                 clip_weights='ViT-B/16', download_root=None):
        super().__init__()
        try:
            try:
                import clip
            except ModuleNotFoundError as error:
                # Preserve errors for CLIP's own dependencies so the missing package is visible.
                if error.name != 'clip':
                    raise
                raise ModuleNotFoundError(
                    'OpenAI CLIP is missing from the Python environment running training. '
                    'Install with the same interpreter: '
                    '`python -m pip install git+https://github.com/openai/CLIP.git`. '
                    'For a local source checkout: `python -m pip install /path/to/CLIP`. '
                    'The ViT-B-16.pt checkpoint does not include the Python package. '
                    'See docs/FSC147_COUNTING.md.', name='clip') from error

            self.projection = Talk2DINOProjection()
            self.projection.load_weights(projection_weights)
            if dino_repo and not Path(dino_repo, 'hubconf.py').is_file():
                raise FileNotFoundError(f'DINOv2 repository needs hubconf.py: {dino_repo}')
            if dino_weights and not Path(dino_weights).is_file():
                raise FileNotFoundError(f'DINOv2 checkpoint missing: {dino_weights}')
            self.dino = torch.hub.load(
                str(dino_repo) if dino_repo else DINO_REPOSITORY,
                'dinov2_vitb14_reg', source='local' if dino_repo else 'github',
                pretrained=not bool(dino_weights))
            if dino_weights:
                self.dino.load_state_dict(torch.load(dino_weights, map_location='cpu',
                                                     weights_only=True), strict=True)
            clip_model, _ = clip.load(str(clip_weights), device='cpu', jit=False,
                                     download_root=download_root)
            # Counting uses only language from CLIP; discard its unused visual encoder.
            self.clip = CLIPTextEncoder(clip_model)
            self.tokenize = clip.tokenize
            self.requires_grad_(False).eval()
        except Exception:
            LOGGER.exception('FrozenTextImageEncoder initialization failed; projection=%s '
                             'dino_repo=%s dino_weights=%s clip_weights=%s',
                             projection_weights, dino_repo, dino_weights, clip_weights)
            raise

    def forward(self, images, texts):
        if images.shape[-2] % 14 or images.shape[-1] % 14:
            raise ValueError('DINOv2 image height and width must be divisible by 14')
        tokens = self.dino.forward_features(images)['x_norm_patchtokens']
        b, n, c = tokens.shape
        h, w = images.shape[-2] // 14, images.shape[-1] // 14
        if n != h * w or c != 768:
            raise ValueError(f'Expected {h*w} 768D patch tokens, got {tuple(tokens.shape)}')
        text_tokens = self.tokenize(list(texts), truncate=False).to(images.device)
        text_features = self.projection(self.clip(text_tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w), text_features
