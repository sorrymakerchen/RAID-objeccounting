"""Single-image counting and mass-preserving visualization."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset.fsc147 import image_tensor, resize_density


class CountingPredictor:
    def __init__(self, model, device='cpu', image_size=448, epoch=99):
        self.model = model.to(device).eval()
        self.device, self.image_size, self.epoch = device, image_size, epoch

    @torch.no_grad()
    def predict(self, image, text):
        if isinstance(image, (str, Path)):
            with Image.open(image) as source:
                image = source.convert('RGB')
        else:
            image = image.convert('RGB')
        tensor = image_tensor(image, self.image_size).unsqueeze(0).to(self.device)
        output = self.model(tensor, [text], self.epoch)
        if not torch.isfinite(output['density']).all():
            raise FloatingPointError('Prediction contains non-finite density')
        size = (image.height, image.width)
        density = resize_density(output['density'], size)[0, 0].cpu().numpy()
        similarity = F.interpolate(output['similarity'], size=size, mode='bilinear',
                                   align_corners=False)[0, 0].cpu().numpy()
        return {'count': float(output['count'][0]), 'density': density,
                'similarity': similarity, 'text': text}


def save_prediction(result, image, directory, stem='prediction'):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            image = source.convert('RGB')
    else:
        image = image.convert('RGB')
    np.save(directory / f'{stem}_density.npy', result['density'])
    np.save(directory / f'{stem}_similarity.npy', result['similarity'])
    base = np.asarray(image).astype(np.float32)
    for key in ('density', 'similarity'):
        array = result[key]
        scaled = (array - array.min()) / max(float(array.max() - array.min()), 1e-8)
        heat = np.stack([scaled, 0.3 * (1 - np.abs(scaled * 2 - 1)), 1 - scaled], -1) * 255
        overlay = Image.fromarray(np.clip(base * 0.55 + heat * 0.45, 0, 255).astype(np.uint8))
        overlay.save(directory / f'{stem}_{key}.png')
    with (directory / f'{stem}.json').open('w', encoding='utf-8') as handle:
        json.dump({'text': result['text'], 'count': result['count'],
                   'density_integral': float(result['density'].sum()),
                   'width': image.width, 'height': image.height}, handle, indent=2)
