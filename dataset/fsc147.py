"""FSC147 text counting data, with count-preserving density transforms."""

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def image_tensor(image, size=448):
    image = image.convert('RGB').resize((size, size), Image.Resampling.BICUBIC)
    tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255
    return (tensor - MEAN) / STD


def resize_density(density, size):
    """Preserve each map's own mass, including when restoring predictions for display."""
    if density.ndim not in (3, 4):
        raise ValueError(f'Density must be CHW or BCHW, got {tuple(density.shape)}')
    single = density.ndim == 3
    batch = density.unsqueeze(0) if single else density
    batch = batch.float()
    if not torch.isfinite(batch).all() or (batch < 0).any():
        raise ValueError('Density must contain finite, nonnegative values')
    mass = batch.sum((-2, -1), keepdim=True)
    # Area pooling retains isolated annotations that bilinear downsampling can miss.
    resized = F.interpolate(batch, size=size, mode='area')
    total = resized.sum((-2, -1), keepdim=True)
    resized = resized * (mass / total.clamp_min(torch.finfo(torch.float32).tiny))
    return resized[0] if single else resized


def read_json(path):
    try:
        with Path(path).open(encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, ValueError):
        LOGGER.exception('Cannot read JSON path=%s', path)
        raise


class FSC147Dataset(Dataset):
    def __init__(self, root, text_annotations, split='train', image_size=448,
                 flip_probability=None, limit=None, density_supervision_size=None):
        self.root = Path(root)
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'Unsupported FSC147 split: {split}')
        if image_size < 28 or image_size % 14:
            raise ValueError('image_size must be a multiple of 14 and at least 28')
        self.image_size = image_size
        self.grid_size = image_size // 14 if density_supervision_size is None else density_supervision_size
        if self.grid_size < 1:
            raise ValueError('Density supervision size must be positive')
        self.flip_probability = (0.5 if split == 'train' else 0) if flip_probability is None else flip_probability
        if not 0 <= self.flip_probability <= 1:
            raise ValueError('flip_probability must lie in [0, 1]')
        if split != 'train' and self.flip_probability:
            raise ValueError('Validation and test transforms must be deterministic')
        self.annotations = read_json(self.root / 'annotation_FSC147_384.json')
        self.texts = read_json(text_annotations)
        splits = read_json(self.root / 'Train_Test_Val_FSC_147.json')
        sets = {key: set(splits[key]) for key in ('train', 'val', 'test')}
        for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
            if sets[left] & sets[right]:
                raise ValueError(f'FSC147 split overlap: {left}/{right}')
        for key in sets:
            if len(sets[key]) != len(splits[key]):
                raise ValueError(f'Duplicate image IDs in split {key}')
        self.ids = list(splits[split])
        if limit is not None:
            if limit < 1:
                raise ValueError('Dataset limit must be positive')
            self.ids = self.ids[:limit]
        if not self.ids:
            raise ValueError(f'Empty FSC147 split: {split}')
        for image_id in self.ids:
            if image_id not in self.annotations:
                raise ValueError(f'Missing point annotation for {image_id}')
            entry = self.texts.get(image_id, {})
            if not isinstance(entry.get('text_description'), str) or not entry['text_description'].strip():
                raise ValueError(f'Missing or empty text annotation for {image_id}')
            if entry.get('data_split', split) != split:
                raise ValueError(f'Text split disagrees with official split: {image_id}')
            for path in self.paths(image_id):
                if not path.is_file():
                    raise FileNotFoundError(f'FSC147 {split}: missing file {path}')

    def paths(self, image_id):
        return (self.root / 'images_384_VarV2' / image_id,
                self.root / 'gt_density_map_adaptive_384_VarV2' / f'{Path(image_id).stem}.npy')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        try:
            image_path, density_path = self.paths(image_id)
            with Image.open(image_path) as image:
                original_size = torch.tensor([image.height, image.width])
                tensor = image_tensor(image, self.image_size)
            array = np.load(density_path, allow_pickle=False)
            if array.ndim != 2:
                raise ValueError(f'Expected a 2D density map, got {array.shape}')
            density = resize_density(torch.from_numpy(array).float().unsqueeze(0),
                                     (self.grid_size, self.grid_size))
            count = float(len(self.annotations[image_id]['points']))
            if count == 0:
                density.zero_()
            elif density.sum() <= 0:
                raise ValueError(f'Positive point count {count} but empty density map')
            else:
                # Only training/evaluation targets use point labels for mass correction.
                density = density * (count / density.sum())
            if self.flip_probability and random.random() < self.flip_probability:
                tensor, density = tensor.flip(-1), density.flip(-1)
            return {'image': tensor, 'text': self.texts[image_id]['text_description'].strip(),
                    'density': density, 'count': torch.tensor(count), 'image_id': image_id,
                    'original_size': original_size}
        except Exception:
            LOGGER.exception('FSC147Dataset.__getitem__ image_id=%s index=%s', image_id, index)
            raise
