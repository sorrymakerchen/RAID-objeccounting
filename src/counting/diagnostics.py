"""Annotation-only analysis and rendering; annotations never select model candidates."""

import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from dataset.fsc147 import resize_density

ANCHORS = ('3425.jpg', '3427.jpg', '935.jpg')
COUNT_BINS = ((0, 25, '0-24'), (25, 50, '25-49'), (50, 100, '50-99'),
              (100, 200, '100-199'), (200, 500, '200-499'), (500, math.inf, '500+'))


def count_range(count):
    return next(label for low, high, label in COUNT_BINS if low <= count < high)


def read_predictions(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    seen = set()
    for row in rows:
        if row['image_id'] in seen:
            raise ValueError(f'Duplicate prediction ID: {row["image_id"]}')
        seen.add(row['image_id'])
        for key in ('prediction', 'target', 'absolute_error'):
            row[key] = float(row[key])
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError(f'Invalid {key} for {row["image_id"]}')
        if abs(abs(row['prediction'] - row['target']) - row['absolute_error']) > 1e-6:
            raise ValueError(f'Incorrect absolute_error for {row["image_id"]}')
    if not rows:
        raise ValueError(f'Empty reference CSV: {path}')
    return rows


def metrics(rows):
    if not rows:
        raise ValueError('Cannot summarize zero samples')
    errors = np.array([r['prediction'] - r['target'] for r in rows], dtype=np.float64)
    return {'samples': len(rows), 'mae': float(np.abs(errors).mean()),
            'rmse': float(np.sqrt(np.square(errors).mean())), 'bias': float(errors.mean())}


def compare_reference(actual, reference, tolerance=1e-3):
    original = {r['image_id']: r for r in reference}
    if len(original) != len(reference) or len({r['image_id'] for r in actual}) != len(actual):
        raise ValueError('Duplicate image IDs in comparison')
    if {r['image_id'] for r in actual} != set(original):
        raise ValueError('Image ID sets do not match reference')
    differences = []
    metadata_match = True
    for row in actual:
        ref = original[row['image_id']]
        same = row['target'] == ref['target'] and row['text'] == ref['text']
        metadata_match &= same
        differences.append({'image_id': row['image_id'], 'metadata_match': same,
                            'reference_prediction': ref['prediction'],
                            'prediction': row['prediction'],
                            'prediction_delta': row['prediction'] - ref['prediction']})
    current, baseline = metrics(actual), metrics(reference)
    delta = {key: current[key] - baseline[key] for key in ('mae', 'rmse')}
    maximum = max(abs(r['prediction_delta']) for r in differences)
    # Matching aggregate errors can conceal swapped or compensating predictions.
    passed = metadata_match and all(abs(v) <= tolerance for v in delta.values()) and maximum <= tolerance
    return {'passed': passed, 'tolerance': tolerance, 'current': current,
            'reference': baseline, 'metric_deltas': delta,
            'max_prediction_delta': maximum, 'metadata_match': metadata_match}, differences


def select_samples(rows):
    by_id = {r['image_id']: r for r in rows}
    reasons = {}

    def add(image_id, reason):
        if image_id not in by_id:
            raise ValueError(f'Required diagnostic image missing from validation: {image_id}')
        reasons.setdefault(image_id, []).append(reason)

    for name in ANCHORS:
        add(name, 'anchor')
    for row in sorted(rows, key=lambda r: (-r['absolute_error'], r['image_id']))[:10]:
        add(row['image_id'], 'worst10')
    for low, high, label in COUNT_BINS:
        group = [r for r in rows if low <= r['target'] < high]
        for row in sorted(group, key=lambda r: (
                r['absolute_error'] / max(r['target'], 1.), r['image_id']))[:2]:
            add(row['image_id'], 'control:' + label)
    return [dict(by_id[name], reasons=reasons[name]) for name in sorted(reasons)]


def candidate_mask(indices, grid_shape):
    indices = np.asarray(indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError('Candidate indices must be a one-dimensional integer array')
    if len(np.unique(indices)) != len(indices):
        raise ValueError('Candidate indices must be unique')
    if not len(indices) or indices.min() < 0 or indices.max() >= np.prod(grid_shape):
        raise ValueError('Candidate index outside feature grid')
    mask = np.zeros(grid_shape, dtype=bool)
    mask.flat[indices] = True
    return mask


def candidate_rectangles(indices, grid_shape, original_size):
    candidate_mask(indices, grid_shape)
    height, width = original_size
    gh, gw = grid_shape
    indices = np.asarray(indices)
    y, x = indices // gw, indices % gw
    return np.stack([x * width / gw, y * height / gh,
                     (x + 1) * width / gw, (y + 1) * height / gh], axis=1)


def point_neighborhood(points, original_size, grid_shape):
    height, width = original_size
    gh, gw = grid_shape
    occupied = np.zeros(grid_shape, dtype=bool)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(points).all():
        raise ValueError('Point annotations must be finite')
    if len(points):
        # FSC147 contains 408.00000000000006 for a width-408 image (1989.jpg).
        # Permit floating-point boundary roundoff, not materially misplaced annotations.
        epsilon = 1e-6
        if (points < -epsilon).any() or (points[:, 0] > width + epsilon).any() or (points[:, 1] > height + epsilon).any():
            raise ValueError('Point coordinates outside the original image')
        points = np.clip(points, [0, 0], [width, height])
        xs = np.minimum((points[:, 0] * gw / width).astype(int), gw - 1)
        ys = np.minimum((points[:, 1] * gh / height).astype(int), gh - 1)
        occupied[ys, xs] = True
    dilated = F.max_pool2d(torch.from_numpy(occupied.astype(np.float32))[None, None],
                          kernel_size=3, stride=1, padding=1)[0, 0].numpy().astype(bool)
    return occupied, dilated


def spatial_metrics(indices, target_density, prediction_density, points, original_size):
    target_density = np.asarray(target_density)
    prediction_density = np.asarray(prediction_density)
    if target_density.shape != prediction_density.shape or target_density.ndim != 2:
        raise ValueError('Target and predicted diagnostic densities must have the same 2D grid')
    if not np.isfinite(target_density).all() or not np.isfinite(prediction_density).all():
        raise ValueError('Non-finite diagnostic density')
    candidates = candidate_mask(indices, target_density.shape)
    occupied, nearby = point_neighborhood(points, original_size, target_density.shape)
    target_mass = float(target_density.sum())
    raw_points = np.asarray(points).reshape(-1, 2)
    height, width = original_size
    rounded = ((raw_points < 0).any(1) | (raw_points[:, 0] > width) | (raw_points[:, 1] > height)).sum()
    return {
        'point_boundary_roundoff_count': int(rounded),
        'candidate_gt_mass_fraction': float(target_density[candidates].sum() / target_mass) if target_mass else None,
        'candidate_area_fraction': float(candidates.mean()),
        'candidate_point_cell_hit_rate': float(occupied[candidates].mean()),
        'candidate_neighborhood_hit_rate': float(nearby[candidates].mean()),
        'neighborhood_area_fraction': float(nearby.mean()),
        'neighborhood_nearly_full': bool(nearby.mean() >= 0.9),
        'pred_mass_inside': float(prediction_density[nearby].sum()),
        'pred_mass_outside': float(prediction_density[~nearby].sum()),
        'gt_mass_inside': float(target_density[nearby].sum()),
        'gt_mass_outside': float(target_density[~nearby].sum()),
        'count_bias': float(prediction_density.sum()) - len(points),
    }


def shared_ranges(variants, target_density):
    return {'similarity': [float(min(v['similarity'].min() for v in variants)),
                           float(max(v['similarity'].max() for v in variants))],
            'density': [0., float(max(target_density.max(),
                                     max(v['density'].max() for v in variants)))]}


def colorize(array, limits):
    low, high = limits
    scaled = np.clip((array - low) / max(high - low, 1e-12), 0, 1)
    rgb = np.stack([scaled, 0.3 * (1 - np.abs(scaled * 2 - 1)), 1 - scaled], -1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def render_sample(image, points, target_grid, variants, directory):
    """Render all prompts together using shared scales in original-pixel mass units."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    image = image.convert('RGB')
    size = (image.height, image.width)
    target = resize_density(torch.as_tensor(target_grid).float()[None], size)[0].numpy()
    display = []
    for variant in variants:
        density = resize_density(torch.as_tensor(variant['density']).float()[None], size)[0].numpy()
        similarity = F.interpolate(torch.as_tensor(variant['similarity']).float()[None, None],
                                   size=size, mode='bilinear', align_corners=False)[0, 0].numpy()
        display.append(dict(variant, density=density, similarity=similarity))
    ranges = shared_ranges(display, target)
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    for x, y in points:
        draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill='#ff3838')
    marked.save(directory / 'points.png')
    image.save(directory / 'original.png')
    colorize(target, ranges['density']).save(directory / 'target_density.png')
    cell_w, cell_h = 340, 350
    panel = Image.new('RGB', (cell_w * 5, cell_h * len(variants)), 'white')
    for row, (raw, variant) in enumerate(zip(variants, display)):
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay, 'RGBA')
        grid_shape = raw['density'].shape
        rectangles = candidate_rectangles(raw['indices'], grid_shape, size)
        for x0, y0, x1, y1 in rectangles:
            draw.rectangle((x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)),
                           fill=(255, 205, 0, 65), outline=(255, 175, 0, 200))
        similarity = Image.blend(image, colorize(variant['similarity'], ranges['similarity']), .5)
        prediction = colorize(variant['density'], ranges['density'])
        prefix = raw['name']
        similarity.save(directory / f'{prefix}_similarity.png')
        overlay.save(directory / f'{prefix}_candidates.png')
        prediction.save(directory / f'{prefix}_density.png')
        occupied, nearby = point_neighborhood(points, size, grid_shape)
        np.savez_compressed(directory / f'{prefix}_arrays.npz',
                            similarity_grid=raw['similarity'], density_grid=raw['density'],
                            candidate_indices=raw['indices'], candidate_rectangles_xyxy=rectangles,
                            candidate_mask=candidate_mask(raw['indices'], grid_shape),
                            point_occupied_mask=occupied, point_neighborhood_mask=nearby,
                            target_density_grid=target_grid, target_density=target,
                            predicted_density=variant['density'], similarity=variant['similarity'],
                            points_xy=np.asarray(points).reshape(-1, 2))
        prediction_label = (f"Pred={raw['count']:.3f} | abs error={abs(raw['count']-len(points)):.3f}"
                            if row == 0 else f"Pred={raw['count']:.3f} | delta={raw['count']-variants[0]['count']:+.3f}")
        columns = [(marked, f"{prefix}: {raw['text']}\nOriginal-target GT={len(points)}"),
                   (similarity, f"Text similarity (shared scale)\n{ranges['similarity'][0]:.4g} to {ranges['similarity'][1]:.4g}"),
                   (overlay, 'Selected patches'),
                   (colorize(target, ranges['density']), f"Target density (shared scale)\n0 to {ranges['density'][1]:.4g} count/pixel"),
                   (prediction, prediction_label)]
        for column, (picture, label) in enumerate(columns):
            picture = picture.copy()
            picture.thumbnail((cell_w - 10, cell_h - 100))
            x, y = column * cell_w, row * cell_h
            panel.paste(picture, (x + (cell_w - picture.width) // 2, y + 50))
            ImageDraw.Draw(panel).text((x + 5, y + 5), label, fill='black')
        routes = raw.get('routes', {})
        for stage in ('stage1_weights', 'stage2_weights'):
            line = 0 if stage == 'stage1_weights' else 1
            if routes.get(stage) is not None:
                values = ', '.join(f'{v:.3f}' for v in routes[stage])
                ImageDraw.Draw(panel).text((cell_w * 4 + 5, row * cell_h + cell_h - 38 + line * 15),
                                          f'{stage}: [{values}]', fill='black')
            else:
                ImageDraw.Draw(panel).text((cell_w * 4 + 5, row * cell_h + cell_h - 38 + line * 15),
                                          f'{stage}: N/A', fill='black')
    panel.save(directory / 'comparison.png')
    return ranges
