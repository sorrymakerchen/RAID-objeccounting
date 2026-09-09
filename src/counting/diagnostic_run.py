"""Validation-only diagnostic workflow with a reference-reproduction gate."""

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from dataset.fsc147 import FSC147Dataset
from src.counting.cli import build_model, configure_logging, parse_config, write_json
from src.counting.diagnostics import (
    ANCHORS, candidate_mask, compare_reference, count_range, metrics, read_predictions,
    render_sample, select_samples, spatial_metrics,
)
from src.counting.training import read_checkpoint, restore_checkpoint, seed_worker, set_seed

LOGGER = logging.getLogger(__name__)
EXPECTED_METRICS = {'mae': 31.84626762622252, 'rmse': 96.84393429899362}


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_reference(reference, dataset, classes):
    expected = set(dataset.ids)
    if len(reference) != 1286 or len(expected) != 1286 or {r['image_id'] for r in reference} != expected:
        raise ValueError('Diagnosis requires exactly the complete 1286-image validation split')
    baseline = metrics(reference)
    for key, value in EXPECTED_METRICS.items():
        if abs(baseline[key] - value) > 1e-3:
            raise ValueError(f'Reference {key}={baseline[key]} does not match the agreed formal run {value}')
    for row in reference:
        image_id = row['image_id']
        if image_id not in classes or not classes[image_id].strip():
            raise ValueError(f'Official class name missing: {image_id}')
        if row['target'] != len(dataset.annotations[image_id]['points']):
            raise ValueError(f'Point count disagrees with reference: {image_id}')
        if row['text'] != dataset.texts[image_id]['text_description'].strip():
            raise ValueError(f'Text disagrees with reference: {image_id}')


def validate_checkpoint(state):
    config = state['config']
    if config.get('count_loss_mode', 'relative') != 'relative' or config.get('density_supervision_size', 32) != 32:
        raise ValueError('Formal baseline diagnosis requires the original loss and 32x32 supervision')
    required = {'image_size': 448, 'k': 150, 'warmup_epochs': 30,
                'reduced_dim': 384, 'expert_dim': 384, 'epochs': 100,
                'limit_train': None, 'limit_val': None}
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f'Formal checkpoint requires {key}={value!r}, got {config.get(key)!r}; '
                             'do not use a smoke/overfit checkpoint')
    if state['epoch'] != 33:
        raise ValueError(f'Expected best checkpoint epoch=33, got {state["epoch"]}')


def load_classes(path):
    result = {}
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if line.strip():
            name, category = line.split('\t', 1)
            if name in result:
                raise ValueError(f'Duplicate class annotation: {name}')
            result[name] = category.strip()
    return result


def unpack_variant(output, index, name, text):
    detail = output['diagnostics']
    routes = {}
    for key in ('stage1_probabilities', 'stage1_weights', 'stage2_probabilities', 'stage2_weights'):
        routes[key] = None if detail[key] is None else detail[key][index].cpu().tolist()
    routes['stage2_routing_active'] = detail['stage2_routing_active']
    variant = {'name': name, 'text': text, 'count': float(output['count'][index]),
               'density': output['density'][index, 0].cpu().numpy(),
               'similarity': detail['grid_similarity'][index, 0].cpu().numpy(),
               'indices': detail['candidate_indices'][index].cpu().numpy(), 'routes': routes}
    if not np.isfinite(variant['density']).all() or not np.isfinite(variant['similarity']).all():
        raise FloatingPointError(f'Non-finite diagnostic output for batch index {index}')
    if abs(float(variant['density'].sum()) - variant['count']) > max(1e-3, abs(variant['count']) * 1e-6):
        raise ValueError('Predicted density integral differs from count')
    if candidate_mask(variant['indices'], variant['density'].shape).sum() != 150:
        raise ValueError('Formal diagnostic output must contain exactly 150 candidate cells')
    return variant


def route_columns(routes):
    result = {}
    for key in ('stage1_probabilities', 'stage1_weights', 'stage2_probabilities', 'stage2_weights'):
        values = routes[key]
        for index in range(3):
            result[f'{key}_{index}'] = None if values is None else values[index]
    for index, weight in enumerate(routes['stage1_weights']):
        result[f'stage1_selected_{index}'] = int(weight > 0)
    for index in range(3):
        result[f'stage2_dominant_{index}'] = int(index == int(np.argmax(routes['stage2_weights'])))
    result['stage2_routing_active'] = routes['stage2_routing_active']
    return result


def aggregate_rows(rows, group_key=None):
    groups = {'all': rows} if group_key is None else {
        key: [r for r in rows if r[group_key] == key] for key in sorted({r[group_key] for r in rows})}
    output = []
    for name, group in groups.items():
        result = {'group': name, **metrics(group)}
        # Keep route summaries alongside errors to inspect category-specific concentration.
        for key in group[0]:
            if key.startswith(('stage1_', 'stage2_')):
                values = [r[key] for r in group if r[key] is not None]
                result[key + '_mean'] = float(np.mean(values)) if values else None
        for key in ('candidate_gt_mass_fraction', 'candidate_neighborhood_hit_rate',
                    'neighborhood_area_fraction', 'pred_mass_inside', 'pred_mass_outside'):
            values = [r[key] for r in group if r[key] is not None]
            result[key + '_mean'] = float(np.mean(values)) if values else None
        result['neighborhood_nearly_full_samples'] = sum(r['neighborhood_nearly_full'] for r in group)
        result['point_boundary_roundoff_count'] = sum(r['point_boundary_roundoff_count'] for r in group)
        output.append(result)
    return output


def write_report(directory, summary, rows, samples):
    comparison = summary['reproduction']
    current, reference = comparison['current'], comparison['reference']
    lines = ['# FSC147 验证集定位诊断', '',
             f"模式：{'三图冒烟，尚非完整验证' if summary['smoke'] else '完整验证集'}。",
             f"Checkpoint epoch：{summary['checkpoint_epoch']}；样本数：{len(rows)}。", '',
             '| 指标 | 本次 | 参考 |', '|---|---:|---:|',
             f"| MAE | {current['mae']:.6f} | {reference['mae']:.6f} |",
             f"| RMSE | {current['rmse']:.6f} | {reference['rmse']:.6f} |", '',
             f"复现检查：{'通过' if comparison['passed'] else '未通过'}；"
             f"最大逐图预测差值 {comparison['max_prediction_delta']:.6g}。", '']
    if not comparison['passed']:
        lines += ['本次停止归因。先检查 checkpoint、权重、配置和数值实现差异。',
                  '逐图差异见 `reference_differences.csv`；没有将差异解释为模型退化。']
    else:
        category_rows = sorted(aggregate_rows(rows, 'category'), key=lambda r: r['mae'], reverse=True)
        lines += ['## 数值观察', '',
                  '| 类别 | 样本数 | MAE | 平均偏差（预测−真实） |', '|---|---:|---:|---:|']
        for row in category_rows[:5]:
            lines.append(f"| {row['group']} | {row['samples']} | {row['mae']:.3f} | {row['bias']:.3f} |")
        route = summary['route_summary']
        for stage in ('stage1', 'stage2'):
            weights = ', '.join(f"{route[f'{stage}_weights_{i}_mean']:.3f}" for i in range(3))
            lines += ['', f'{stage} 平均实际混合权重：[{weights}]。']
        lines += ['', f"点邻域面积 ≥90% 的样本数：{route['neighborhood_nearly_full_samples']}；"
                  f"边界舍入点数：{route['point_boundary_roundoff_count']}。", '']
        lines += ['## 定位证据', '',
                  '以下是观察指标，不是因果结论。详细统计见分类、数量区间和路由 CSV。', '',
                  '| 图像 | 类别 | GT | 预测 | 候选承载 GT 质量 | 点邻域命中率 | 邻域面积占比 |',
                  '|---|---|---:|---:|---:|---:|---:|']
        indexed = {r['image_id']: r for r in rows}
        for sample in samples:
            row = indexed[sample['image_id']]
            mass = row['candidate_gt_mass_fraction']
            mass_text = 'N/A' if mass is None else f'{mass:.3f}'
            lines.append(f"| [{row['image_id']}](samples/{Path(row['image_id']).stem}/comparison.png) "
                         f"| {row['category']} | {row['target']:.0f} | {row['prediction']:.2f} "
                         f"| {mass_text} | {row['candidate_neighborhood_hit_rate']:.3f} "
                         f"| {row['neighborhood_area_fraction']:.3f} |")
        lines += ['', '## 如何解释', '',
                  '- 文本响应与候选都偏离目标：优先调查文本对齐或语义定位。',
                  '- 响应覆盖目标但候选集中：调查候选选择与空间覆盖；top-k 不要求覆盖每个目标。',
                  '- 候选位置合理但预测密度不足：调查密度回归或尺度表示。',
                  '- 路由集中且与失败类别一致：记录为消融假设，不直接断言路由造成失败。', '',
                  '点所在格及一格邻域只是定位代理，不是分割掩码。邻域面积 ≥90% 时标记为区分能力有限。',
                  '≤1e-6 像素的标注边界浮点误差仅在网格投影时截到边界；原始点和真实数量不变，次数列入 CSV。',
                  '同图文本响应共用颜色范围，真实/所有预测密度共用颜色范围，详见各图 JSON。',
                  '`the elephants` 仅是提示对照，没有赋予其零数量真值；替换提示结果不计入基准 MAE。',
                  '原始网格数组、原图尺寸数组、候选坐标和路由权重保存在各图目录。']
    lines += ['', '## 复现信息', '',
              '见 `summary.json`：生效配置、原 checkpoint 配置、路径与文件 SHA-256、'
              '软件版本、checkpoint epoch、预期完整验证指标及数值差异。',
              '本任务未训练模型，未使用测试集。', '']
    (Path(directory) / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


@torch.no_grad()
def run_diagnosis(model, dataset, reference, classes, config, epoch, directory, smoke=False,
                  provenance=None):
    """Inject a model for workflow tests; the public CLI enforces the formal checkpoint."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.eval()
    manifest = select_samples(reference)
    if smoke:
        manifest = [r for r in manifest if r['image_id'] in ANCHORS]
    write_json(directory / 'samples.json', manifest)
    positions = {name: index for index, name in enumerate(dataset.ids)}
    selected = {r['image_id'] for r in manifest}
    source = Subset(dataset, [positions[name] for name in ANCHORS]) if smoke else dataset
    loader = DataLoader(source, batch_size=config['batch_size'], num_workers=config['workers'],
                        shuffle=False, worker_init_fn=seed_worker)
    rows, cached = [], {}
    if str(config['device']).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(config['device'])
    for step, batch in enumerate(loader):
        try:
            # Only images and text enter inference; annotations are read afterwards.
            output = model(batch['image'].to(config['device']), batch['text'], epoch,
                           return_diagnostics=True)
            for index, image_id in enumerate(batch['image_id']):
                variant = unpack_variant(output, index, 'original', batch['text'][index])
                gt = batch['density'][index, 0].numpy()
                points = dataset.annotations[image_id]['points']
                original_size = tuple(batch['original_size'][index].tolist())
                prediction, target = variant['count'], float(batch['count'][index])
                row = {'image_id': image_id, 'category': classes[image_id],
                       'text': batch['text'][index], 'target': target, 'prediction': prediction,
                       'absolute_error': abs(prediction - target), 'signed_error': prediction - target,
                       'count_range': count_range(target),
                       **spatial_metrics(variant['indices'], gt, variant['density'], points, original_size),
                       **route_columns(variant['routes'])}
                rows.append(row)
                if image_id in selected:
                    cached[image_id] = {
                        'baseline': variant,
                        'images': batch['image'].clone(),
                        'texts': list(batch['text']),
                        'index': index,
                    }
            if step % 50 == 0 or step + 1 == len(loader):
                LOGGER.info('baseline batch=%d/%d samples=%d', step + 1, len(loader), len(rows))
        except Exception:
            LOGGER.exception('Diagnosis failed batch=%d image_ids=%s', step, batch['image_id'])
            raise
    reference_subset = [r for r in reference if r['image_id'] in {x['image_id'] for x in rows}]
    reproduction, differences = compare_reference(rows, reference_subset)
    summary = {'smoke': smoke, 'checkpoint_epoch': epoch, 'config': config,
               'expected_full_validation': EXPECTED_METRICS, 'reproduction': reproduction,
               'provenance': provenance or {}, 'torch_version': torch.__version__,
               'scope': 'validation only; no training; no oracle candidates',
               'point_neighborhood_full_threshold': 0.9,
               'route_summary': aggregate_rows(rows)[0]}
    write_csv(directory / 'predictions.csv', rows)
    write_csv(directory / 'reference_differences.csv', differences)
    write_csv(directory / 'by_category.csv', aggregate_rows(rows, 'category'))
    write_csv(directory / 'by_count.csv', aggregate_rows(rows, 'count_range'))
    write_csv(directory / 'routes.csv', [{k: v for k, v in r.items()
                                         if k in ('image_id', 'category') or k.startswith(('stage1_', 'stage2_'))}
                                        for r in rows])
    if not reproduction['passed']:
        summary['status'] = 'reference_mismatch_no_attribution'
        write_json(directory / 'summary.json', summary)
        write_report(directory, summary, rows, manifest)
        LOGGER.error('Reference reproduction failed: %s', reproduction)
        return 2

    prompt_rows = []
    for sample in manifest:
        image_id = sample['image_id']
        batch = dataset[positions[image_id]]
        context = cached[image_id]
        context_images = context['images'].to(config['device'])

        def evaluate_prompt(name, text):
            texts = list(context['texts'])
            texts[context['index']] = text
            output = model(context_images, texts, epoch, return_diagnostics=True)
            return unpack_variant(output, context['index'], name, text)

        # Keep the exact validation batch shape, peers, and sample position while changing text.
        original = evaluate_prompt('original', batch['text'])
        batch_delta = original['count'] - context['baseline']['count']
        if abs(batch_delta) > 1e-3:
            raise ValueError(
                f'Repeated validation-batch prediction differs: {image_id}, delta={batch_delta}')
        variants = [original]
        for name, text in [('template', 'a photo of ' + classes[image_id]), ('control', 'the elephants')]:
            variants.append(evaluate_prompt(name, text))
        points = dataset.annotations[image_id]['points']
        gt = batch['density'][0].numpy()
        original_size = tuple(batch['original_size'].tolist())
        details = []
        for variant in variants:
            overlap = len(set(original['indices'].tolist()) & set(variant['indices'].tolist()))
            detail = {'name': variant['name'], 'text': variant['text'], 'count': variant['count'],
                      'count_delta_from_original': variant['count'] - original['count'],
                      'topk_overlap_fraction': overlap / len(original['indices']),
                      'topk_jaccard': overlap / (2 * len(original['indices']) - overlap),
                      'similarity_mean_abs_change': float(np.abs(variant['similarity'] - original['similarity']).mean()),
                      'routes': variant['routes'],
                      **spatial_metrics(variant['indices'], gt, variant['density'], points, original_size)}
            details.append(detail)
            prompt_rows.append({'image_id': image_id, 'category': classes[image_id],
                                **{k: v for k, v in detail.items() if k != 'routes'},
                                **route_columns(variant['routes'])})
        sample_dir = directory / 'samples' / Path(image_id).stem
        with Image.open(dataset.paths(image_id)[0]) as image:
            ranges = render_sample(image, points, gt, variants, sample_dir)
        write_json(sample_dir / 'diagnostics.json', {'image_id': image_id, 'target_count': len(points),
                   'annotation_reference_text': original['text'],
                   'target_count_is_for_original_prompt': True,
                   'original_size_hw': list(original_size), 'grid_shape': list(gt.shape),
                   'repeat_delta_from_validation_batch': batch_delta,
                   'display_ranges': ranges, 'variants': details,
                   'control_prompt_zero_ground_truth_assumed': False})
        LOGGER.info('Rendered sample=%s prompts=%d', image_id, len(variants))
    write_csv(directory / 'prompt_comparisons.csv', prompt_rows)
    summary['status'] = 'smoke_passed_not_full_validation' if smoke else 'complete'
    if str(config['device']).startswith('cuda'):
        summary['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated(config['device'])
        summary['gpu'] = torch.cuda.get_device_name(config['device'])
    write_json(directory / 'summary.json', summary)
    write_report(directory, summary, rows, manifest)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description='FSC147 formal-checkpoint validation diagnosis (no training)')
    parser.add_argument('--checkpoint', default='outputs/fsc147_text/best.pt')
    parser.add_argument('--config', required=True, help='Server config with prepared local weights')
    parser.add_argument('--reference-csv', required=True, help='Full val_predictions.csv from the formal run')
    parser.add_argument('--output-dir', default='outputs/diagnosis_full')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--class-file', help='Defaults to data_root/ImageClasses_FSC147.txt')
    parser.add_argument('--smoke', action='store_true', help='Only three anchor images; not full validation')
    args = parser.parse_args(argv)
    configure_logging(args.output_dir)
    try:
        directory = Path(args.output_dir)
        if any(directory.iterdir()):
            existing = [p for p in directory.iterdir() if p.name != 'run.log']
            if existing:
                raise FileExistsError(f'Use a fresh diagnostic output directory: {directory}')
        state = read_checkpoint(args.checkpoint)
        validate_checkpoint(state)
        config, _ = parse_config('evaluate', ['--checkpoint', args.checkpoint, '--config', args.config,
                                             '--output-dir', args.output_dir, '--device', args.device])
        for key in ('image_size', 'k', 'warmup_epochs', 'reduced_dim', 'expert_dim', 'epochs',
                    'limit_train', 'limit_val', 'batch_size'):
            if config[key] != state['config'][key]:
                raise ValueError(f'Diagnosis cannot override checkpoint setting {key}')
        paths = {key: config[key] for key in ('data_root', 'text_annotations', 'projection_weights',
                                             'dino_repo', 'dino_weights', 'clip_weights')}
        for key in ('text_annotations', 'projection_weights', 'dino_weights', 'clip_weights'):
            if not paths[key] or not Path(paths[key]).is_file():
                raise FileNotFoundError(f'Prepared local {key} file missing: {paths[key]}')
        if not paths['dino_repo'] or not Path(paths['dino_repo'], 'hubconf.py').is_file():
            raise FileNotFoundError(f'Prepared DINOv2 repository missing: {paths["dino_repo"]}')
        dataset = FSC147Dataset(config['data_root'], config['text_annotations'], 'val',
                                image_size=config['image_size'], flip_probability=0)
        classes = load_classes(args.class_file or Path(config['data_root']) / 'ImageClasses_FSC147.txt')
        reference = read_predictions(args.reference_csv)
        validate_reference(reference, dataset, classes)
        provenance = {'checkpoint': str(Path(args.checkpoint).resolve()),
                      'checkpoint_sha256': sha256(args.checkpoint),
                      'checkpoint_config': state['config'],
                      'reference_csv': str(Path(args.reference_csv).resolve()),
                      'reference_sha256': sha256(args.reference_csv),
                      'resolved_paths': {k: str(Path(v).resolve()) for k, v in paths.items()},
                      'weight_sha256': {k: sha256(paths[k]) for k in
                                        ('projection_weights', 'dino_weights', 'clip_weights')},
                      'dino_hubconf_sha256': sha256(Path(paths['dino_repo']) / 'hubconf.py')}
        # Optimizer states are unnecessary for this inference-only job.
        del state
        set_seed(config['seed'])
        model = build_model(config)
        restored = restore_checkpoint(args.checkpoint, model, restore_rng=False)
        epoch = restored['epoch']
        del restored
        return run_diagnosis(model, dataset, reference, classes, config, epoch, args.output_dir,
                             args.smoke, provenance)
    except Exception:
        LOGGER.exception('Formal validation diagnosis failed')
        return 1
