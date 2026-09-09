"""Controlled validation interventions and independent counting training experiments."""

import argparse
import hashlib
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from dataset.fsc147 import FSC147Dataset, read_json
from src.counting.cli import build_model, configure_logging, parse_config, write_json
from src.counting.diagnostic_run import (aggregate_rows, load_classes, route_columns, sha256,
                                        unpack_variant, validate_checkpoint, validate_reference, write_csv)
from src.counting.diagnostics import (ANCHORS, compare_reference, count_range, metrics,
                                      read_predictions, render_sample, select_samples, spatial_metrics)
from src.counting.training import (counting_loss, evaluate, read_checkpoint, restore_checkpoint,
                                    save_checkpoint, seed_worker, set_seed, train_epoch)

LOGGER = logging.getLogger(__name__)
GROUPS = ('T0', 'T1', 'T2')
INTERVENTIONS = ('D0', 'D1', 'D2', 'D3', 'D4')


def experiment_config(base, group):
    if group not in GROUPS:
        raise ValueError(f'Unknown training group: {group}')
    return dict(base, image_size=896 if group == 'T2' else 448,
                count_loss_mode='absolute' if group == 'T1' else 'relative',
                density_supervision_size=32, seed=42, epochs=100, batch_size=2,
                accumulation=4, lr=1e-4, weight_decay=1e-4, warmup_epochs=30,
                k=150, reduced_dim=384, expert_dim=384, augment=True,
                limit_train=None, limit_val=None)


def screen_results(base, variant):
    checks = {'mae': variant['mae'] <= base['mae'] * .99,
              'rmse': variant['rmse'] <= base['rmse'] * 1.01,
              'high_count': variant['high_mae'] <= base['high_mae'] * .95,
              'low_count': variant['low_mae'] <= base['low_mae'] + .5}
    return {'eligible': all(checks.values()), 'checks': checks}


def scoring(rows):
    high = [r for r in rows if r['target'] >= 200]
    low = [r for r in rows if r['target'] < 25]
    return dict(metrics(rows), high_mae=metrics(high)['mae'] if high else None,
                low_mae=metrics(low)['mae'] if low else None)


def dataset_for(config, split, training=False, native_target=False):
    return FSC147Dataset(config['data_root'], config['text_annotations'], split,
                         image_size=config['image_size'], flip_probability=.5 if training else 0,
                         density_supervision_size=None if native_target else 32)


def loader_for(dataset, config, training=False, epoch=0):
    # Data order and worker augmentation RNG do not depend on model RNG consumption or resolution.
    generator = torch.Generator().manual_seed(config.get('seed', 42) + epoch)
    return DataLoader(dataset, batch_size=config['batch_size'], num_workers=config['workers'],
                      shuffle=training, drop_last=False, worker_init_fn=seed_worker, generator=generator)


def state_digest(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        if not key.startswith('encoder.'):
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def evaluate_variant(model, dataset, config, epoch, intervention, selected, directory,
                     classes=None, smoke=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.eval()
    device = config['device']
    cuda = str(device).startswith('cuda')
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    rows, cached = [], {}
    # Filter after forward so anchors retain the exact validation batch shape and peers.
    loader = loader_for(dataset, config)
    for step, batch in enumerate(loader):
        if smoke and not set(batch['image_id']) & set(ANCHORS):
            continue
        output = model(batch['image'].to(device), batch['text'], epoch,
                       return_diagnostics=True, intervention=intervention)
        for index, image_id in enumerate(batch['image_id']):
            if smoke and image_id not in ANCHORS:
                continue
            variant = unpack_variant(output, index, intervention, batch['text'][index])
            detail = output['diagnostics']
            for key in ('mixed_logits', 'stage2_expert_logits'):
                if key in detail:
                    variant[key] = detail[key][index].detach().cpu().numpy()
                    if not np.isfinite(variant[key]).all():
                        raise FloatingPointError(f'Non-finite {key}: {image_id}')
            gt = batch['density'][index, 0].numpy()
            points = dataset.annotations[image_id]['points']
            original_size = tuple(batch['original_size'][index].tolist())
            prediction, target = variant['count'], float(batch['count'][index])
            row = {'image_id': image_id, 'text': batch['text'][index],
                   'category': classes[image_id] if classes else 'fixture',
                   'target': target, 'prediction': prediction,
                   'absolute_error': abs(prediction - target), 'signed_error': prediction - target,
                   'count_range': count_range(target),
                   **spatial_metrics(variant['indices'], gt, variant['density'], points, original_size),
                   **route_columns(variant['routes'])}
            rows.append(row)
            if image_id in selected:
                cached[image_id] = variant
        del output
        if step % 50 == 0:
            LOGGER.info('evaluate_variant mode=%s batch=%d/%d samples=%d', intervention, step, len(loader), len(rows))
    if cuda:
        torch.cuda.synchronize(device)
    timing = {'seconds': time.perf_counter() - start,
              'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(device) if cuda else 0}
    write_csv(directory / 'predictions.csv', rows)
    write_csv(directory / 'by_category.csv', aggregate_rows(rows, 'category'))
    write_csv(directory / 'by_count.csv', aggregate_rows(rows, 'count_range'))
    write_csv(directory / 'routes.csv', [{k: v for k, v in r.items()
                                         if k == 'image_id' or k.startswith(('stage1_', 'stage2_'))} for r in rows])
    write_json(directory / 'summary.json', dict(scoring(rows), **timing, epoch=epoch,
                                               intervention=intervention, smoke=smoke))
    return rows, cached, timing


def export_comparisons(dataset, caches, directory):
    directory = Path(directory)
    ids = sorted(set.intersection(*(set(v) for v in caches.values())))
    for image_id in ids:
        sample = dataset[dataset.ids.index(image_id)]
        variants = [cache[image_id] for cache in caches.values()]
        destination = directory / Path(image_id).stem
        with Image.open(dataset.paths(image_id)[0]) as image:
            ranges = render_sample(image, dataset.annotations[image_id]['points'],
                                   sample['density'][0].numpy(), variants, destination)
        write_json(destination / 'diagnostics.json', {'image_id': image_id, 'display_ranges': ranges,
                   'variants': [{'name': v['name'], 'count': v['count'], 'routes': v['routes'],
                                 'grid_shape': list(v['density'].shape)} for v in variants]})
        for variant in variants:
            arrays = {k: variant[k] for k in ('mixed_logits', 'stage2_expert_logits') if k in variant}
            if arrays:
                np.savez_compressed(destination / (variant['name'] + '_logits.npz'), **arrays)


def prepared_provenance(config):
    paths = {k: config[k] for k in ('text_annotations', 'projection_weights', 'dino_weights', 'clip_weights')}
    for key, path in paths.items():
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f'Experiment requires prepared local {key}: {path}')
    hub = Path(config['dino_repo'] or '') / 'hubconf.py'
    if not hub.is_file():
        raise FileNotFoundError(f'Prepared DINOv2 hubconf missing: {hub}')
    root = Path(__file__).resolve().parents[2]
    return {'files': {k: {'path': str(Path(v).resolve()), 'sha256': sha256(v)} for k, v in paths.items()},
            'dino_hubconf_sha256': sha256(hub), 'torch_version': torch.__version__,
            'annotation_sha256': {name: sha256(Path(config['data_root']) / name) for name in
                ('Train_Test_Val_FSC_147.json', 'annotation_FSC147_384.json', 'ImageClasses_FSC147.txt')},
            'source_sha256': {name: sha256(root / name) for name in
                ('src/counting/model.py', 'src/counting/training.py', 'src/counting/ablation_run.py',
                 'src/counting/encoders.py', 'src/expert.py', 'dataset/fsc147.py')}}


def write_experiment_report(directory, table, status, selected=None):
    lines = ['# FSC147 controlled experiments', '', 'Status: ' + status, '',
             '| Experiment | MAE | RMSE | >=200 MAE | 0-24 MAE |', '|---|---:|---:|---:|---:|']
    for row in table:
        def fmt(value):
            return 'N/A' if value is None else f'{value:.4f}'
        lines.append('| ' + row['experiment'] + ' | ' + ' | '.join(fmt(row[k]) for k in
                     ('mae', 'rmse', 'high_mae', 'low_mae')) + ' |')
    lines += ['', 'Validation only; single seed. No test-set selection or causal claim.',
              'D1 changes both resolution and candidate-area ratio. Higher counts alone are not evidence of improvement.',
              'Expert logits are before Softplus; per-expert Softplus counts are not additive contributions.']
    if selected is not None:
        lines += ['', 'Next replication candidate: ' + selected,
                  'Screen: MAE <= 0.99*T0; RMSE <= 1.01*T0; high-count MAE <= 0.95*T0; low-count MAE <= T0+0.5.']
    (Path(directory) / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run_interventions(args, directory):
    state = read_checkpoint(args.checkpoint)
    validate_checkpoint(state)
    if state['config'].get('count_loss_mode', 'relative') != 'relative':
        raise ValueError('D0-D4 require the original relative-count-loss checkpoint')
    config, _ = parse_config('evaluate', ['--config', args.config, '--checkpoint', args.checkpoint,
                                          '--output-dir', str(directory), '--device', args.device])
    if config['batch_size'] != 2 or state['config']['batch_size'] != 2:
        raise ValueError('D0-D4 require the original batch size 2')
    for key in ('image_size', 'warmup_epochs', 'epochs', 'limit_train', 'limit_val'):
        if config[key] != state['config'][key]:
            raise ValueError(f'Cannot change formal intervention setting {key}')
    provenance = prepared_provenance(config)
    provenance.update(checkpoint_sha256=sha256(args.checkpoint), reference_sha256=sha256(args.reference_csv))
    reference = read_predictions(args.reference_csv)
    dataset = dataset_for(config, 'val', native_target=True)
    classes = load_classes(Path(config['data_root']) / 'ImageClasses_FSC147.txt')
    validate_reference(reference, dataset, classes)
    manifest = select_samples(reference)
    if args.smoke:
        manifest = [r for r in manifest if r['image_id'] in ANCHORS]
    write_json(directory / 'samples.json', manifest)
    set_seed(config['seed'])
    model = build_model(config).eval()
    restore_checkpoint(args.checkpoint, model, restore_rng=False)
    before = state_digest(model)
    del state
    table, caches = [], {}
    modes = INTERVENTIONS if args.experiment == 'all' else tuple(dict.fromkeys(('D0', args.experiment)))
    for mode in modes:
        current = dict(config, image_size=896 if mode == 'D1' else 448)
        source = dataset_for(current, 'val', native_target=True)
        rows, cached, timing = evaluate_variant(model, source, current, 33, mode,
                                                {r['image_id'] for r in manifest}, directory / mode,
                                                classes, args.smoke)
        if mode == 'D0':
            ref = [r for r in reference if r['image_id'] in {x['image_id'] for x in rows}]
            gate, differences = compare_reference(rows, ref)
            write_json(directory / 'reproduction.json', gate)
            write_csv(directory / 'reference_differences.csv', differences)
            if not gate['passed']:
                write_json(directory / 'summary.json', {'status': 'reference_mismatch_no_attribution', 'reproduction': gate})
                write_experiment_report(directory, [], 'reference_mismatch_no_attribution')
                return 2
        table.append(dict(experiment=mode, **scoring(rows), **timing))
        caches[mode] = cached
    if state_digest(model) != before:
        raise RuntimeError('Intervention modified counting parameters or BatchNorm buffers')
    export_comparisons(dataset, caches, directory / 'samples')
    write_csv(directory / 'experiments.csv', table)
    status = 'smoke_passed_not_full_validation' if args.smoke else 'complete'
    write_json(directory / 'summary.json', {'status': status, 'config': config, 'provenance': provenance,
                                          'experiments': table, 'state_unchanged': True})
    write_experiment_report(directory, table, status)
    return 0


def probe_ids(dataset):
    groups = {}
    for name in sorted(dataset.ids):
        key = count_range(len(dataset.annotations[name]['points']))
        if len(groups.setdefault(key, [])) < 2:
            groups[key].append(name)
    return [name for group in groups.values() for name in group]


def gradient_probe(model, dataset, ids, config, epoch):
    model.eval()
    rows = []
    loader = loader_for(Subset(dataset, [dataset.ids.index(name) for name in ids]), config)
    for batch in loader:
        output = model(batch['image'].to(config['device']), batch['text'], epoch, return_diagnostics=True)
        logits = output['diagnostics']['mixed_logits']
        loss = counting_loss(output, batch['density'].to(config['device']), batch['count'].to(config['device']),
                             config['count_loss_mode'], 32)
        gradients = [torch.autograd.grad(loss[key], logits, retain_graph=True)[0].detach()
                     for key in ('density_loss', 'count_loss')]
        if any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError(f'Non-finite probe gradient: {batch["image_id"]}')
        for index, name in enumerate(batch['image_id']):
            rows.append({'image_id': name, 'target': float(batch['count'][index]),
                         'density_gradient_l2': float(gradients[0][index].norm()),
                         'count_gradient_l2': float(gradients[1][index].norm()),
                         'weighted_count_gradient_l2': float(.1 * gradients[1][index].norm())})
        del output, logits, loss, gradients
    return rows


def run_training(args, directory):
    base, _ = parse_config('train', ['--config', args.config, '--device', args.device])
    config = experiment_config(base, args.group)
    config['output_dir'] = str(directory)
    provenance = prepared_provenance(config)
    provenance['reference_sha256'] = sha256(args.reference_csv)
    reference = read_predictions(args.reference_csv)
    validation = dataset_for(config, 'val')
    classes = load_classes(Path(config['data_root']) / 'ImageClasses_FSC147.txt')
    validate_reference(reference, validation, classes)
    if config['workers'] == 0:
        raise ValueError('Controlled training requires workers>=1 to isolate augmentation RNG from model RNG')
    trainset = dataset_for(config, 'train', training=True)
    probeset = dataset_for(config, 'train')
    ids = probe_ids(probeset)
    set_seed(42)
    model = build_model(config)
    initial_digest = state_digest(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    start, best = 0, float('inf')
    if args.resume:
        state = read_checkpoint(args.resume)
        if state.get('experiment_group') != args.group:
            raise ValueError('Resume requires a checkpoint from the same controlled experiment group')
        if state.get('experiment_smoke') or state.get('experiment_profile'):
            raise ValueError('Smoke/profile checkpoints cannot initialize formal experiments')
        for key, value in config.items():
            if key not in ('device', 'output_dir', 'data_root', 'text_annotations', 'projection_weights',
                           'dino_repo', 'dino_weights', 'clip_weights', 'download_root') and state['config'].get(key) != value:
                raise ValueError(f'Resume cannot change experiment setting {key}')
        if Path(args.resume).resolve().parent != directory.resolve():
            raise ValueError('Resume in the original experiment directory to preserve its best checkpoint and history')
        old = read_json(directory / 'provenance.json')
        if {k: old[k] for k in provenance} != provenance:
            raise ValueError('Resume resources differ from recorded experiment provenance')
        restored = restore_checkpoint(args.resume, model, optimizer)
        start, best = restored['epoch'] + 1, restored['best_mae']
        del state, restored
    write_json(directory / 'probe_samples.json', ids)
    write_json(directory / 'config.json', config)
    write_json(directory / 'provenance.json', dict(provenance, initial_state_sha256=initial_digest))
    if args.smoke:
        trainset = Subset(trainset, list(range(min(8, len(trainset)))))
    last_epoch = start + 1 if args.smoke or args.command == 'profile' else 100
    if start > last_epoch:
        raise ValueError('No training epochs remain')
    epoch_times = []
    if str(config['device']).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(config['device'])
    for epoch in range(start, last_epoch):
        started = time.perf_counter()
        training = train_epoch(model, loader_for(trainset, config, True, epoch), optimizer,
                               config['device'], epoch, 4, count_loss_mode=config['count_loss_mode'],
                               density_supervision_size=32, collect_diagnostics=True)
        if args.smoke:
            validation_loader = loader_for(Subset(validation, list(range(min(4, len(validation))))), config)
        else:
            validation_loader = loader_for(validation, config)
        val_metrics, _ = evaluate(model, validation_loader, config['device'], epoch)
        probes = gradient_probe(model, probeset, ids, config, epoch)
        seconds = time.perf_counter() - started
        epoch_times.append(seconds)
        write_csv(directory / f'probe_epoch_{epoch:03d}.csv', probes)
        record = {'epoch': epoch, 'train': training, 'val': val_metrics, 'seconds': seconds}
        with (directory / 'history.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, allow_nan=False) + '\n')
        LOGGER.info('group=%s epoch=%d train=%s val=%s seconds=%.1f', args.group, epoch, training, val_metrics, seconds)
        improved = val_metrics['mae'] < best
        best = min(best, val_metrics['mae'])
        for name in (('best.pt', 'latest.pt') if improved else ('latest.pt',)):
            save_checkpoint(directory / name, model, optimizer, epoch, best, config,
                            metadata={'experiment_group': args.group,
                                      'experiment_smoke': args.smoke, 'experiment_profile': args.command == 'profile'})
    if args.smoke or args.command == 'profile':
        status = 'training_smoke_passed' if args.smoke else 'profile_complete_not_formal_training'
        write_json(directory / 'summary.json', {'status': status, 'epoch_seconds': epoch_times,
                   'recommended_time_minutes': None if args.smoke else math.ceil(max(epoch_times) * 100 * 1.3 / 60),
                   'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(config['device'])
                       if str(config['device']).startswith('cuda') else 0,
                   'group': args.group})
        return 0
    restore = restore_checkpoint(directory / 'best.pt', model, restore_rng=False)
    best_epoch = restore['epoch']
    del restore, optimizer
    training_peak = torch.cuda.max_memory_allocated(config['device']) if str(config['device']).startswith('cuda') else 0
    native = dataset_for(config, 'val', native_target=True)
    selected = {r['image_id'] for r in select_samples(reference)}
    rows, cached, timing = evaluate_variant(model, native, config, best_epoch, 'D0', selected,
                                            directory / 'best_validation', classes)
    for variant in cached.values():
        variant['name'] = args.group
    export_comparisons(native, {args.group: cached}, directory / 'samples')
    write_json(directory / 'summary.json', dict(status='complete', group=args.group, **scoring(rows),
               best_epoch=best_epoch, initial_state_sha256=initial_digest,
               training_peak_gpu_allocated_bytes=training_peak,
               checkpoint_sha256=sha256(directory / 'best.pt'), **timing))
    write_experiment_report(directory, [dict(experiment=args.group, **scoring(rows))], 'complete')
    return 0


def summarize(args, directory):
    roots = [Path(path) for path in args.runs]
    summaries = [read_json(root / 'summary.json') for root in roots]
    if [s.get('group') for s in summaries] != list(GROUPS) or any(s['status'] != 'complete' for s in summaries):
        raise ValueError('--runs must be completed T0 T1 T2 runs in that order; smoke/profile results are excluded')
    provenance = [read_json(root / 'provenance.json') for root in roots]
    if any(p != provenance[0] for p in provenance[1:]):
        raise ValueError('Initial weights, software or prepared resources differ between training runs')
    configs = [read_json(root / 'config.json') for root in roots]
    baseline = {k: v for k, v in configs[0].items() if k not in ('image_size', 'count_loss_mode', 'output_dir')}
    for group, root, config, summary in zip(GROUPS, roots, configs, summaries):
        if {k: v for k, v in config.items() if k not in ('image_size', 'count_loss_mode', 'output_dir')} != baseline:
            raise ValueError('Training settings differ beyond the declared single factors')
        expected = experiment_config(config, group)
        if config != expected:
            raise ValueError(f'{group} config does not match the prescribed experiment')
        rows = read_predictions(root / 'best_validation/predictions.csv')
        if len(rows) != 1286:
            raise ValueError('Screening requires the full validation set')
        for key, value in scoring(rows).items():
            if abs(summary[key] - value) > 1e-6:
                raise ValueError(f'{group} summary disagrees with saved predictions: {key}')
    table = [dict(experiment=s['group'], **{k: s[k] for k in ('mae', 'rmse', 'high_mae', 'low_mae')}) for s in summaries]
    decisions = {s['group']: screen_results(summaries[0], s) for s in summaries[1:]}
    eligible = [s for s in summaries[1:] if decisions[s['group']]['eligible']]
    selected = min(eligible, key=lambda s: (s['mae'], s['group']))['group'] if eligible else 'T0'
    write_csv(directory / 'experiments.csv', table)
    write_json(directory / 'summary.json', {'status': 'complete', 'selection': selected, 'screening': decisions})
    # Read saved native arrays, never rerun models or normalize each variant independently.
    config = read_json(roots[0] / 'config.json')
    config.update(read_json(args.config))
    config['image_size'] = 448
    dataset = dataset_for(config, 'val')
    caches = {}
    for group, root in zip(GROUPS, roots):
        caches[group] = {}
        for sample_dir in sorted((root / 'samples').iterdir()):
            detail = read_json(sample_dir / 'diagnostics.json')['variants'][0]
            with np.load(sample_dir / (group + '_arrays.npz'), allow_pickle=False) as data:
                name = sample_dir.name + '.jpg'
                caches[group][name] = {'name': group, 'text': dataset.texts[name]['text_description'],
                    'count': detail['count'], 'routes': detail['routes'], 'density': data['density_grid'].copy(),
                    'similarity': data['similarity_grid'].copy(), 'indices': data['candidate_indices'].copy()}
    export_comparisons(dataset, caches, directory / 'samples')
    write_experiment_report(directory, table, 'complete', selected)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('intervene', 'train', 'profile', 'summarize'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--checkpoint', default='outputs/fsc147_text/best.pt')
    parser.add_argument('--reference-csv')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--experiment', choices=INTERVENTIONS + ('all',), default='all')
    parser.add_argument('--group', choices=GROUPS, default='T0')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume')
    parser.add_argument('--runs', nargs=3, metavar=('T0_DIR', 'T1_DIR', 'T2_DIR'))
    args = parser.parse_args(argv)
    configure_logging(args.output_dir)
    try:
        directory = Path(args.output_dir)
        if not args.resume and any(p.name != 'run.log' for p in directory.iterdir()):
            raise FileExistsError(f'Use a fresh experiment directory: {directory}')
        if args.resume and (args.command != 'train' or args.smoke):
            raise ValueError('Resume is only supported for formal experiment training')
        if args.command == 'summarize':
            if not args.runs:
                raise ValueError('Summarize requires --runs T0_DIR T1_DIR T2_DIR')
            return summarize(args, directory)
        if not args.reference_csv:
            raise ValueError('A complete formal --reference-csv is required')
        if args.command == 'intervene':
            return run_interventions(args, directory)
        return run_training(args, directory)
    except Exception:
        LOGGER.exception('FSC147 experiment failed command=%s group=%s', args.command, args.group)
        return 1
