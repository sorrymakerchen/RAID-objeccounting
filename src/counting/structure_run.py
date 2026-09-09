"""Predeclared A0-A3 experiments; validation-only architecture screening."""
import argparse
import functools
import logging
from pathlib import Path

import numpy as np
import torch

from dataset.fsc147 import read_json
from src.counting import ablation_run as shared
from src.counting.cli import build_model, configure_logging, write_json
from src.counting.diagnostic_run import sha256, write_csv
from src.counting.diagnostics import read_predictions
from src.counting.model import SpatialCountingHead

LOGGER = logging.getLogger(__name__)
GROUPS = ('A0', 'A1', 'A2', 'A3')


def structure_config(base, group, seed=42):
    if group not in GROUPS or seed not in (42, 43):
        raise ValueError('Structure experiments require A0-A3 and seed 42 or 43')
    config = shared.experiment_config(base, 'T0')
    config.update(head_type='spatial' if group in ('A1', 'A3') else 'raid',
                  local_count_weight=.1 if group in ('A2', 'A3') else 0.,
                  seed=seed, routing_seed=seed + 20000)
    return config


def structure_model(config):
    model = build_model(config)
    if config['head_type'] == 'spatial':
        # Initialize new modules independently while retaining the common projection.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config['seed'] + 10000)
            model.spatial_head = SpatialCountingHead(2 * config['reduced_dim'] + 1 + config['k'])
        model.to(config['device'])
    return model


def next_experiments(eligible):
    winners = [g for g in ('A1', 'A2') if eligible[g]]
    if len(winners) == 2:
        return [('A3', 42)]
    if len(winners) == 1:
        return [('A0', 43), (winners[0], 43)]
    return []


def check_training_gate(args):
    if not args.smoke_run or not args.profile_run:
        raise ValueError('Formal training requires --smoke-run and --profile-run')
    for root, status in ((args.smoke_run, 'training_smoke_passed'),
                         (args.profile_run, 'profile_complete_not_formal_training')):
        summary = read_json(Path(root) / 'summary.json')
        config = read_json(Path(root) / 'config.json')
        if summary['status'] != status or summary['group'] != args.group or config['seed'] != args.seed:
            raise ValueError(f'Wrong or incomplete preflight run: {root}')
    profile = read_json(Path(args.profile_run) / 'summary.json')
    if args.budget_hours <= 0 or profile['recommended_time_minutes'] > args.budget_hours * 60:
        raise ValueError('Profile estimate exceeds remaining budget; mark run pending, do not shorten training')


def summarize(args, directory):
    roots = [Path(p) for p in args.runs]
    summaries = [read_json(p / 'summary.json') for p in roots]
    groups = [s.get('group') for s in summaries]
    replication = groups in [['A0', 'A1'], ['A0', 'A2']]
    if not replication and groups not in [list(GROUPS[:3]), list(GROUPS)]:
        raise ValueError('Expected A0 A1 A2 [A3], or seed43 A0 and its selected variant')
    configs = [read_json(p / 'config.json') for p in roots]
    provenance = [read_json(p / 'provenance.json') for p in roots]
    common = None
    identity = None
    for group, root, config, summary, prov in zip(groups, roots, configs, summaries, provenance):
        if summary.get('status') != 'complete':
            raise ValueError(f'{group}: smoke/profile/incomplete runs cannot be screened')
        if config != structure_config(config, group, 43 if replication else 42):
            raise ValueError(f'{group}: config differs from the registered experiment definition')
        comparable = {k: v for k, v in config.items() if k not in
                      ('head_type', 'local_count_weight', 'output_dir')}
        if common is not None and comparable != common:
            raise ValueError('Experiment settings differ beyond declared factors')
        common = comparable
        resources = {k: v for k, v in prov.items() if k != 'initial_state_sha256'}
        if identity is not None and resources != identity:
            raise ValueError('Resources, source, or common initialization differ')
        identity = resources
        rows = read_predictions(root / 'best_validation/predictions.csv')
        if len(rows) != 1286 or len({r['image_id'] for r in rows}) != 1286:
            raise ValueError(f'{group}: full unique validation predictions required')
        if summary['checkpoint_sha256'] != sha256(root / 'best.pt'):
            raise ValueError(f'{group}: best checkpoint hash mismatch')
        for key, value in shared.scoring(rows).items():
            if value is None or not np.isfinite(value) or abs(summary[key] - value) > 1e-6:
                raise ValueError(f'{group}: summary/CSV disagreement for {key}')
    if 'A2' in groups and provenance[0]['initial_state_sha256'] != provenance[groups.index('A2')]['initial_state_sha256']:
        raise ValueError('A0 and A2 must have identical initial model state')
    decisions = {g: shared.screen_results(summaries[0], s) for g, s in zip(groups[1:], summaries[1:])}
    eligible = [s for s in summaries[1:] if decisions[s['group']]['eligible']]
    selected = min(eligible, key=lambda s: (s['mae'], s['trainable_parameters'], s['group']))['group'] if eligible else 'A0'
    tasks = [] if replication or 'A3' in groups else next_experiments(
        {g: decisions[g]['eligible'] for g in ('A1', 'A2')})
    table = [{k: s[k] for k in ('group', 'mae', 'rmse', 'high_mae', 'low_mae',
                                'best_epoch', 'trainable_parameters', 'training_seconds',
                                'training_peak_gpu_allocated_bytes')} for s in summaries]
    write_csv(directory / 'experiments.csv', table)
    # Reuse the native arrays so colors are shared across all variants of one image.
    config = dict(configs[0])
    config.update({k: v for k, v in read_json(args.config).items() if k in
                   ('data_root', 'text_annotations')})
    dataset = shared.dataset_for(config, 'val')
    predictions = [read_predictions(p / 'best_validation/predictions.csv') for p in roots]
    expected = {r['image_id']: (r['text'], r['target']) for r in predictions[0]}
    if set(expected) != set(dataset.ids) or any(
            {r['image_id']: (r['text'], r['target']) for r in rows} != expected for rows in predictions[1:]):
        raise ValueError('Validation identities/texts/targets differ')
    caches = {}
    for group, root in zip(groups, roots):
        caches[group] = {}
        for item in sorted((root / 'samples').iterdir()):
            detail = read_json(item / 'diagnostics.json')['variants'][0]
            with np.load(item / (group + '_arrays.npz'), allow_pickle=False) as data:
                image_id = item.name + '.jpg'
                caches[group][image_id] = dict(name=group, text=expected[image_id][0],
                    count=detail['count'], routes=detail['routes'], density=data['density_grid'].copy(),
                    similarity=data['similarity_grid'].copy(), indices=data['candidate_indices'].copy())
    if any(set(c) != set(caches[groups[0]]) for c in caches.values()):
        raise ValueError('Focus sample manifests differ')
    shared.export_comparisons(dataset, caches, directory / 'samples')
    lines = ['# RAID structure experiments', '', '| Group | MAE | RMSE | >=200 MAE | 0-24 MAE |',
             '|---|---:|---:|---:|---:|']
    lines += [f"| {s['group']} | {s['mae']:.4f} | {s['rmse']:.4f} | {s['high_mae']:.4f} | {s['low_mae']:.4f} |" for s in summaries]
    lines += ['', f'Selected for follow-up: {selected}.', f'Next runs (group, seed): {tasks}.',
              'Validation only; no SOTA or causal claim. Spatial heads have no applicable routing statistics.',
              'A1 changes the whole head and parameter count. A2 adds local supervision, not encoder quantity awareness.',
              'Budget: profile every new variant; run only when measured time * 100 * 1.3 fits the remaining 48-hour window.']
    (directory / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(directory / 'summary.json', dict(status='complete', selection=selected,
               replication=replication, screening=decisions,
               next_runs=[{'group': g, 'seed': s, 'status': 'pending_profile_and_budget_check'} for g, s in tasks]))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('smoke', 'profile', 'train', 'summarize'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--group', choices=GROUPS, default='A0')
    parser.add_argument('--seed', type=int, choices=(42, 43), default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--reference-csv')
    parser.add_argument('--resume')
    parser.add_argument('--smoke-run')
    parser.add_argument('--profile-run')
    parser.add_argument('--budget-hours', type=float, default=48.)
    parser.add_argument('--runs', nargs='+')
    args = parser.parse_args(argv)
    configure_logging(args.output_dir)
    try:
        directory = Path(args.output_dir)
        if args.resume and args.command != 'train':
            raise ValueError('Only formal training supports resume')
        if not args.resume and any(p.name != 'run.log' for p in directory.iterdir()):
            raise FileExistsError(f'Use a fresh output directory: {directory}')
        if args.command == 'summarize':
            if not args.runs:
                raise ValueError('Supply --runs A0 A1 A2 [A3]')
            return summarize(args, directory)
        if not args.reference_csv:
            raise ValueError('Provide the complete formal validation --reference-csv')
        if args.command == 'train' and not args.resume:
            check_training_gate(args)
            current, _ = shared.parse_config('train', ['--config', args.config, '--device', args.device])
            expected = structure_config(current, args.group, args.seed)
            for root in (args.smoke_run, args.profile_run):
                previous = read_json(Path(root) / 'config.json')
                if {k: v for k, v in previous.items() if k != 'output_dir'} != {
                        k: v for k, v in expected.items() if k != 'output_dir'}:
                    raise ValueError('Preflight config differs from requested formal training')
                previous_prov = read_json(Path(root) / 'provenance.json')
                current_prov = shared.prepared_provenance(expected)
                if any(previous_prov.get(k) != v for k, v in current_prov.items() if k != 'source_sha256'):
                    raise ValueError('Preflight resources have changed')
                project = Path(__file__).resolve().parents[2]
                if any(sha256(project / name) != value for name, value in previous_prov['source_sha256'].items()):
                    raise ValueError('Source changed after preflight; rerun smoke/profile')
                if previous_prov['reference_sha256'] != sha256(args.reference_csv):
                    raise ValueError('Reference predictions changed after preflight')
        args.smoke = args.command == 'smoke'
        return shared.run_training(args, directory,
                    config_factory=functools.partial(structure_config, seed=args.seed),
                    model_factory=structure_model)
    except Exception:
        LOGGER.exception('Structure experiment failed command=%s group=%s seed=%s', args.command, args.group, args.seed)
        return 1
