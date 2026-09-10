"""Predeclared structure and retrieval-input experiments on FSC147 validation."""
import argparse
import functools
import logging
import math
import statistics
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
A_GROUPS = ('A0', 'A1', 'A2', 'A3')
B_GROUPS = ('B0', 'B1')
GROUPS = A_GROUPS + B_GROUPS


def structure_config(base, group, seed=42):
    if group not in GROUPS or seed not in (42, 43, 44):
        raise ValueError('Structure experiments require A0-A3/B0-B1 and seed 42, 43, or 44')
    config = shared.experiment_config(base, 'T0')
    if group in B_GROUPS:
        config.update(head_type='spatial', local_count_weight=0., seed=seed,
                      routing_seed=seed + 20000,
                      retrieval_input_mode='full' if group == 'B0' else 'query_text')
    else:
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


def replication_decision(rows):
    """Apply the registered gate to three paired seeds without test-set access."""
    required = {(group, seed) for group in ('A0', 'A1') for seed in (42, 43, 44)}
    observed = [(row['group'], row['seed']) for row in rows]
    if set(observed) != required or len(observed) != len(required):
        raise ValueError('Replication requires exactly one A0/A1 result for each seed 42, 43, and 44')
    aggregates = {}
    for group in ('A0', 'A1'):
        group_rows = [row for row in rows if row['group'] == group]
        aggregates[group] = {}
        for metric in ('mae', 'rmse', 'high_mae', 'low_mae'):
            values = [row[metric] for row in group_rows]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f'Non-finite replication metric: {group} {metric}')
            aggregates[group][metric] = statistics.mean(values)
            aggregates[group][metric + '_std'] = statistics.stdev(values)
    checks = shared.screen_results(aggregates['A0'], aggregates['A1'])['checks']
    by_seed = {seed: {row['group']: row for row in rows if row['seed'] == seed}
               for seed in (42, 43, 44)}
    wins = sum(pair['A1']['mae'] < pair['A0']['mae'] for pair in by_seed.values())
    return {'eligible': all(checks.values()) and wins >= 2, 'checks': checks,
            'a1_mae_wins': wins, 'required_mae_wins': 2, 'aggregates': aggregates}


def retrieval_decision(rows):
    """Classify the contribution of retrieval inputs using three paired validation seeds."""
    required = {(group, seed) for group in B_GROUPS for seed in (42, 43, 44)}
    observed = [(row['group'], row['seed']) for row in rows]
    if set(observed) != required or len(observed) != len(required):
        raise ValueError('Retrieval comparison requires exactly one B0/B1 result for each seed 42, 43, and 44')
    aggregates = {}
    for group in B_GROUPS:
        group_rows = [row for row in rows if row['group'] == group]
        aggregates[group] = {}
        for metric in ('mae', 'rmse', 'high_mae', 'low_mae'):
            values = [row[metric] for row in group_rows]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f'Non-finite retrieval metric: {group} {metric}')
            aggregates[group][metric] = statistics.mean(values)
            aggregates[group][metric + '_std'] = statistics.stdev(values)
    pairs = {seed: {row['group']: row for row in rows if row['seed'] == seed}
             for seed in (42, 43, 44)}
    b0_wins = sum(pair['B0']['mae'] < pair['B1']['mae'] for pair in pairs.values())
    b1_wins = sum(pair['B1']['mae'] < pair['B0']['mae'] for pair in pairs.values())
    beneficial_checks = shared.screen_results(aggregates['B1'], aggregates['B0'])['checks']
    harmful_checks = shared.screen_results(aggregates['B0'], aggregates['B1'])['checks']
    b0, b1 = aggregates['B0'], aggregates['B1']
    equivalent_checks = {
        'mae': abs(b0['mae'] - b1['mae']) <= abs(b0['mae']) * .01,
        'rmse': abs(b0['rmse'] - b1['rmse']) <= abs(b0['rmse']) * .01,
        'high_count': abs(b0['high_mae'] - b1['high_mae']) <= abs(b0['high_mae']) * .05,
        'low_count': abs(b0['low_mae'] - b1['low_mae']) <= .5,
    }
    if all(beneficial_checks.values()) and b0_wins >= 2:
        conclusion, selection = 'retrieval_beneficial', 'B0'
    elif all(harmful_checks.values()) and b1_wins >= 2:
        conclusion, selection = 'retrieval_harmful', 'B1'
    elif all(equivalent_checks.values()):
        conclusion, selection = 'practically_equivalent', 'B1'
    else:
        conclusion, selection = 'inconclusive', 'B0'
    return {'conclusion': conclusion, 'selection': selection, 'b0_mae_wins': b0_wins,
            'b1_mae_wins': b1_wins, 'required_mae_wins': 2,
            'beneficial_checks': beneficial_checks, 'harmful_checks': harmful_checks,
            'equivalent_checks': equivalent_checks, 'aggregates': aggregates}


def _validated_run(root, expected_group=None, expected_seed=None):
    root = Path(root)
    summary = read_json(root / 'summary.json')
    config = read_json(root / 'config.json')
    provenance = read_json(root / 'provenance.json')
    group, seed = summary.get('group'), config.get('seed')
    if summary.get('status') != 'complete' or group not in ('A0', 'A1', 'B0', 'B1'):
        raise ValueError(f'{root}: completed paired formal run required')
    if expected_group is not None and (group, seed) != (expected_group, expected_seed):
        raise ValueError(f'{root}: expected {expected_group} seed {expected_seed}, got {group} seed {seed}')
    if config != structure_config(config, group, seed):
        raise ValueError(f'{root}: config differs from the registered definition')
    predictions = read_predictions(root / 'best_validation/predictions.csv')
    if len(predictions) != 1286 or len({row['image_id'] for row in predictions}) != 1286:
        raise ValueError(f'{root}: full unique validation predictions required')
    scores = shared.scoring(predictions)
    for key, value in scores.items():
        if value is None or not math.isfinite(value) or abs(summary[key] - value) > 1e-6:
            raise ValueError(f'{root}: summary disagrees with predictions for {key}')
    if summary['checkpoint_sha256'] != sha256(root / 'best.pt'):
        raise ValueError(f'{root}: best checkpoint hash mismatch')
    return summary, config, provenance, predictions


def summarize_replication(args, directory):
    if len(args.runs or ()) != 6:
        raise ValueError('Supply six runs: A0 A1 A0_s43 A1_s43 A0_s44 A1_s44')
    expected = [(group, seed) for seed in (42, 43, 44) for group in ('A0', 'A1')]
    runs = [_validated_run(root, group, seed)
            for root, (group, seed) in zip(args.runs, expected)]

    # The seed extension changes only this experiment wrapper. Training/model/data sources
    # and prepared resources must remain byte-identical to the original seed-42 runs.
    def comparable_provenance(value):
        value = dict(value)
        value.pop('initial_state_sha256', None)
        value.pop('shared_projection_sha256', None)
        sources = dict(value.get('source_sha256', {}))
        sources.pop('src/counting/structure_run.py', None)
        value['source_sha256'] = sources
        return value
    identity = comparable_provenance(runs[0][2])
    if any(comparable_provenance(run[2]) != identity for run in runs[1:]):
        raise ValueError('Prepared resources or behavior-defining source files differ across seeds')
    for offset, seed in zip((0, 2, 4), (42, 43, 44)):
        if runs[offset][2].get('shared_projection_sha256') != runs[offset + 1][2].get(
                'shared_projection_sha256'):
            raise ValueError(f'A0/A1 shared projection initialization differs for seed {seed}')

    reference = {row['image_id']: (row['text'], row['category'], row['target']) for row in runs[0][3]}
    if any({row['image_id']: (row['text'], row['category'], row['target']) for row in run[3]} != reference
           for run in runs[1:]):
        raise ValueError('Validation image IDs, prompts, categories, or targets differ across runs')

    table = []
    predictions = {}
    for (group, seed), (summary, _, _, rows) in zip(expected, runs):
        table.append({'group': group, 'seed': seed,
                      **{key: summary[key] for key in ('mae', 'rmse', 'high_mae', 'low_mae',
                                                       'best_epoch', 'trainable_parameters',
                                                       'training_seconds', 'training_peak_gpu_allocated_bytes')}})
        predictions[group, seed] = {row['image_id']: float(row['prediction']) for row in rows}
    decision = replication_decision(table)
    aggregates = []
    for group in ('A0', 'A1'):
        aggregate = decision['aggregates'][group]
        aggregates.append({'group': group, **aggregate,
                           'mae_wins': decision['a1_mae_wins'] if group == 'A1' else None})

    paired = []
    for image_id in sorted(reference):
        text_prompt, category, target_text = reference[image_id]
        target = float(target_text)
        p0 = statistics.mean(predictions['A0', seed][image_id] for seed in (42, 43, 44))
        p1 = statistics.mean(predictions['A1', seed][image_id] for seed in (42, 43, 44))
        paired.append({'image_id': image_id, 'text': text_prompt, 'category': category,
                       'target': target, 'a0_mean_prediction': p0, 'a1_mean_prediction': p1,
                       'a0_mean_absolute_error': statistics.mean(
                           abs(predictions['A0', seed][image_id] - target) for seed in (42, 43, 44)),
                       'a1_mean_absolute_error': statistics.mean(
                           abs(predictions['A1', seed][image_id] - target) for seed in (42, 43, 44))})
        paired[-1]['a1_error_reduction'] = (paired[-1]['a0_mean_absolute_error'] -
                                            paired[-1]['a1_mean_absolute_error'])

    write_csv(directory / 'seed_results.csv', table)
    write_csv(directory / 'aggregate_results.csv', aggregates)
    write_csv(directory / 'paired_predictions.csv', paired)
    selected = 'A1' if decision['eligible'] else 'A0'
    write_json(directory / 'summary.json', {'status': 'complete', 'selection': selected,
               'seeds': [42, 43, 44], 'decision': decision,
               'source_compatibility': {'structure_wrapper_hash_ignored': True,
                   'reason': 'seed 44 support changed validation/CLI only; model, loss, data, and training sources matched'},
               'initialization_check': 'A0/A1 shared projection matched within each seed'})
    lines = ['# RAID A0/A1 three-seed replication', '',
             '| Group | MAE mean±SD | RMSE mean±SD | >=200 MAE mean±SD | 0-24 MAE mean±SD |',
             '|---|---:|---:|---:|---:|']
    for row in aggregates:
        lines.append(f"| {row['group']} | {row['mae']:.4f}±{row['mae_std']:.4f} | "
                     f"{row['rmse']:.4f}±{row['rmse_std']:.4f} | "
                     f"{row['high_mae']:.4f}±{row['high_mae_std']:.4f} | "
                     f"{row['low_mae']:.4f}±{row['low_mae_std']:.4f} |")
    lines += ['', f"A1 validation-MAE wins: {decision['a1_mae_wins']}/3.",
              f'Selection: {selected}. Checks: {decision["checks"]}.',
              'The registered mean-metric gate and at least two paired MAE wins are both required.',
              'Validation only; the test set was not read.']
    (directory / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return 0


def summarize_retrieval(args, directory):
    if len(args.runs or ()) != 6:
        raise ValueError('Supply six runs: B0 B1 B0_s43 B1_s43 B0_s44 B1_s44')
    expected = [(group, seed) for seed in (42, 43, 44) for group in B_GROUPS]
    runs = [_validated_run(root, group, seed)
            for root, (group, seed) in zip(args.runs, expected)]

    common_config = None
    for _, config, _, _ in runs:
        comparable = {key: value for key, value in config.items()
                      if key not in ('output_dir', 'retrieval_input_mode', 'seed', 'routing_seed')}
        if common_config is not None and comparable != common_config:
            raise ValueError('B0/B1 settings differ beyond retrieval mode, seed, routing seed, or output directory')
        common_config = comparable

    def comparable_provenance(value):
        value = dict(value)
        value.pop('initial_state_sha256', None)
        value.pop('shared_projection_sha256', None)
        return value

    identity = comparable_provenance(runs[0][2])
    if any(comparable_provenance(run[2]) != identity for run in runs[1:]):
        raise ValueError('Prepared resources or behavior-defining source files differ across retrieval runs')
    for offset, seed in zip((0, 2, 4), (42, 43, 44)):
        first, second = runs[offset][2], runs[offset + 1][2]
        if first.get('shared_projection_sha256') != second.get('shared_projection_sha256'):
            raise ValueError(f'B0/B1 shared projection initialization differs for seed {seed}')
        if first.get('initial_state_sha256') != second.get('initial_state_sha256'):
            raise ValueError(f'B0/B1 initial trainable counting state differs for seed {seed}')

    reference = {row['image_id']: (row['text'], row['category'], row['target']) for row in runs[0][3]}
    if any({row['image_id']: (row['text'], row['category'], row['target']) for row in run[3]} != reference
           for run in runs[1:]):
        raise ValueError('Validation image IDs, prompts, categories, or targets differ across retrieval runs')

    table, predictions = [], {}
    for (group, seed), (summary, _, _, rows) in zip(expected, runs):
        table.append({'group': group, 'seed': seed,
                      **{key: summary[key] for key in ('mae', 'rmse', 'high_mae', 'low_mae',
                                                       'best_epoch', 'trainable_parameters',
                                                       'training_seconds', 'training_peak_gpu_allocated_bytes')}})
        predictions[group, seed] = {row['image_id']: float(row['prediction']) for row in rows}
    decision = retrieval_decision(table)
    aggregates = []
    for group in B_GROUPS:
        aggregate = decision['aggregates'][group]
        aggregates.append({'group': group, **aggregate,
                           'mae_wins': decision[group.lower() + '_mae_wins']})

    paired = []
    for image_id in sorted(reference):
        text_prompt, category, target_text = reference[image_id]
        target = float(target_text)
        b0_errors = [abs(predictions['B0', seed][image_id] - target) for seed in (42, 43, 44)]
        b1_errors = [abs(predictions['B1', seed][image_id] - target) for seed in (42, 43, 44)]
        paired.append({'image_id': image_id, 'text': text_prompt, 'category': category,
                       'target': target,
                       'b0_mean_prediction': statistics.mean(
                           predictions['B0', seed][image_id] for seed in (42, 43, 44)),
                       'b1_mean_prediction': statistics.mean(
                           predictions['B1', seed][image_id] for seed in (42, 43, 44)),
                       'b0_mean_absolute_error': statistics.mean(b0_errors),
                       'b1_mean_absolute_error': statistics.mean(b1_errors),
                       'b0_error_reduction': statistics.mean(b1_errors) - statistics.mean(b0_errors)})

    write_csv(directory / 'seed_results.csv', table)
    write_csv(directory / 'aggregate_results.csv', aggregates)
    write_csv(directory / 'paired_predictions.csv', paired)
    write_json(directory / 'summary.json', {'status': 'complete',
               'selection': decision['selection'], 'conclusion': decision['conclusion'],
               'seeds': [42, 43, 44], 'decision': decision,
               'initialization_check': 'B0/B1 trainable counting state and shared projection matched within each seed',
               'test_set_accessed': False})
    lines = ['# RAID B0/B1 retrieval-input ablation', '',
             '| Group | MAE mean±SD | RMSE mean±SD | >=200 MAE mean±SD | 0-24 MAE mean±SD | MAE wins |',
             '|---|---:|---:|---:|---:|---:|']
    for row in aggregates:
        lines.append(f"| {row['group']} | {row['mae']:.4f}±{row['mae_std']:.4f} | "
                     f"{row['rmse']:.4f}±{row['rmse_std']:.4f} | "
                     f"{row['high_mae']:.4f}±{row['high_mae_std']:.4f} | "
                     f"{row['low_mae']:.4f}±{row['low_mae_std']:.4f} | {row['mae_wins']}/3 |")
    lines += ['', f"Conclusion: {decision['conclusion']}.",
              f"Selected representation for follow-up: {decision['selection']}.",
              f"Beneficial checks: {decision['beneficial_checks']}.",
              f"Harmful checks: {decision['harmful_checks']}.",
              f"Practical-equivalence checks: {decision['equivalent_checks']}.",
              'B0 supplies query, reconstructed retrieval, text similarity, and 150 matching channels.',
              'B1 keeps the same head and parameter count but zeros reconstructed retrieval and matching channels.',
              'Validation only; the test set was not read and this experiment does not establish SOTA.']
    (directory / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return 0


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
    if not replication and groups not in [list(A_GROUPS[:3]), list(A_GROUPS)]:
        raise ValueError('Expected A0 A1 A2 [A3], or seed43 A0 and its selected variant')
    configs = [read_json(p / 'config.json') for p in roots]
    provenance = [read_json(p / 'provenance.json') for p in roots]
    replication_seed = configs[0].get('seed') if replication else 42
    if replication and any(config.get('seed') != replication_seed for config in configs):
        raise ValueError('Paired replication runs must use the same seed')
    common = None
    identity = None
    for group, root, config, summary, prov in zip(groups, roots, configs, summaries, provenance):
        if summary.get('status') != 'complete':
            raise ValueError(f'{group}: smoke/profile/incomplete runs cannot be screened')
        if config != structure_config(config, group, replication_seed if replication else 42):
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
    parser.add_argument('command', choices=('smoke', 'profile', 'train', 'summarize',
                                            'summarize-replication', 'summarize-retrieval'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--group', choices=GROUPS, default='A0')
    parser.add_argument('--seed', type=int, choices=(42, 43, 44), default=42)
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
        if args.command in ('summarize', 'summarize-replication', 'summarize-retrieval'):
            if not args.runs:
                raise ValueError('Supply the completed formal run directories with --runs')
            function = {'summarize': summarize,
                        'summarize-replication': summarize_replication,
                        'summarize-retrieval': summarize_retrieval}[args.command]
            return function(args, directory)
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
