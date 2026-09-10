"""Behavior contracts for the structure experiments."""
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import json
from pathlib import Path

import torch

from test_counting import FakeEncoder
from src.counting.model import RAIDCounter, CountingRouter
from src.counting.training import local_count_loss, save_checkpoint, restore_checkpoint, counting_loss
from src.counting.structure_run import (structure_config, next_experiments, check_training_gate,
                                        replication_decision, summarize_replication,
                                        retrieval_decision, summarize_retrieval)
from src.counting import ablation_run as run
from src.counting.diagnostic_run import sha256, write_csv
from test_diagnostics import SyntheticDiagnosticDataset


class GridEncoder(FakeEncoder):
    def forward(self, images, texts):
        features, text = super().forward(images, texts)
        return torch.nn.functional.interpolate(features, (32, 32)), text


class StructureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_preflight_budget_and_group_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, status in [('smoke', 'training_smoke_passed'),
                                 ('profile', 'profile_complete_not_formal_training')]:
                directory = Path(tmp) / name
                directory.mkdir()
                (directory / 'summary.json').write_text(json.dumps(dict(
                    status=status, group='A0', recommended_time_minutes=120)))
                (directory / 'config.json').write_text(json.dumps(dict(seed=42)))
            args = SimpleNamespace(smoke_run=str(Path(tmp) / 'smoke'),
                                   profile_run=str(Path(tmp) / 'profile'), group='A0', seed=42, budget_hours=3)
            check_training_gate(args)
            args.budget_hours = 1
            with self.assertRaisesRegex(ValueError, 'budget'):
                check_training_gate(args)
            args.budget_hours, args.group = 3, 'A1'
            with self.assertRaisesRegex(ValueError, 'preflight'):
                check_training_gate(args)

    def test_loss_switch_only_adds_registered_local_term(self):
        density = torch.rand(2, 1, 32, 32, requires_grad=True)
        output = dict(density=density, count=density.sum((1, 2, 3)), balance_loss=torch.tensor(1.))
        target = torch.rand_like(density)
        counts = target.sum((1, 2, 3))
        baseline = counting_loss(output, target, counts)
        variant = counting_loss(output, target, counts, local_count_weight=.1)
        torch.testing.assert_close(variant['loss'], baseline['loss'] + .1 * local_count_loss(density, target))
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            counting_loss(output, target, counts, local_count_weight=-1)

    def test_data_order_and_flips_independent_of_model_rng(self):
        # FSC flips use Python random in workers; sampler owns a separate generator.
        from test_counting import DataTests
        fixture = DataTests()
        fixture.setUp()
        try:
            from dataset.fsc147 import FSC147Dataset
            dataset = FSC147Dataset(fixture.root, fixture.root / 'FSC-147-D.json', 'train', image_size=448)
            config = dict(seed=42, workers=1, batch_size=2)
            first = list(run.loader_for(dataset, config, True, 3))
            torch.rand(300)
            second = list(run.loader_for(dataset, config, True, 3))
            self.assertEqual(first[0]['image_id'], second[0]['image_id'])
            torch.testing.assert_close(first[0]['image'], second[0]['image'], rtol=0, atol=0)
            torch.testing.assert_close(first[0]['density'], second[0]['density'], rtol=0, atol=0)
        finally:
            fixture.tmp.cleanup()

    def test_all_structure_smokes_and_spatial_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = SyntheticDiagnosticDataset(tmp)
            base = dict(data_root=tmp, device='cpu', workers=1)
            for group in ('A0', 'A1', 'A2', 'A3'):
                directory = Path(tmp) / group
                directory.mkdir()
                args = SimpleNamespace(config='fixture', device='cpu', reference_csv='fixture',
                                       group=group, resume=None, smoke=True, command='smoke')
                models = []
                def factory(config):
                    model = RAIDCounter(GridEncoder(), feature_dim=4, reduced_dim=4, k=150,
                                        expert_dim=4, head_type=config['head_type'],
                                        routing_seed=config['routing_seed'])
                    models.append(model)
                    return model
                with patch.object(run, 'parse_config', return_value=(base, {})), \
                     patch.object(run, 'prepared_provenance', return_value={'source_sha256': {}}), \
                     patch.object(run, 'sha256', return_value='fixture'), \
                     patch.object(run, 'read_predictions', return_value=[]), \
                     patch.object(run, 'validate_reference'), \
                     patch.object(run, 'load_classes', return_value={}), \
                     patch.object(run, 'dataset_for', return_value=dataset), \
                     patch.object(run, 'loader_for', side_effect=lambda data, config, *args:
                                  torch.utils.data.DataLoader(data, batch_size=2)):
                    self.assertEqual(run.run_training(args, directory, structure_config, factory), 0)
                record = json.loads((directory / 'history.jsonl').read_text())
                self.assertEqual(record['train']['components']['local_count_loss'] > 0, group in ('A2', 'A3'))
                rows, cached, _ = run.evaluate_variant(models[0], dataset,
                        dict(device='cpu', workers=0, batch_size=2), 0, 'D0', set(dataset.ids), directory / 'eval')
                run.export_comparisons(dataset, {group: cached}, directory / 'samples')
                if group in ('A1', 'A3'):
                    self.assertIsNone(rows[0]['stage1_selected_0'])
                    self.assertEqual(record['train']['routes'], {})

    def test_next_training_update_restores_router_and_optimizer(self):
        model = RAIDCounter(FakeEncoder(), feature_dim=4, reduced_dim=4, k=10,
                            expert_dim=4, routing_seed=99, warmup_epochs=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        images = torch.randn(2, 3, 16, 16)
        def step():
            model.train()
            optimizer.zero_grad()
            model(images, ['red', 'green'])['count'].sum().backward()
            optimizer.step()
        step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'latest.pt'
            save_checkpoint(path, model, optimizer, 0, 1., {})
            step()
            expected = {k: v.clone() for k, v in model.state_dict().items()}
            restore_checkpoint(path, model, optimizer)
            torch.rand(100)
            step()
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)

    def test_local_mass_and_location(self):
        target = torch.zeros(1, 1, 32, 32)
        target[..., 0, 0] = 4
        self.assertEqual(local_count_loss(target, target).item(), 0)
        moved = torch.roll(target, 16, -1).requires_grad_()
        loss = local_count_loss(moved, target)
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(moved.grad).all())

    def test_spatial_prediction_and_checkpoint(self):
        model = RAIDCounter(FakeEncoder(), feature_dim=4, reduced_dim=4,
                            k=10, expert_dim=4, head_type='spatial').eval()
        images = torch.randn(2, 3, 16, 16)
        output = model(images, ['red', 'green'], return_diagnostics=True)
        self.assertEqual(output['density'].shape, (2, 1, 4, 4))
        self.assertTrue((output['density'] >= 0).all())
        torch.testing.assert_close(output['count'], output['density'].sum((1, 2, 3)))
        self.assertIsNone(output['diagnostics']['stage1_weights'])
        output['count'].sum().backward()
        self.assertIsNone(model.encoder.scale.grad)
        self.assertIsNotNone(model.projection.weight.grad)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'best.pt'
            save_checkpoint(path, model, None, 0, 1., {'head_type': 'spatial'})
            restore_checkpoint(path, model)
            torch.testing.assert_close(output['density'], model(images, ['red', 'green'])['density'])

    def test_retrieval_ablation_zeros_only_retrieved_inputs(self):
        full = RAIDCounter(FakeEncoder(), feature_dim=4, reduced_dim=4, k=10,
                           expert_dim=4, head_type='spatial').eval()
        reduced = RAIDCounter(FakeEncoder(), feature_dim=4, reduced_dim=4, k=10,
                              expert_dim=4, head_type='spatial',
                              retrieval_input_mode='query_text').eval()
        reduced.load_state_dict(full.state_dict())
        captured = []
        hook = reduced.spatial_head.register_forward_pre_hook(
            lambda module, inputs: captured.append(tuple(value.detach().clone() for value in inputs)))
        images = torch.randn(2, 3, 16, 16)
        baseline = full(images, ['red', 'green'], return_diagnostics=True)
        ablated = reduced(images, ['red', 'green'], return_diagnostics=True)
        hook.remove()
        guidance, matching = captured[0]
        self.assertTrue(torch.equal(matching, torch.zeros_like(matching)))
        self.assertTrue(torch.equal(guidance[:, 4:8], torch.zeros_like(guidance[:, 4:8])))
        self.assertFalse(torch.equal(guidance[:, :4], torch.zeros_like(guidance[:, :4])))
        torch.testing.assert_close(baseline['similarity'], ablated['similarity'])
        self.assertTrue(torch.equal(baseline['indices'], ablated['indices']))
        self.assertFalse(torch.equal(baseline['density'], ablated['density']))

    def test_router_stream_survives_unrelated_rng_and_restore(self):
        router = CountingRouter(4, 3)
        router.set_noise_seed(123)
        x = torch.randn(2, 4, 4, 4)
        saved = router.state_dict()
        saved = {k: v.clone() for k, v in saved.items()}
        expected = router(x)
        torch.rand(500)
        router.load_state_dict(saved)
        torch.testing.assert_close(expected, router(x), rtol=0, atol=0)

    def test_definitions_and_selection(self):
        self.assertEqual(structure_config({}, 'A2')['local_count_weight'], .1)
        self.assertEqual(structure_config({}, 'A1')['head_type'], 'spatial')
        self.assertEqual(structure_config({}, 'A1', 44)['seed'], 44)
        self.assertEqual(next_experiments({'A1': True, 'A2': True}), [('A3', 42)])
        self.assertEqual(next_experiments({'A1': False, 'A2': True}), [('A0', 43), ('A2', 43)])
        self.assertEqual(next_experiments({'A1': False, 'A2': False}), [])
        b0, b1 = structure_config({}, 'B0'), structure_config({}, 'B1')
        self.assertEqual([key for key in b0 if b0[key] != b1[key]], ['retrieval_input_mode'])

    def test_retrieval_decision_distinguishes_useful_harmful_and_equivalent(self):
        def rows(b0, b1):
            result = []
            for seed in (42, 43, 44):
                for group, values in (('B0', b0), ('B1', b1)):
                    result.append(dict(group=group, seed=seed, mae=values[0], rmse=values[1],
                                       high_mae=values[2], low_mae=values[3]))
            return result
        useful = retrieval_decision(rows((20, 50, 100, 4), (22, 52, 110, 5)))
        self.assertEqual(useful['conclusion'], 'retrieval_beneficial')
        harmful = retrieval_decision(rows((22, 52, 110, 5), (20, 50, 100, 4)))
        self.assertEqual(harmful['conclusion'], 'retrieval_harmful')
        equivalent = retrieval_decision(rows((20, 50, 100, 4), (20.1, 50.2, 102, 4.2)))
        self.assertEqual(equivalent['conclusion'], 'practically_equivalent')

    def test_replication_requires_mean_gate_and_two_seed_wins(self):
        rows = []
        for seed, a0_mae, a1_mae in ((42, 30., 27.), (43, 31., 28.), (44, 29., 30.)):
            rows += [dict(group='A0', seed=seed, mae=a0_mae, rmse=100.,
                          high_mae=300., low_mae=6.),
                     dict(group='A1', seed=seed, mae=a1_mae, rmse=98.,
                          high_mae=280., low_mae=6.1)]
        decision = replication_decision(rows)
        self.assertEqual(decision['a1_mae_wins'], 2)
        self.assertTrue(decision['eligible'])
        rows[-1]['high_mae'] = 400.
        decision = replication_decision(rows)
        self.assertFalse(decision['eligible'])
        self.assertFalse(decision['checks']['high_count'])

    def test_replication_rejects_missing_or_duplicate_seed_pair(self):
        rows = [dict(group='A0', seed=42, mae=1., rmse=1., high_mae=1., low_mae=1.),
                dict(group='A1', seed=42, mae=.5, rmse=.5, high_mae=.5, low_mae=.5)] * 3
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            replication_decision(rows)

    def test_replication_summary_recomputes_six_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = []
            for seed in (42, 43, 44):
                for group in ('A0', 'A1'):
                    root = Path(tmp) / f'{group}_{seed}'
                    (root / 'best_validation').mkdir(parents=True)
                    config = structure_config({'output_dir': str(root)}, group, seed)
                    predictions = []
                    error = 10 if group == 'A0' else 8
                    for index in range(1286):
                        target = 300. if index < 73 else 10.
                        predictions.append({'image_id': f'{index}.jpg', 'text': 'the dots',
                                            'category': f'category-{index % 29}', 'target': target,
                                            'prediction': target + error, 'absolute_error': error})
                    write_csv(root / 'best_validation/predictions.csv', predictions)
                    scores = run.scoring(predictions)
                    (root / 'best.pt').write_bytes(f'{group}-{seed}'.encode())
                    summary = dict(status='complete', group=group, **scores, best_epoch=50,
                                   trainable_parameters=2 if group == 'A0' else 1,
                                   training_seconds=1., training_peak_gpu_allocated_bytes=1,
                                   checkpoint_sha256=sha256(root / 'best.pt'))
                    (root / 'config.json').write_text(json.dumps(config))
                    (root / 'summary.json').write_text(json.dumps(summary))
                    provenance = {'initial_state_sha256': f'{group}-{seed}',
                                  'shared_projection_sha256': f'projection-{seed}',
                                  'files': {'weights': 'same'},
                                  'source_sha256': {'src/counting/model.py': 'same',
                                                    'src/counting/structure_run.py': f'wrapper-{seed}'}}
                    (root / 'provenance.json').write_text(json.dumps(provenance))
                    roots.append(str(root))
            output = Path(tmp) / 'comparison'
            output.mkdir()
            self.assertEqual(summarize_replication(SimpleNamespace(runs=roots), output), 0)
            result = json.loads((output / 'summary.json').read_text())
            self.assertEqual(result['selection'], 'A1')
            self.assertEqual(result['decision']['a1_mae_wins'], 3)
            self.assertEqual(len((output / 'paired_predictions.csv').read_text().splitlines()), 1287)

    def test_retrieval_summary_recomputes_six_runs_and_checks_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = []
            for seed in (42, 43, 44):
                for group in ('B0', 'B1'):
                    root = Path(tmp) / f'{group}_{seed}'
                    (root / 'best_validation').mkdir(parents=True)
                    config = structure_config({'output_dir': str(root)}, group, seed)
                    error = 8 if group == 'B0' else 10
                    predictions = []
                    for index in range(1286):
                        target = 300. if index < 73 else 10.
                        predictions.append({'image_id': f'{index}.jpg', 'text': 'the dots',
                                            'category': f'category-{index % 29}', 'target': target,
                                            'prediction': target + error, 'absolute_error': error})
                    write_csv(root / 'best_validation/predictions.csv', predictions)
                    scores = run.scoring(predictions)
                    (root / 'best.pt').write_bytes(f'{group}-{seed}'.encode())
                    summary = dict(status='complete', group=group, **scores, best_epoch=50,
                                   trainable_parameters=1, training_seconds=1.,
                                   training_peak_gpu_allocated_bytes=1,
                                   checkpoint_sha256=sha256(root / 'best.pt'))
                    (root / 'config.json').write_text(json.dumps(config))
                    (root / 'summary.json').write_text(json.dumps(summary))
                    provenance = {'initial_state_sha256': f'initial-{seed}',
                                  'shared_projection_sha256': f'projection-{seed}',
                                  'files': {'weights': 'same'},
                                  'source_sha256': {'src/counting/model.py': 'same',
                                                    'src/counting/structure_run.py': 'same'}}
                    (root / 'provenance.json').write_text(json.dumps(provenance))
                    roots.append(str(root))
            output = Path(tmp) / 'comparison'
            output.mkdir()
            self.assertEqual(summarize_retrieval(SimpleNamespace(runs=roots), output), 0)
            result = json.loads((output / 'summary.json').read_text())
            self.assertEqual(result['selection'], 'B0')
            self.assertEqual(result['conclusion'], 'retrieval_beneficial')
            self.assertFalse(result['test_set_accessed'])
            self.assertEqual(len((output / 'paired_predictions.csv').read_text().splitlines()), 1287)
