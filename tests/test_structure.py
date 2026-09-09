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
from src.counting.structure_run import structure_config, next_experiments, check_training_gate
from src.counting import ablation_run as run
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
        self.assertEqual(next_experiments({'A1': True, 'A2': True}), [('A3', 42)])
        self.assertEqual(next_experiments({'A1': False, 'A2': True}), [('A0', 43), ('A2', 43)])
        self.assertEqual(next_experiments({'A1': False, 'A2': False}), [])
