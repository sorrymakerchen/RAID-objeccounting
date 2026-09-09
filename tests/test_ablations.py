import unittest
import tempfile
from pathlib import Path
import json
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

import torch

from test_counting import small_counter
from src.counting.model import spatial_candidate_indices
from src.counting.training import counting_loss, sum_pool_density, save_checkpoint
from src.counting.ablation_run import (screen_results, experiment_config, evaluate_variant,
                                       export_comparisons, gradient_probe, main)
from src.counting.cli import parse_config
from src.counting.diagnostics import candidate_rectangles
from test_diagnostics import SyntheticDiagnosticDataset, SyntheticDiagnosticModel


class AblationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_spatial_candidates_unique_balanced_and_stable(self):
        scores = torch.zeros(2, 1024)
        indices = spatial_candidate_indices(scores, 32, 32)
        self.assertEqual(indices.shape, (2, 150))
        self.assertEqual(len(set(indices[0].tolist())), 150)
        self.assertTrue(torch.equal(indices, indices.sort(-1).values))
        for y in range(5):
            for x in range(5):
                iy, ix = indices[0] // 32, indices[0] % 32
                selected = ((iy >= y * 32 // 5) & (iy < (y + 1) * 32 // 5)
                            & (ix >= x * 32 // 5) & (ix < (x + 1) * 32 // 5))
                self.assertEqual(int(selected.sum()), 6)

    def test_pool_preserves_mass_and_gradient(self):
        density = torch.rand(2, 1, 64, 64, requires_grad=True)
        pooled = sum_pool_density(density, 32)
        torch.testing.assert_close(pooled.sum((1, 2, 3)), density.sum((1, 2, 3)))
        pooled.sum().backward()
        self.assertTrue(torch.equal(density.grad, torch.ones_like(density)))
        with self.assertRaises(ValueError):
            sum_pool_density(density, 30)

    def test_absolute_count_loss_has_no_target_denominator(self):
        for target in (10., 1000.):
            output = {'density': torch.zeros(1, 1, 32, 32),
                      'count': torch.tensor([target + 5]), 'balance_loss': torch.tensor(0.)}
            loss = counting_loss(output, output['density'], torch.tensor([target]), 'absolute')
            self.assertEqual(loss['count_loss'].item(), 5.)
        with self.assertRaises(ValueError):
            counting_loss(output, output['density'], torch.tensor([target]), 'invalid')

    def test_routes_logits_and_state_preserved(self):
        model = small_counter().eval()
        images = torch.rand(2, 3, 56, 56)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            baseline = model(images, ['dots'] * 2, 2)
            for intervention in ('D0', 'D3', 'D4'):
                result = model(images, ['dots'] * 2, 2, True, intervention=intervention)
                d = result['diagnostics']
                expected = (d['stage2_expert_logits'] * d['stage2_weights'][:, :, None]).sum(1)
                torch.testing.assert_close(d['mixed_logits'], expected)
                torch.testing.assert_close(result['density'].flatten(1), torch.nn.functional.softplus(expected))
                if intervention == 'D0':
                    self.assertTrue(torch.equal(result['density'], baseline['density']))
                if intervention == 'D3':
                    torch.testing.assert_close(d['stage1_weights'], torch.tensor([[0., .5, .5]]).expand(2, -1))
                if intervention == 'D4':
                    torch.testing.assert_close(d['stage2_weights'], torch.full((2, 3), 1 / 3))
            restored = model(images, ['dots'] * 2, 2)
        self.assertTrue(torch.equal(restored['density'], baseline['density']))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)

    def test_training_definitions_are_single_factor(self):
        base = experiment_config({}, 'T0')
        for name, key in [('T1', 'count_loss_mode'), ('T2', 'image_size')]:
            variant = experiment_config({}, name)
            self.assertEqual([k for k in base if base[k] != variant[k]], [key])

    def test_screen_requires_all_thresholds(self):
        base = {'mae': 10., 'rmse': 20., 'high_mae': 100., 'low_mae': 4.}
        good = dict(base, mae=9.8, high_mae=94.)
        self.assertTrue(screen_results(base, good)['eligible'])
        self.assertFalse(screen_results(base, dict(good, low_mae=4.6))['eligible'])

    def test_workflow_exports_only_observed_predictions(self):
        class Model(SyntheticDiagnosticModel):
            def forward(self, *args, **kwargs):
                kwargs.pop('intervention', None)
                return super().forward(*args, **kwargs)
        with tempfile.TemporaryDirectory() as root:
            dataset = SyntheticDiagnosticDataset(root)
            config = {'device': 'cpu', 'batch_size': 2, 'workers': 0}
            rows, cached, timing = evaluate_variant(Model(), dataset, config, 33, 'D0',
                                                   set(dataset.ids), Path(root) / 'D0')
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]['prediction'], 1.)
            export_comparisons(dataset, {'D0': cached}, Path(root) / 'panels')
            self.assertTrue((Path(root) / 'panels/3425/comparison.png').is_file())
            self.assertGreaterEqual(timing['seconds'], 0)

    def test_mixed_resolution_panels_use_native_candidates(self):
        with tempfile.TemporaryDirectory() as root:
            dataset = SyntheticDiagnosticDataset(root)
            model = SyntheticDiagnosticModel()
            config = {'device': 'cpu', 'batch_size': 2, 'workers': 0}
            class Wrapped(SyntheticDiagnosticModel):
                def forward(self, *args, **kwargs):
                    kwargs.pop('intervention', None)
                    return super().forward(*args, **kwargs)
            _, cache, _ = evaluate_variant(Wrapped(), dataset, config, 33, 'D0', set(dataset.ids), Path(root) / 'out')
            second = {}
            for name, original in cache.items():
                second[name] = dict(original, name='D1', density=np.full((64, 64), original['count'] / 4096),
                                    similarity=np.ones((64, 64)), indices=np.arange(3946, 4096))
            export_comparisons(dataset, {'D0': cache, 'D1': second}, Path(root) / 'panels')
            with np.load(Path(root) / 'panels/3425/D1_arrays.npz') as saved:
                self.assertEqual(saved['candidate_mask'].shape, (64, 64))
                self.assertEqual(saved['candidate_mask'].sum(), 150)
                np.testing.assert_allclose(saved['candidate_rectangles_xyxy'],
                                           candidate_rectangles(second['3425.jpg']['indices'], (64, 64), (32, 48)))
                self.assertAlmostEqual(float(saved['predicted_density'].sum()), 1., places=5)

    def test_old_checkpoint_defaults_and_resume_loss_cannot_change(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'best.pt'
            defaults = json.loads(Path('configs/fsc147_text.json').read_text())
            del defaults['count_loss_mode'], defaults['density_supervision_size']
            save_checkpoint(path, small_counter(), None, 2, 1., defaults)
            config, _ = parse_config('evaluate', ['--checkpoint', str(path)])
            self.assertEqual(config['count_loss_mode'], 'relative')
            self.assertEqual(config['density_supervision_size'], 32)
            with self.assertRaisesRegex(ValueError, 'count_loss_mode'):
                parse_config('train', ['--resume', str(path), '--count-loss-mode', 'absolute'])

    def test_probe_has_gradients_without_mutating_parameters_or_bn(self):
        class Dataset:
            ids = ['a.jpg', 'b.jpg']
            def __len__(self):
                return 2
            def __getitem__(self, index):
                return {'image_id': self.ids[index], 'image': torch.ones(3, 56, 56),
                        'text': 'red', 'density': torch.full((1, 32, 32), 5 / 1024), 'count': torch.tensor(5.)}
        class ProbeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.zeros(1, 1024))
                self.bn = torch.nn.BatchNorm1d(1024)
            def forward(self, images, texts, epoch, return_diagnostics):
                logits = self.bn(self.logits.expand(len(images), -1))
                density = torch.nn.functional.softplus(logits).reshape(-1, 1, 32, 32)
                return {'density': density, 'count': density.sum((1, 2, 3)),
                        'balance_loss': density.new_zeros(()), 'diagnostics': {'mixed_logits': logits}}
        model = ProbeModel()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        rows = gradient_probe(model, Dataset(), Dataset.ids,
                              {'device': 'cpu', 'workers': 0, 'batch_size': 2, 'count_loss_mode': 'relative'}, 0)
        self.assertEqual(len(rows), 2)
        self.assertGreater(rows[0]['density_gradient_l2'], 0)
        self.assertGreater(rows[0]['count_gradient_l2'], 0)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_d0_gate_stops_before_interventions(self):
        import src.counting.ablation_run as run
        with tempfile.TemporaryDirectory() as root:
            dataset = SyntheticDiagnosticDataset(root)
            config = dict(image_size=448, batch_size=2, workers=0, device='cpu', seed=42,
                          data_root=root, warmup_epochs=30, epochs=100, limit_train=None, limit_val=None)
            reference = [{'image_id': name, 'text': 'the dots', 'target': 2., 'prediction': 99.,
                          'absolute_error': 97.} for name in dataset.ids]
            calls = []
            class Model(SyntheticDiagnosticModel):
                def forward(self, *args, **kwargs):
                    calls.append(kwargs.pop('intervention'))
                    return super().forward(*args, **kwargs)
            with patch.object(run, 'read_checkpoint', return_value={'config': config, 'epoch': 33}), \
                 patch.object(run, 'configure_logging', side_effect=lambda path: Path(path).mkdir(parents=True)), \
                 patch.object(run, 'validate_checkpoint'), patch.object(run, 'validate_reference'), \
                 patch.object(run, 'parse_config', return_value=(config, {})), \
                 patch.object(run, 'prepared_provenance', return_value={}), \
                 patch.object(run, 'sha256', return_value='fixture'), \
                 patch.object(run, 'dataset_for', return_value=dataset), \
                 patch.object(run, 'load_classes', return_value={n: 'dots' for n in dataset.ids}), \
                 patch.object(run, 'read_predictions', return_value=reference), \
                 patch.object(run, 'build_model', return_value=Model()), \
                 patch.object(run, 'restore_checkpoint'):
                result = main(['intervene', '--config', 'fixture', '--reference-csv', 'fixture',
                               '--device', 'cpu', '--output-dir', str(Path(root) / 'gate')])
            self.assertEqual(result, 2)
            self.assertEqual(set(calls), {'D0'})

    def test_all_training_smokes_save_checkpoints_and_probes(self):
        import src.counting.ablation_run as run
        class TrainModel(torch.nn.Module):
            def __init__(self, grid):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.tensor(-3.))
                self.grid = grid
            def forward(self, images, texts, epoch=0, return_diagnostics=False):
                b, g = len(images), self.grid
                logits = self.bias.expand(b, g * g)
                density = torch.nn.functional.softplus(logits).reshape(b, 1, g, g)
                result = {'density': density, 'count': density.sum((1, 2, 3)),
                          'balance_loss': self.bias * 0}
                if return_diagnostics:
                    result['diagnostics'] = {'mixed_logits': logits,
                        'stage1_weights': torch.tensor([[0., .5, .5]]).expand(b, -1),
                        'stage2_weights': torch.full((b, 3), 1 / 3)}
                return result
        with tempfile.TemporaryDirectory() as root:
            dataset = SyntheticDiagnosticDataset(root)
            base = dict(data_root=root, device='cpu', workers=1)
            for group in ('T0', 'T1', 'T2'):
                directory = Path(root) / group
                directory.mkdir()
                args = SimpleNamespace(config='fixture', device='cpu', reference_csv='fixture',
                                       group=group, resume=None, smoke=True, command='train')
                with patch.object(run, 'parse_config', return_value=(base, {})), \
                     patch.object(run, 'prepared_provenance', return_value={}), \
                     patch.object(run, 'sha256', return_value='fixture'), \
                     patch.object(run, 'read_predictions', return_value=[]), \
                     patch.object(run, 'validate_reference'), \
                     patch.object(run, 'load_classes', return_value={}), \
                     patch.object(run, 'dataset_for', return_value=dataset), \
                     patch.object(run, 'loader_for', side_effect=lambda data, config, *args: torch.utils.data.DataLoader(data, batch_size=2)), \
                     patch.object(run, 'build_model', side_effect=lambda config: TrainModel(config['image_size'] // 14)):
                    self.assertEqual(run.run_training(args, directory), 0)
                checkpoint = torch.load(directory / 'latest.pt', weights_only=False)
                self.assertEqual(checkpoint['experiment_group'], group)
                self.assertTrue(checkpoint['experiment_smoke'])
                self.assertNotEqual(float(checkpoint['model']['bias']), -3.)
                self.assertTrue((directory / 'probe_epoch_000.csv').is_file())
                self.assertEqual(json.loads((directory / 'summary.json').read_text())['status'], 'training_smoke_passed')


if __name__ == '__main__':
    unittest.main()
