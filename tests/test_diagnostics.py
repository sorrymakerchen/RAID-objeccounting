import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from test_counting import small_counter
from src.counting.diagnostics import (
    candidate_mask, candidate_rectangles, point_neighborhood, spatial_metrics,
    select_samples, shared_ranges, compare_reference, render_sample,
)
from src.counting.diagnostic_run import run_diagnosis, validate_checkpoint


class SyntheticDiagnosticDataset:
    """A three-image fixture for file/report tests, not a substitute for formal validation."""
    ids = ['3425.jpg', '3427.jpg', '935.jpg']

    def __init__(self, root):
        self.root = Path(root)
        self.annotations = {name: {'points': [[3, 3], [30, 20]]} for name in self.ids}
        for name in self.ids:
            Image.new('RGB', (48, 32), '#406080').save(self.root / name)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return {'image_id': self.ids[index], 'image': torch.full((3, 32, 32), float(index + 1)),
                'text': 'the dots', 'density': torch.full((1, 32, 32), 2 / 1024),
                'count': torch.tensor(2.), 'original_size': torch.tensor([32, 48])}

    def paths(self, name):
        return self.root / name, None


class SyntheticDiagnosticModel(torch.nn.Module):
    def forward(self, images, texts, epoch=0, return_diagnostics=False):
        batch = len(images)
        count = images[:, 0, 0, 0] + images.new_tensor([0 if t == 'the dots' else 1 for t in texts])
        density = count[:, None, None, None].expand(-1, 1, 32, 32) / 1024
        similarity = torch.arange(1024, device=images.device).float().reshape(1, 1, 32, 32).repeat(batch, 1, 1, 1) / 1024
        indices = torch.arange(150, device=images.device).repeat(batch, 1)
        return {'density': density, 'count': count, 'diagnostics': {
            'grid_similarity': similarity, 'candidate_indices': indices,
            'stage1_probabilities': images.new_tensor([[.5, .3, .2]]).repeat(batch, 1),
            'stage1_weights': images.new_tensor([[.625, .375, 0]]).repeat(batch, 1),
            'stage2_probabilities': images.new_tensor([[.2, .3, .5]]).repeat(batch, 1),
            'stage2_weights': images.new_tensor([[.2, .3, .5]]).repeat(batch, 1),
            'stage2_routing_active': True}}


class BatchSensitiveDiagnosticModel(SyntheticDiagnosticModel):
    """Expose accidental batch-shape changes in the prompt comparison workflow."""

    def forward(self, images, texts, epoch=0, return_diagnostics=False):
        result = super().forward(images, texts, epoch, return_diagnostics)
        offset = images.new_tensor(float(len(images) - 1) * 5)
        result['count'] = result['count'] + offset
        result['density'] = result['count'][:, None, None, None].expand(-1, 1, 32, 32) / 1024
        return result


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_optional_output_preserves_values_state_and_actual_mixture(self):
        model = small_counter().eval()
        image = torch.rand(2, 3, 56, 56)
        state_keys = list(model.state_dict())
        for epoch in (0, 2):
            logits = []
            first_outputs, second_inputs = {}, []
            hooks = [expert.register_forward_hook(lambda module, inputs, output: logits.append(output[0]))
                     for expert in model.moe.models2]
            for number, expert in enumerate(model.moe.models1):
                hooks.append(expert.register_forward_hook(
                    lambda module, inputs, output, number=number: first_outputs.update({number: output})))
            hooks.append(model.moe.models2[0].register_forward_pre_hook(
                lambda module, inputs: second_inputs.append(inputs[0])))
            with torch.no_grad():
                diagnostic = model(image, ['red', 'green'], epoch, return_diagnostics=True)
            for hook in hooks:
                hook.remove()
            normal = model(image, ['red', 'green'], epoch)
            torch.testing.assert_close(normal['density'], diagnostic['density'], rtol=0, atol=0)
            torch.testing.assert_close(normal['count'], diagnostic['count'], rtol=0, atol=0)
            detail = diagnostic['diagnostics']
            expected = (torch.stack(logits, 1) * detail['stage2_weights'].unsqueeze(-1)).sum(1)
            expected = torch.nn.functional.softplus(expected).reshape_as(diagnostic['density'])
            torch.testing.assert_close(expected, diagnostic['density'], rtol=0, atol=0)
            self.assertTrue(((detail['stage1_weights'] > 0).sum(1) == 2).all())
            torch.testing.assert_close(detail['stage1_weights'].sum(1), torch.ones(2))
            mixed = torch.zeros_like(second_inputs[0])
            for number, expert_output in first_outputs.items():
                selected = (detail['stage1_weights'][:, number] > 0).nonzero(as_tuple=True)[0]
                mixed[selected] += expert_output * detail['stage1_weights'][selected, number, None, None, None]
            torch.testing.assert_close(mixed, second_inputs[0], rtol=0, atol=0)
            self.assertEqual(detail['stage2_probabilities'] is None, epoch == 0)
        self.assertEqual(state_keys, list(model.state_dict()))
        self.assertNotIn('diagnostics', normal)

    def test_150_unique_cells_and_original_coordinate_rectangles(self):
        mask = candidate_mask(np.arange(150), (32, 32))
        self.assertEqual(mask.sum(), 150)
        rectangles = candidate_rectangles([0, 1023], (32, 32), (320, 640))
        np.testing.assert_allclose(rectangles, [[0, 0, 20, 10], [620, 310, 640, 320]])
        with self.assertRaisesRegex(ValueError, 'unique'):
            candidate_mask([1, 1], (32, 32))

    def test_point_projection_and_spatial_mass(self):
        points = [[0, 0], [639, 319]]
        occupied, neighborhood = point_neighborhood(points, (320, 640), (32, 32))
        self.assertEqual(occupied.sum(), 2)
        self.assertEqual(neighborhood.sum(), 8)
        gt = np.zeros((32, 32)); gt[0, 0] = 1; gt[-1, -1] = 1
        pred = gt * 0.5
        metrics = spatial_metrics([0, 1023], gt, pred, points, (320, 640))
        self.assertEqual(metrics['candidate_gt_mass_fraction'], 1)
        self.assertEqual(metrics['candidate_neighborhood_hit_rate'], 1)
        self.assertEqual(metrics['pred_mass_inside'], 1)
        self.assertEqual(metrics['pred_mass_outside'], 0)
        self.assertFalse(metrics['neighborhood_nearly_full'])

    def test_fsc147_floating_point_boundary_roundoff_is_recorded(self):
        points = [[408.00000000000006, 350.832]]
        occupied, _ = point_neighborhood(points, (384, 408), (32, 32))
        self.assertTrue(occupied[29, 31])
        result = spatial_metrics([959], np.ones((32, 32)), np.ones((32, 32)), points, (384, 408))
        self.assertEqual(result['point_boundary_roundoff_count'], 1)
        with self.assertRaisesRegex(ValueError, 'outside'):
            point_neighborhood([[409, 350]], (384, 408), (32, 32))

    def test_annotation_changes_cannot_change_inference(self):
        model = small_counter().eval()
        image = torch.rand(1, 3, 56, 56)
        a = model(image, ['red'], 2, return_diagnostics=True)
        gt = np.ones((4, 4))
        spatial_metrics(a['indices'][0].numpy(), gt, a['density'][0, 0].detach().numpy(),
                        [[0, 0]], (56, 56))
        spatial_metrics(a['indices'][0].numpy(), gt * 10, a['density'][0, 0].detach().numpy(),
                        [[55, 55]], (56, 56))
        b = model(image, ['red'], 2, return_diagnostics=True)
        torch.testing.assert_close(a['density'], b['density'], rtol=0, atol=0)
        torch.testing.assert_close(a['indices'], b['indices'])

    def test_selection_is_stable_and_ties_use_id(self):
        rows = [{'image_id': name, 'target': target, 'prediction': target - error,
                 'absolute_error': error} for name, target, error in
                [('3425.jpg', 20, 10), ('3427.jpg', 24, 12), ('935.jpg', 700, 300),
                 ('a.jpg', 50, 1), ('b.jpg', 50, 1), ('c.jpg', 50, 2)]]
        first = select_samples(rows)
        self.assertEqual(first, select_samples(list(reversed(rows))))
        by_id = {r['image_id']: r for r in first}
        self.assertIn('control:50-99', by_id['a.jpg']['reasons'])
        self.assertIn('control:50-99', by_id['b.jpg']['reasons'])
        self.assertNotIn('control:50-99', by_id['c.jpg']['reasons'])

    def test_ranges_are_shared_and_raw_units_retained(self):
        variants = [{'similarity': np.array([[0., 0.2]]), 'density': np.array([[1., 2.]])},
                    {'similarity': np.array([[-0.3, 0.1]]), 'density': np.array([[3., 4.]])}]
        ranges = shared_ranges(variants, np.array([[8., 0.]]))
        self.assertEqual(ranges['similarity'], [-0.3, 0.2])
        self.assertEqual(ranges['density'], [0., 8.])

    def test_mismatch_is_detected_not_silently_accepted(self):
        reference = [{'image_id': 'a', 'text': 'red', 'target': 10., 'prediction': 9.}]
        actual = [dict(reference[0], prediction=8.)]
        result, differences = compare_reference(actual, reference)
        self.assertFalse(result['passed'])
        self.assertEqual(differences[0]['prediction_delta'], -1)

    def test_small_checkpoint_cannot_be_used_as_formal_model(self):
        with self.assertRaisesRegex(ValueError, 'Formal checkpoint'):
            validate_checkpoint({'epoch': 33, 'config': {'image_size': 448, 'k': 150}})

    def test_report_exports_and_stops_at_reference_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = SyntheticDiagnosticDataset(directory)
            reference = [{'image_id': name, 'text': 'the dots', 'target': 2.,
                          'prediction': float(index + 1), 'absolute_error': abs(index - 1.)}
                         for index, name in enumerate(dataset.ids)]
            config = {'device': 'cpu', 'batch_size': 2, 'workers': 0}
            classes = {name: 'dots' for name in dataset.ids}
            success_dir = Path(directory) / 'success'
            result = run_diagnosis(SyntheticDiagnosticModel(), dataset, reference, classes,
                                   config, 33, success_dir, smoke=True)
            self.assertEqual(result, 0)
            summary = json.loads((success_dir / 'summary.json').read_text())
            self.assertEqual(summary['status'], 'smoke_passed_not_full_validation')
            sample_dir = success_dir / 'samples' / '3425'
            detail = json.loads((sample_dir / 'diagnostics.json').read_text())
            self.assertEqual(len(detail['variants']), 3)
            self.assertFalse(detail['control_prompt_zero_ground_truth_assumed'])
            with np.load(sample_dir / 'original_arrays.npz') as arrays:
                self.assertEqual(arrays['candidate_mask'].sum(), 150)
                self.assertAlmostEqual(float(arrays['predicted_density'].sum()), 1., places=5)
            self.assertTrue((sample_dir / 'comparison.png').is_file())
            self.assertIn('三图冒烟', (success_dir / 'report.md').read_text(encoding='utf-8'))
            reference[0]['prediction'] = 100.
            failure_dir = Path(directory) / 'failure'
            result = run_diagnosis(SyntheticDiagnosticModel(), dataset, reference, classes,
                                   config, 33, failure_dir, smoke=True)
            self.assertEqual(result, 2)
            self.assertTrue((failure_dir / 'reference_differences.csv').exists())
            self.assertFalse((failure_dir / 'samples').exists())
            self.assertIn('停止归因', (failure_dir / 'report.md').read_text(encoding='utf-8'))

    def test_prompt_comparison_preserves_validation_batch_context(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = SyntheticDiagnosticDataset(directory)
            predictions = [6., 7., 3.]  # Baseline batches contain two, two, then one sample.
            reference = [
                {'image_id': name, 'text': 'the dots', 'target': 2., 'prediction': prediction,
                 'absolute_error': abs(prediction - 2.)}
                for name, prediction in zip(dataset.ids, predictions)
            ]
            config = {'device': 'cpu', 'batch_size': 2, 'workers': 0}
            classes = {name: 'dots' for name in dataset.ids}
            output = Path(directory) / 'batch_context'

            result = run_diagnosis(BatchSensitiveDiagnosticModel(), dataset, reference, classes,
                                   config, 33, output, smoke=True)

            self.assertEqual(result, 0)
            for image_id in dataset.ids:
                detail = json.loads(
                    (output / 'samples' / Path(image_id).stem / 'diagnostics.json').read_text())
                self.assertEqual(detail['repeat_delta_from_validation_batch'], 0)


if __name__ == '__main__':
    unittest.main()
