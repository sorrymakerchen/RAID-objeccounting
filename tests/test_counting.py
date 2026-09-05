import json
import random
import builtins
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from dataset.fsc147 import FSC147Dataset, resize_density
from src.counting.model import RAIDCounter, TextGuidedRetrieval, CountingMoE
from src.counting.training import (counting_loss, evaluate, save_checkpoint, restore_checkpoint,
                                   train_epoch, set_seed)
from src.counting.encoders import CLIPTextEncoder, Talk2DINOProjection, FrozenTextImageEncoder
from src.counting.inference import CountingPredictor


class FakeEncoder(nn.Module):
    """Deterministic dependency substitute; never used by a training CLI."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()), requires_grad=False)

    def forward(self, images, texts):
        feature = torch.nn.functional.adaptive_avg_pool2d(images, (4, 4))
        feature = torch.cat([feature, feature[:, :1]], dim=1) * self.scale
        text = feature.new_tensor([[1, 0, 0, 0] if t == 'red' else [0, 1, 0, 0]
                                   for t in texts])
        return feature, text


def small_counter():
    return RAIDCounter(FakeEncoder(), feature_dim=4, reduced_dim=4, k=10,
                       expert_dim=4, warmup_epochs=1)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'images_384_VarV2').mkdir()
        (self.root / 'gt_density_map_adaptive_384_VarV2').mkdir()
        self.annotations, self.texts = {}, {}
        for i, split in enumerate(['train', 'val', 'test']):
            name = f'{i}.jpg'
            image = np.zeros((12, 20, 3), dtype=np.uint8)
            image[:, :8, 0] = 255
            Image.fromarray(image).save(self.root / 'images_384_VarV2' / name)
            density = np.zeros((12, 20), dtype=np.float32)
            density[3, 2] = 2
            np.save(self.root / 'gt_density_map_adaptive_384_VarV2' / f'{i}.npy', density)
            self.annotations[name] = {'points': [[2, 3], [3, 3]]}
            self.texts[name] = {'text_description': 'red', 'data_split': split}
        for name, value in [('annotation_FSC147_384.json', self.annotations),
                            ('Train_Test_Val_FSC_147.json',
                             {'train': ['0.jpg'], 'val': ['1.jpg'], 'test': ['2.jpg']}),
                            ('FSC-147-D.json', self.texts)]:
            (self.root / name).write_text(json.dumps(value))

    def tearDown(self):
        self.tmp.cleanup()

    def test_density_mass_survives_downsampling_and_display_resize(self):
        density = torch.zeros(1, 12, 20)
        density[0, 3, 2] = 2
        for size in [(4, 4), (31, 71)]:
            result = resize_density(density, size)
            self.assertAlmostEqual(result.sum().item(), 2, places=5)
        self.assertEqual(resize_density(torch.zeros_like(density), (4, 4)).sum(), 0)

    def test_dataset_and_flip_are_synchronized_without_boxes(self):
        plain = FSC147Dataset(self.root, self.root / 'FSC-147-D.json', 'train',
                              image_size=56, flip_probability=0)
        flipped = FSC147Dataset(self.root, self.root / 'FSC-147-D.json', 'train',
                                image_size=56, flip_probability=1)
        a, b = plain[0], flipped[0]
        self.assertEqual(a['text'], 'red')
        self.assertEqual(a['original_size'].tolist(), [12, 20])
        self.assertAlmostEqual(a['density'].sum().item(), 2, places=5)
        torch.testing.assert_close(b['image'], a['image'].flip(-1))
        torch.testing.assert_close(b['density'], a['density'].flip(-1))
        self.assertNotIn('boxes', a)

    def test_split_overlap_and_missing_text_fail(self):
        path = self.root / 'Train_Test_Val_FSC_147.json'
        path.write_text(json.dumps({'train': ['0.jpg'], 'val': ['0.jpg'], 'test': []}))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            FSC147Dataset(self.root, self.root / 'FSC-147-D.json')

    def test_missing_text_and_density_fail_with_image_context(self):
        self.texts['0.jpg']['text_description'] = ''
        path = self.root / 'FSC-147-D.json'
        path.write_text(json.dumps(self.texts))
        with self.assertRaisesRegex(ValueError, 'text.*0.jpg'):
            FSC147Dataset(self.root, path)
        self.texts['0.jpg']['text_description'] = 'red'
        path.write_text(json.dumps(self.texts))
        (self.root / 'gt_density_map_adaptive_384_VarV2' / '0.npy').unlink()
        with self.assertRaisesRegex(FileNotFoundError, '0.npy'):
            FSC147Dataset(self.root, path)

    def test_invalid_density_rejected_and_val_augmentation_disallowed(self):
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            resize_density(torch.full((1, 2, 2), -1.), (4, 4))
        with self.assertRaisesRegex(ValueError, 'deterministic'):
            FSC147Dataset(self.root, self.root / 'FSC-147-D.json', 'val', flip_probability=1)


class EncoderDependencyTests(unittest.TestCase):
    def test_missing_clip_reports_install_command_and_logs_context(self):
        with patch.dict('sys.modules', {'clip': None}):
            with self.assertLogs('src.counting.encoders', level='ERROR') as logs:
                with self.assertRaisesRegex(ImportError, 'python -m pip install') as raised:
                    FrozenTextImageEncoder('projection.pth')
        self.assertIn('github.com/openai/CLIP.git', str(raised.exception))
        self.assertIsInstance(raised.exception.__cause__, ModuleNotFoundError)
        self.assertIn('projection.pth', logs.output[0])

    def test_missing_clip_transitive_dependency_is_preserved(self):
        original_import = builtins.__import__
        missing = ModuleNotFoundError("No module named 'ftfy'", name='ftfy')

        def import_without_ftfy(name, *args, **kwargs):
            if name == 'clip':
                raise missing
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=import_without_ftfy):
            with self.assertLogs('src.counting.encoders', level='ERROR'):
                with self.assertRaises(ModuleNotFoundError) as raised:
                    FrozenTextImageEncoder('projection.pth')
        self.assertIs(raised.exception, missing)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_prompt_changes_similarity_and_retrieval(self):
        feature = torch.eye(4).T.reshape(1, 4, 2, 2)
        retrieval = TextGuidedRetrieval(k=2)
        a = retrieval(feature, torch.tensor([[1., 0, 0, 0]]))
        b = retrieval(feature, torch.tensor([[0., 1, 0, 0]]))
        self.assertFalse(torch.equal(a['indices'], b['indices']))
        self.assertFalse(torch.equal(a['similarity'], b['similarity']))
        self.assertEqual(a['matching'].shape, (1, 2, 2, 2))

    def test_density_gradients_and_deterministic_evaluation(self):
        model = small_counter()
        images = torch.rand(2, 3, 56, 56)
        output = model(images, ['red', 'green'], epoch=2)
        self.assertEqual(output['density'].shape, (2, 1, 4, 4))
        self.assertTrue((output['density'] >= 0).all())
        torch.testing.assert_close(output['count'], output['density'].sum((1, 2, 3)))
        loss = counting_loss(output, torch.ones_like(output['density']), torch.tensor([16., 16.]))
        loss['loss'].backward()
        self.assertIsNone(model.encoder.scale.grad)
        self.assertGreater(model.projection.weight.grad.abs().sum().item(), 0)
        model.eval()
        torch.testing.assert_close(model(images, ['red', 'green'], epoch=2)['density'],
                                   model(images, ['red', 'green'], epoch=2)['density'],
                                   rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'text'):
            model(images, ['', 'red'])

    def test_sparse_weights_have_exactly_two_contributors(self):
        probs = torch.tensor([[0.6, 0.3, 0.1]], requires_grad=True)
        weights = CountingMoE.top2_weights(probs)
        torch.testing.assert_close(weights, torch.tensor([[2/3, 1/3, 0.]]))
        self.assertAlmostEqual(weights.sum().item(), 1)

    def test_warmup_only_disables_second_router_gradient(self):
        for epoch, should_route in [(0, False), (1, True)]:
            model = small_counter()
            output = model(torch.rand(2, 3, 56, 56), ['red', 'green'], epoch)
            output['density'].sum().backward()
            self.assertEqual(model.moe.router2.conv.weight.grad is not None, should_route)

    def test_predict_restores_mass_and_original_size(self):
        predictor = CountingPredictor(small_counter(), image_size=56, epoch=2)
        result = predictor.predict(Image.new('RGB', (73, 31), 'red'), 'red')
        self.assertEqual(result['density'].shape, (31, 73))
        self.assertEqual(result['similarity'].shape, (31, 73))
        self.assertAlmostEqual(result['count'], float(result['density'].sum()), places=5)

    def test_projection_checkpoint_and_legacy_key_compatibility(self):
        source = Talk2DINOProjection()
        legacy = {k.replace('hidden_layers.0', 'linear_layer2'): v
                  for k, v in source.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'projection.pt'
            torch.save(legacy, path)
            restored = Talk2DINOProjection()
            restored.load_weights(path)
            text = torch.rand(2, 512)
            torch.testing.assert_close(source(text), restored(text))
            torch.save({'linear_layer.weight': torch.zeros(2, 2)}, path)
            with self.assertRaisesRegex(ValueError, 'Incompatible Talk2DINO'):
                restored.load_weights(path)

    def test_clip_text_encoder_does_not_depend_on_visual_dtype(self):
        clip_model = nn.Module()
        clip_model.transformer = nn.Identity()
        clip_model.token_embedding = nn.Embedding(8, 4)
        clip_model.positional_embedding = nn.Parameter(torch.rand(3, 4))
        clip_model.ln_final = nn.LayerNorm(4)
        clip_model.text_projection = nn.Parameter(torch.rand(4, 2))
        encoder = CLIPTextEncoder(clip_model)
        tokens = torch.tensor([[1, 7, 0], [1, 2, 7]])
        expected = clip_model.ln_final(clip_model.token_embedding(tokens)
                                       + clip_model.positional_embedding)
        expected = expected[[0, 1], [1, 2]] @ clip_model.text_projection
        torch.testing.assert_close(encoder(tokens), expected)

    def test_checkpoint_restores_prediction_optimizer_and_rng(self):
        model = small_counter().eval()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        images = torch.rand(1, 3, 56, 56)
        output = model(images, ['red'], epoch=2)
        output['density'].sum().backward()
        optimizer.step()
        expected = model(images, ['red'], epoch=2)['density'].detach()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            save_checkpoint(path, model, optimizer, epoch=2, best_mae=3.0, config={})
            random_expected, torch_expected = random.random(), torch.rand(2)
            restored = small_counter().eval()
            # Same trainable parameter grouping as the training command.
            restored_optimizer = torch.optim.AdamW([p for p in restored.parameters()
                                                   if p.requires_grad])
            state = restore_checkpoint(path, restored, restored_optimizer)
            self.assertEqual(state['epoch'], 2)
            self.assertEqual(random.random(), random_expected)
            torch.testing.assert_close(torch.rand(2), torch_expected)
            torch.testing.assert_close(restored(images, ['red'], epoch=2)['density'], expected)
            self.assertTrue(restored_optimizer.state)

    def test_nonfinite_loss_fails_explicitly(self):
        with self.assertRaisesRegex(FloatingPointError, 'non-finite'):
            counting_loss({'density': torch.full((1, 1, 2, 2), float('nan')),
                           'count': torch.ones(1), 'balance_loss': torch.tensor(0.)},
                          torch.zeros(1, 1, 2, 2), torch.zeros(1))

    def test_partial_accumulation_group_matches_full_batch_update(self):
        class ScalarCounter(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(0.1))

            def forward(self, images, texts, epoch=0):
                density = images * self.scale
                return {'density': density, 'count': density.sum((1, 2, 3)),
                        'balance_loss': density.new_tensor(0.)}

        samples = [{'image': torch.ones(1, 2, 2) * (i + 1),
                    'text': 'red', 'density': torch.ones(1, 2, 2),
                    'count': torch.tensor(4.), 'image_id': str(i)} for i in range(5)]
        full, accumulated = ScalarCounter(), ScalarCounter()
        for model, size, accumulation in [(full, 5, 1), (accumulated, 2, 4)]:
            train_epoch(model, DataLoader(samples, batch_size=size),
                        torch.optim.SGD(model.parameters(), lr=1e-6), 'cpu', 0, accumulation)
        torch.testing.assert_close(full.scale, accumulated.scale)

    def test_resumed_next_epoch_matches_uninterrupted_training(self):
        set_seed(42)
        samples = [{'image': torch.rand(3, 56, 56), 'text': 'red',
                    'density': torch.ones(1, 4, 4), 'count': torch.tensor(16.),
                    'image_id': str(i)} for i in range(4)]
        loader = DataLoader(samples, batch_size=2, shuffle=True)
        model = small_counter()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        train_epoch(model, loader, optimizer, 'cpu', 0, accumulation=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'resume.pt'
            save_checkpoint(path, model, optimizer, 0, 1., {})
            train_epoch(model, loader, optimizer, 'cpu', 1, accumulation=2)
            expected = {k: v.clone() for k, v in model.state_dict().items()}
            resumed = small_counter()
            resumed_optimizer = torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad])
            restore_checkpoint(path, resumed, resumed_optimizer)
            train_epoch(resumed, loader, resumed_optimizer, 'cpu', 1, accumulation=2)
            for key, actual in resumed.state_dict().items():
                torch.testing.assert_close(actual, expected[key], rtol=0, atol=0)

    def test_metrics_use_per_image_errors(self):
        class FixedCounter(nn.Module):
            def forward(self, images, texts, epoch=0):
                return {'count': images.flatten()}

        samples = [{'image': torch.tensor(float(i)), 'text': 'red',
                    'count': torch.tensor(2.), 'image_id': str(i)} for i in [1, 5]]
        metrics, rows = evaluate(FixedCounter(), DataLoader(samples, batch_size=2), 'cpu')
        self.assertEqual(metrics['mae'], 2)
        self.assertAlmostEqual(metrics['rmse'], 5 ** 0.5)
        self.assertEqual(len(rows), 2)


if __name__ == '__main__':
    unittest.main()
