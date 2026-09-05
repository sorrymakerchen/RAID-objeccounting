"""Run real-weight acceptance checks on a small FSC147 subset.

Run after the documented 8-image overfit experiment. This is not a benchmark run.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.counting.cli import build_loader, build_model, configure_logging, write_json
from src.counting.inference import CountingPredictor, save_prediction
from src.counting.training import counting_loss, evaluate, read_checkpoint, restore_checkpoint, set_seed


@torch.no_grad()
def inspect_fit(model, loader, device, epoch):
    model.eval()
    losses, errors, size = 0., 0., 0
    for batch in loader:
        output = model(batch['image'].to(device), batch['text'], epoch)
        loss = counting_loss(output, batch['density'].to(device), batch['count'].to(device))
        losses += loss['loss'].item() * len(batch['text'])
        errors += (output['count'].cpu() - batch['count']).abs().sum().item()
        size += len(batch['text'])
    return {'loss': losses / size, 'mae': errors / size, 'samples': size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', default='outputs/acceptance')
    parser.add_argument('--device')
    parser.add_argument('--limit', type=int, default=8)
    args = parser.parse_args()
    configure_logging(args.output_dir)
    state = read_checkpoint(args.checkpoint)
    config = dict(state['config'])
    if args.device:
        config['device'] = args.device
    config['workers'] = 0
    config['batch_size'] = 1
    set_seed(config['seed'])
    model = build_model(config)
    loader = build_loader(config, 'train', args.limit)
    initial = inspect_fit(model, loader, config['device'], state['epoch'])
    restore_checkpoint(args.checkpoint, model, restore_rng=False)
    fitted = inspect_fit(model, loader, config['device'], state['epoch'])
    assert fitted['loss'] < initial['loss'], (initial, fitted)
    assert fitted['mae'] < initial['mae'], (initial, fitted)
    report = {'protocol': 'Small-subset acceptance only; not full FSC147 benchmark',
              'checkpoint': args.checkpoint, 'epoch': state['epoch'],
              'initial_train': initial, 'fitted_train': fitted,
              'torch': torch.__version__, 'device': config['device']}
    if torch.device(config['device']).type == 'cuda':
        report['gpu'] = torch.cuda.get_device_name(torch.device(config['device']))
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    for split in ('val', 'test'):
        metrics, _ = evaluate(model, build_loader(config, split, args.limit),
                              config['device'], state['epoch'])
        report[split + '_subset'] = metrics
    sample = loader.dataset[0]
    images = sample['image'].unsqueeze(0).to(config['device'])
    with torch.no_grad():
        first = model(images, [sample['text']], state['epoch'])
        second = model(images, [sample['text']], state['epoch'])
        alternate = model(images, ['the elephants'], state['epoch'])
    torch.testing.assert_close(first['density'], second['density'], rtol=0, atol=0)
    assert not torch.equal(first['indices'], alternate['indices'])
    assert not torch.equal(first['similarity'], alternate['similarity'])
    assert all(not p.requires_grad and p.grad is None for p in model.encoder.parameters())
    report['deterministic'] = True
    report['prompt_check'] = {
        'image_id': sample['image_id'], 'text': sample['text'],
        'alternate_text': 'the elephants', 'count': first['count'].item(),
        'alternate_count': alternate['count'].item(),
        'similarity_mean_absolute_change': (first['similarity'] - alternate['similarity']).abs().mean().item()}
    image_path = loader.dataset.paths(sample['image_id'])[0]
    predictor = CountingPredictor(model, config['device'], config['image_size'], state['epoch'])
    for text, stem in [(sample['text'], 'target'), ('the elephants', 'alternate')]:
        prediction = predictor.predict(image_path, text)
        assert abs(float(prediction['density'].sum()) - prediction['count']) < 1e-3
        save_prediction(prediction, image_path, args.output_dir, stem)
    write_json(Path(args.output_dir) / 'acceptance.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
