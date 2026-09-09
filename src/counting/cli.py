"""Shared command wiring; model and dataset remain independently testable."""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset.fsc147 import FSC147Dataset, read_json
from src.counting.encoders import FrozenTextImageEncoder
from src.counting.inference import CountingPredictor, save_prediction
from src.counting.model import RAIDCounter
from src.counting.training import (evaluate, read_checkpoint, restore_checkpoint, save_checkpoint,
                                   seed_worker, set_seed, train_epoch)

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / 'configs' / 'fsc147_text.json'
PATH_KEYS = {'data_root', 'text_annotations', 'projection_weights', 'dino_repo',
             'dino_weights', 'clip_weights', 'download_root', 'output_dir', 'device'}
TRAIN_KEYS = {'epochs', 'batch_size', 'accumulation', 'lr', 'weight_decay', 'workers',
              'limit_train', 'limit_val', 'augment'}


def configure_logging(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [DEBUG] %(name)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(Path(directory) / 'run.log', encoding='utf-8')],
                        force=True)


def parse_config(mode, argv=None):
    parser = argparse.ArgumentParser(description=f'RAID FSC147 text counting: {mode}')
    parser.add_argument('--config', help='JSON overrides; relative paths resolve from current directory')
    for key in sorted(PATH_KEYS):
        parser.add_argument('--' + key.replace('_', '-'))
    for key in ('epochs', 'batch_size', 'accumulation', 'seed', 'workers', 'image_size',
                'k', 'warmup_epochs', 'limit_train', 'limit_val'):
        parser.add_argument('--' + key.replace('_', '-'), type=int)
    for key in ('lr', 'weight_decay'):
        parser.add_argument('--' + key.replace('_', '-'), type=float)
    parser.add_argument('--no-augment', dest='augment', action='store_false', default=None)
    parser.add_argument('--count-loss-mode', choices=('relative', 'absolute'))
    parser.add_argument('--density-supervision-size', type=int)
    if mode == 'train':
        parser.add_argument('--resume')
    else:
        parser.add_argument('--checkpoint', required=True)
    if mode == 'evaluate':
        parser.add_argument('--split', choices=('val', 'test'), default='test')
        parser.add_argument('--limit', type=int)
        parser.add_argument('--save-examples', type=int, default=3)
    if mode == 'predict':
        parser.add_argument('--image', required=True)
        parser.add_argument('--text', required=True)
    args = vars(parser.parse_args(argv))
    defaults = read_json(DEFAULT_CONFIG)
    defaults.update(head_type='raid', local_count_weight=0., routing_seed=None)
    overrides = read_json(args['config']) if args['config'] else {}
    unknown = set(overrides) - set(defaults)
    if unknown:
        raise ValueError(f'Unknown config settings: {sorted(unknown)}')
    overrides.update({k: v for k, v in args.items() if k in defaults and v is not None})
    checkpoint = args.get('resume') or args.get('checkpoint')
    config = dict(defaults)
    if checkpoint:
        config.update(read_checkpoint(checkpoint)['config'])
        for key, value in overrides.items():
            if key not in PATH_KEYS | TRAIN_KEYS and value != config[key]:
                raise ValueError(f'Checkpoint setting {key}={config[key]} cannot change to {value}')
        if 'output_dir' not in overrides and mode != 'train':
            config['output_dir'] = str(Path(config['output_dir']) / mode)
    config.update(overrides)
    for key in ('epochs', 'batch_size', 'accumulation', 'image_size', 'k'):
        if config[key] < 1:
            raise ValueError(f'{key} must be positive')
    if config['workers'] < 0 or config['lr'] <= 0 or config['weight_decay'] < 0:
        raise ValueError('workers and weight_decay must be nonnegative; lr must be positive')
    if config['image_size'] % 14 or config['k'] > (config['image_size'] // 14) ** 2:
        raise ValueError('image_size must be divisible by 14 and provide at least k patches')
    if config['count_loss_mode'] not in ('relative', 'absolute'):
        raise ValueError('count_loss_mode must be relative or absolute')
    if config['head_type'] not in ('raid', 'spatial'):
        raise ValueError('head_type must be raid or spatial')
    if config['density_supervision_size'] < 1 or (config['image_size'] // 14) % config['density_supervision_size']:
        raise ValueError('density_supervision_size must divide the native output grid')
    return config, args


def build_model(config):
    device = torch.device(config['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA requested ({device}) but unavailable; select --device cpu')
    encoder = FrozenTextImageEncoder(**{k: config[k] for k in
                                       ('projection_weights', 'dino_repo', 'dino_weights',
                                        'clip_weights', 'download_root')})
    return RAIDCounter(encoder, reduced_dim=config['reduced_dim'], k=config['k'],
                       expert_dim=config['expert_dim'], warmup_epochs=config['warmup_epochs'],
                       head_type=config.get('head_type', 'raid'),
                       routing_seed=config.get('routing_seed')).to(device)


def build_loader(config, split, limit=None, training=False):
    dataset = FSC147Dataset(config['data_root'], config['text_annotations'], split,
                            image_size=config['image_size'],
                            flip_probability=0.5 if training and config['augment'] else 0,
                            limit=limit, density_supervision_size=config.get('density_supervision_size', 32))
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=training,
                      num_workers=config['workers'], worker_init_fn=seed_worker,
                      pin_memory=str(config['device']).startswith('cuda'), drop_last=False)


def write_json(path, value):
    with Path(path).open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def train(config, args):
    directory = Path(config['output_dir'])
    if not args.get('resume') and (directory / 'latest.pt').exists():
        raise FileExistsError(f'{directory} already contains a run; use --resume or a new --output-dir')
    train_loader = build_loader(config, 'train', config['limit_train'], True)
    val_loader = build_loader(config, 'val', config['limit_val'])
    model = build_model(config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config['lr'], weight_decay=config['weight_decay'])
    start, best_mae = 0, float('inf')
    if args.get('resume'):
        state = restore_checkpoint(args['resume'], model, optimizer)
        start, best_mae = state['epoch'] + 1, state['best_mae']
        # A new output directory must establish its own best checkpoint.
        if not (directory / 'best.pt').is_file():
            best_mae = float('inf')
        for group in optimizer.param_groups:
            group['lr'], group['weight_decay'] = config['lr'], config['weight_decay']
    if start >= config['epochs']:
        raise ValueError(f'Checkpoint completed {start} epochs; --epochs must exceed {start}')
    write_json(directory / 'config.json', config)
    for epoch in range(start, config['epochs']):
        training = train_epoch(model, train_loader, optimizer, config['device'], epoch,
                               config['accumulation'], count_loss_mode=config.get('count_loss_mode', 'relative'),
                               density_supervision_size=config.get('density_supervision_size', 32),
                               local_count_weight=config.get('local_count_weight', 0.))
        validation, _ = evaluate(model, val_loader, config['device'], epoch)
        LOGGER.info('epoch=%d train=%s val=%s', epoch, training, validation)
        record = {'epoch': epoch, 'train': training, 'val': validation}
        with (directory / 'history.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, allow_nan=False) + '\n')
        if validation['mae'] < best_mae:
            best_mae = validation['mae']
            save_checkpoint(directory / 'best.pt', model, optimizer, epoch, best_mae, config)
        save_checkpoint(directory / 'latest.pt', model, optimizer, epoch, best_mae, config)


def evaluate_command(config, args):
    model = build_model(config)
    state = restore_checkpoint(args['checkpoint'], model, restore_rng=False)
    loader = build_loader(config, args['split'], args['limit'])
    metrics, rows = evaluate(model, loader, config['device'], state['epoch'])
    directory = Path(config['output_dir'])
    write_json(directory / f'{args["split"]}_metrics.json',
               dict(metrics, split=args['split'], checkpoint=str(args['checkpoint']),
                    limited=args['limit'] is not None))
    with (directory / f'{args["split"]}_predictions.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info('Evaluation split=%s metrics=%s', args['split'], metrics)
    predictor = CountingPredictor(model, config['device'], config['image_size'], state['epoch'])
    for row in rows[:max(0, args['save_examples'])]:
        image = Path(config['data_root']) / 'images_384_VarV2' / row['image_id']
        save_prediction(predictor.predict(image, row['text']), image,
                        directory / 'examples', Path(row['image_id']).stem)


def predict_command(config, args):
    model = build_model(config)
    state = restore_checkpoint(args['checkpoint'], model, restore_rng=False)
    predictor = CountingPredictor(model, config['device'], config['image_size'], state['epoch'])
    result = predictor.predict(args['image'], args['text'])
    save_prediction(result, args['image'], config['output_dir'])
    LOGGER.info('predict image=%s text=%r count=%.3f', args['image'], args['text'], result['count'])


def main(mode, argv=None):
    try:
        config, args = parse_config(mode, argv)
        configure_logging(config['output_dir'])
        set_seed(config['seed'])
        {'train': train, 'evaluate': evaluate_command, 'predict': predict_command}[mode](config, args)
        return 0
    except Exception:
        if not logging.getLogger().handlers:
            logging.basicConfig(format='%(asctime)s [DEBUG] %(name)s %(message)s')
        LOGGER.exception('Counting command failed mode=%s', mode)
        return 1
