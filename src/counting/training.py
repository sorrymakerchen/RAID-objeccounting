"""Training, evaluation and epoch-boundary reproducible checkpoints."""

import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def counting_loss(output, target, count):
    if output['density'].shape != target.shape:
        raise ValueError(f'Density shape mismatch: {output["density"].shape} vs {target.shape}')
    density_loss = F.mse_loss(output['density'] * 100, target * 100)
    count_loss = ((output['count'] - count).abs() / (count + 1)).mean()
    loss = density_loss + 0.1 * count_loss + 0.005 * output['balance_loss']
    if not torch.isfinite(loss):
        raise FloatingPointError('Counting loss is non-finite; inspect inputs, density and gradients')
    return {'loss': loss, 'density_loss': density_loss, 'count_loss': count_loss,
            'balance_loss': output['balance_loss']}


def train_epoch(model, loader, optimizer, device, epoch, accumulation=4, log_every=20):
    if accumulation < 1 or not len(loader):
        raise ValueError('Gradient accumulation and training loader length must be positive')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, total_error, samples = 0., 0., 0
    for step, batch in enumerate(loader):
        try:
            output = model(batch['image'].to(device), batch['text'], epoch)
            losses = counting_loss(output, batch['density'].to(device), batch['count'].to(device))
            group_start = (step // accumulation) * accumulation
            group_end = min(group_start + accumulation, len(loader))
            # Account for both the last accumulation group and a smaller final batch.
            group_samples = min(group_end * loader.batch_size, len(loader.dataset)) - group_start * loader.batch_size
            batch_size = len(batch['text'])
            (losses['loss'] * (batch_size / group_samples)).backward()
            if step + 1 == group_end:
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f'Non-finite gradient: {name}')
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += losses['loss'].item() * batch_size
            total_error += (output['count'].detach().cpu() - batch['count']).abs().sum().item()
            samples += batch_size
            if step % log_every == 0 or step + 1 == len(loader):
                LOGGER.info('train_epoch epoch=%d batch=%d/%d loss=%.5f MAE=%.3f',
                            epoch, step + 1, len(loader), total_loss / samples, total_error / samples)
        except Exception:
            LOGGER.exception('train_epoch failed epoch=%s step=%s image_ids=%s',
                             epoch, step, batch.get('image_id'))
            raise
    return {'loss': total_loss / samples, 'mae': total_error / samples}


@torch.no_grad()
def evaluate(model, loader, device, epoch=0):
    model.eval()
    rows, absolute, squared = [], 0., 0.
    for batch in loader:
        output = model(batch['image'].to(device), batch['text'], epoch)
        predictions = output['count'].cpu().double()
        if not torch.isfinite(predictions).all():
            raise FloatingPointError(f'Non-finite evaluation predictions: {batch["image_id"]}')
        for image_id, text, prediction, target in zip(
                batch['image_id'], batch['text'], predictions.tolist(), batch['count'].tolist()):
            error = prediction - target
            absolute += abs(error)
            squared += error ** 2
            rows.append({'image_id': image_id, 'text': text, 'prediction': prediction,
                         'target': target, 'absolute_error': abs(error)})
    if not rows:
        raise ValueError('Cannot evaluate an empty dataset')
    return {'mae': absolute / len(rows), 'rmse': (squared / len(rows)) ** 0.5,
            'samples': len(rows)}, rows


def save_checkpoint(path, model, optimizer, epoch, best_mae, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        'format_version': 1,
        # Frozen pretrained weights are identified in config, not duplicated every epoch.
        'model': {k: v for k, v in model.state_dict().items() if not k.startswith('encoder.')},
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'epoch': epoch, 'best_mae': best_mae, 'config': config,
        'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
    }
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        torch.save(state, temporary)
        temporary.replace(path)
    except Exception:
        LOGGER.exception('save_checkpoint failed path=%s epoch=%s', path, epoch)
        raise


def read_checkpoint(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f'Counting checkpoint missing: {path}')
    # Our own checkpoints include Python/NumPy RNG states; load trusted files only.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('format_version') != 1:
        raise ValueError(f'Unsupported counting checkpoint format: {path}')
    return state


def restore_checkpoint(path, model, optimizer=None, restore_rng=True):
    state = read_checkpoint(path)
    missing, unexpected = model.load_state_dict(state['model'], strict=False)
    missing = [key for key in missing if not key.startswith('encoder.')]
    if missing or unexpected:
        raise ValueError(f'Counting checkpoint incompatible: missing={missing}, unexpected={unexpected}')
    if optimizer is not None:
        if state['optimizer'] is None:
            raise ValueError('Checkpoint has no optimizer state for training resume')
        optimizer.load_state_dict(state['optimizer'])
    if restore_rng:
        random.setstate(state['rng']['python'])
        np.random.set_state(state['rng']['numpy'])
        torch.set_rng_state(state['rng']['torch'])
        if torch.cuda.is_available() and state['rng']['cuda'] is not None:
            if len(state['rng']['cuda']) != torch.cuda.device_count():
                LOGGER.warning('GPU count changed; CUDA RNG cannot be restored exactly')
            else:
                torch.cuda.set_rng_state_all(state['rng']['cuda'])
    return state
