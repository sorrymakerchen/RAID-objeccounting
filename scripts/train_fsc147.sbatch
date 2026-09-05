#!/usr/bin/env bash
#SBATCH --job-name=raid-fsc147
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --export=ALL

# Submit from the RAID project root after activating the prepared conda environment.
# Pass cluster-specific --partition/--account/--qos options to sbatch, before this file.
set -Eeuo pipefail
trap 'status=$?; printf "%s [DEBUG] train_fsc147.sbatch failed: line=%s status=%s\n" "$(date -Is)" "$LINENO" "$status" >&2; exit "$status"' ERR

project_dir="${RAID_PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit this script with sbatch}}"
cd -- "$project_dir"
config="${1:-configs/server.json}"
if (( $# > 0 )); then shift; fi

# sbatch does not start an interactive shell; use the submitted conda environment directly.
if [[ -n "${RAID_PYTHON:-}" ]]; then
    python_bin="$RAID_PYTHON"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    python_bin="$CONDA_PREFIX/bin/python"
else
    printf '%s [DEBUG] Activate your conda environment before sbatch, or export RAID_PYTHON=/absolute/path/to/python.\n' "$(date -Is)" >&2
    exit 2
fi

[[ -x "$python_bin" ]] || { printf 'Python executable missing: %s\n' "$python_bin" >&2; exit 2; }
[[ -f run_train_fsc147.py ]] || { printf 'RAID project root is incorrect: %s\n' "$PWD" >&2; exit 2; }
[[ -f "$config" ]] || { printf 'Server configuration missing: %s\n' "$config" >&2; exit 2; }

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
# cuda:0 means the first GPU assigned by Slurm; preserve CUDA_VISIBLE_DEVICES.
train_args=(--config "$config" --device cuda:0 "$@")
printf '%s [DEBUG] job=%s host=%s project=%s python=%s config=%s\n' \
    "$(date -Is)" "$SLURM_JOB_ID" "$(hostname)" "$PWD" "$python_bin" "$config"

"$python_bin" - "${train_args[@]}" <<'PY'
import logging
import sys
from pathlib import Path

logging.basicConfig(format='%(asctime)s [DEBUG] preflight %(message)s', level=logging.INFO)
try:
    import clip
    import torch
    from src.counting.cli import parse_config

    config, _ = parse_config('train', sys.argv[1:])
    if config['limit_train'] is not None or config['limit_val'] is not None:
        raise ValueError('Formal training requires limit_train=null and limit_val=null')
    for key in ('text_annotations', 'projection_weights', 'dino_weights', 'clip_weights'):
        if not config[key] or not Path(config[key]).is_file():
            raise FileNotFoundError(f'{key} must point to a prepared local file: {config[key]}')
    if not config['dino_repo'] or not Path(config['dino_repo'], 'hubconf.py').is_file():
        raise FileNotFoundError('dino_repo must point to the prepared DINOv2 repository')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable inside the Slurm allocation')
    logging.info('CUDA=%s epochs=%s batch_size=%s accumulation=%s output_dir=%s',
                 torch.cuda.get_device_name(0), config['epochs'], config['batch_size'],
                 config['accumulation'], config['output_dir'])
except Exception:
    logging.exception('Preflight failed')
    raise
PY

exec srun --ntasks=1 "$python_bin" -u run_train_fsc147.py "${train_args[@]}"
