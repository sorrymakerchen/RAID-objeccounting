#!/usr/bin/env bash
#SBATCH --job-name=raid-structure
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=log/slurm-%x-%j.out
#SBATCH --error=log/slurm-%x-%j.err
#SBATCH --export=ALL

set -Eeuo pipefail
trap 'status=$?; printf "%s [DEBUG] structure_fsc147.sbatch line=%s status=%s\n" "$(date -Is)" "$LINENO" "$status" >&2; exit "$status"' ERR
cd -- "${RAID_PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the RAID root}}"
conda_root="${RAID_CONDA_ROOT:-/public/home/yzhang712/anaconda3}"
source "$conda_root/etc/profile.d/conda.sh"
conda activate "${RAID_CONDA_ENV:-raid}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
printf '%s [DEBUG] job=%s host=%s python=%s\n' "$(date -Is)" "$SLURM_JOB_ID" "$(hostname)" "$(command -v python)"
exec srun --ntasks=1 python -u tools/structure_fsc147.py "$@"
