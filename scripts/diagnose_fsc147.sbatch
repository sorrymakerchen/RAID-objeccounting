#!/usr/bin/env bash
#SBATCH --job-name=raid-diagnose
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
trap 'status=$?; printf "%s [DEBUG] diagnose_fsc147.sbatch failed: line=%s status=%s\n" "$(date -Is)" "$LINENO" "$status" >&2; exit "$status"' ERR

cd -- "${RAID_PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the RAID root with sbatch}}"
conda_root=/public/home/yzhang712/anaconda3
conda_env=raid
conda_init="$conda_root/etc/profile.d/conda.sh"
[[ -f "$conda_init" ]] || { printf 'Conda initialization script missing: %s\n' "$conda_init" >&2; exit 2; }
# Batch shells do not reliably inherit the environment shown in the submission prompt.
source "$conda_init"
conda activate "$conda_env"
python_bin="$(command -v python)"
[[ -x "$python_bin" ]] || { printf 'Python missing: %s\n' "$python_bin" >&2; exit 2; }
[[ "${CONDA_DEFAULT_ENV:-}" == "$conda_env" ]] || {
    printf 'Expected conda environment %s, got %s\n' "$conda_env" "${CONDA_DEFAULT_ENV:-<none>}" >&2
    exit 2
}
[[ -f tools/diagnose_fsc147.py ]] || { printf 'Submit from the RAID project root.\n' >&2; exit 2; }
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
printf '%s [DEBUG] job=%s host=%s conda_env=%s python=%s\n' \
    "$(date -Is)" "$SLURM_JOB_ID" "$(hostname)" "$CONDA_DEFAULT_ENV" "$python_bin"
# Keep Slurm's CUDA_VISIBLE_DEVICES. All paths and --smoke are supplied by the caller.
exec srun --ntasks=1 "$python_bin" -u tools/diagnose_fsc147.py "$@"
