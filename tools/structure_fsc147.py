"""Run validation-only RAID structure experiments."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.counting.structure_run import main

if __name__ == '__main__':
    raise SystemExit(main())
