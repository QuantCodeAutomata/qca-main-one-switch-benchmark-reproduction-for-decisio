"""Master runner: exp_1 → exp_2 → exp_3 in sequence."""
import logging
import time
import warnings
import sys
import os
import runpy

# Ensure project root is on sys.path regardless of working directory
_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Change to project root so relative paths in sub-scripts resolve correctly
os.chdir(_ROOT)

warnings.filterwarnings("ignore")

_RESULTS_DIR = os.path.join(_ROOT, "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(os.path.join(_RESULTS_DIR, "run_log.txt"), mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── Exp 1 ──────────────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("EXP 1: full benchmark grid")
logger.info("=" * 60)
t0 = time.time()
from src.runner import run_benchmark_grid

df = run_benchmark_grid(save_dir=_RESULTS_DIR)
logger.info("Exp 1 done: %d rows in %.1f min", len(df), (time.time() - t0) / 60)

# ── Exp 2 ──────────────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("EXP 2: year-level stability analysis")
logger.info("=" * 60)
t1 = time.time()
runpy.run_path(os.path.join(_THIS_DIR, "exp_2_yearly_stability.py"), run_name="__main__")
logger.info("Exp 2 done in %.1f min", (time.time() - t1) / 60)

# ── Exp 3 ──────────────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("EXP 3: cost-sensitivity analysis")
logger.info("=" * 60)
t2 = time.time()
runpy.run_path(os.path.join(_THIS_DIR, "exp_3_cost_sensitivity.py"), run_name="__main__")
logger.info("Exp 3 done in %.1f min", (time.time() - t2) / 60)

logger.info("ALL DONE in %.1f min total", (time.time() - t0) / 60)
