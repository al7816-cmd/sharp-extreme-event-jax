# Usage:
#   parallel -j 10 python run_instanton.py ::: 1 2 3 4 5 6 7 8 9 10

import os
import sys
import subprocess

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
os.environ["JAX_PLATFORMS"] = "cpu"

targetObs = sys.argv[1]
subprocess.run([sys.executable, "dn-shell-model-inst.py", targetObs], check=True)