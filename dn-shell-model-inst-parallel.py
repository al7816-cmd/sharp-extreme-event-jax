"""
Run instanton computations in parallel for multiple target observables.
Run this script from PyCharm — it uses the same Python interpreter and environment.

Usage: just run this file. Edit target_obs_list below to change the values.
"""

import subprocess
import sys
import os
from multiprocessing import Pool

# ---- CONFIGURE HERE ----
target_obs_list = [
    -41, -42, -43, -44, -45, -46, -47, -48, -49, -50
]
n_workers = 10  # number of parallel processes
script_name = "dn-shell-model-inst.py"  # your main instanton script


# -------------------------

def run_single(target_obs):
    """Run the instanton script for a single target observable."""
    env = os.environ.copy()
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"

    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)

    result = subprocess.run(
        [sys.executable, script_path, str(target_obs)],
        env=env,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"[FAILED] targetObs={target_obs}")
        print(result.stderr[-500:])  # last 500 chars of error
    else:
        # print last line of stdout as summary
        lines = result.stdout.strip().split('\n')
        print(f"[DONE] targetObs={target_obs}  |  {lines[-1]}")

    return target_obs, result.returncode


if __name__ == '__main__':
    print(f"Running {len(target_obs_list)} instanton computations with {n_workers} workers")
    print(f"Using Python: {sys.executable}")
    print()

    with Pool(processes=n_workers) as pool:
        results = pool.map(run_single, target_obs_list)

    failed = [obs for obs, rc in results if rc != 0]
    if failed:
        print(f"\nFailed runs: {failed}")
    else:
        print(f"\nAll {len(results)} runs completed successfully.")