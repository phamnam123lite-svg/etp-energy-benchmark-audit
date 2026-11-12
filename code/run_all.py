"""Runs the whole pipeline in order and records library versions. Usage:  python run_all.py"""
import platform, subprocess, sys, time
from pathlib import Path
here = Path(__file__).resolve().parent
steps = ["01_build_dataset.py", "02_audit.py", "03_models.py", "04_uncertainty.py", "05_interpret.py",
         "06_augmentation_leakage.py", "07_out_of_time.py", "08_summary_figures.py"]
t0 = time.time()
for s in steps:
    print(f"\n===== {s} =====", flush=True)
    subprocess.run([sys.executable, str(here / s)], check=True, cwd=here)
import numpy, pandas, scipy, sklearn, matplotlib
v = {"python": platform.python_version(), "platform": platform.platform(), "numpy": numpy.__version__, "pandas": pandas.__version__,
     "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__}
(here.parent / "results" / "environment_versions.txt").write_text("\n".join(f"{k}: {x}" for k, x in v.items()) + f"\nruntime_s: {time.time() - t0:.0f}\n")
print(v, f"\nall steps finished in {time.time() - t0:.0f} s")
