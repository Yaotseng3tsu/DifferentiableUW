# DifferentiableUW — Getting Started

A **differentiable underwater-robot dynamics simulator** based on the **Fossen model**,
built with **NVIDIA Warp**. Beyond forward simulation, the physics can be run *backwards*
via automatic differentiation, so we can **identify hydrodynamic parameters** and
**estimate fluid (current) disturbances** from observed trajectories.

> **Status:** research code. The 3-DOF planar model is validated on synthetic data
> (parameter identification + inverse disturbance recovery). See `results/RESULTS.md` for
> quantitative results. The full research plan is in `README.md` (written in Chinese).

---

## 1. Requirements
- An **NVIDIA GPU** with **driver ≥ 525** (CUDA 12.x). You do **not** need to install the
  CUDA Toolkit separately — the `warp-lang` wheel bundles the CUDA runtime.
- **Python 3.10+** (3.11 recommended).

## 2. Install
```bash
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## 3. Windows only — enable UTF-8
Some Warp kernels contain non-ASCII (CJK) comments. On a non-UTF-8 Windows locale, set
Python's UTF-8 mode so kernel compilation and console output don't fail:
```powershell
setx PYTHONUTF8 1        # persistent (new shells); or per-session:  $env:PYTHONUTF8=1
```
Not needed on Linux/macOS (UTF-8 by default).

## 4. Quick check
```bash
python notebooks/00_warp_basics.py     # confirms Warp sees your GPU
python notebooks/01_autodiff_demo.py   # confirms automatic differentiation works
```
Expected: a `cuda:0` device is listed, and the autodiff gradient matches the analytic value.

## 5. Run the demos (suggested order)
| Command | What it does | Output |
|---|---|---|
| `python scripts/run_forward_3dof.py` | 3-DOF forward simulation (thrust + turn) | `pic/forward_3dof.png` |
| `python -m pytest tests/test_gradients.py -v` | gradient gate: analytic vs finite-difference | pass/fail |
| `python scripts/identify_params_3dof.py` | identify 9 hydrodynamic parameters (synthetic) | `pic/identify_params_convergence.png` |
| `python scripts/robustness_3dof.py` | robustness to initial guess + measurement noise | `pic/robustness.png` |
| `python scripts/recover_disturbance_3dof.py` | inverse problem: recover the current force | `pic/recover_disturbance_*.png` |

The `notebooks/p1_*`, `p2_*`, `p3_*` files are a **step-by-step tutorial**: Warp basics →
3-DOF Fossen model → differentiable identification pipeline.

## 6. Repository layout
```
src/         core library
  dynamics/  Fossen equations of motion (3-DOF; 6-DOF stub)
  integrators.py     differentiable integrators (semi-implicit Euler, RK4)
  diff_rollout.py    taped multi-step rollout + trajectory loss
  params.py          identifiable parameter vector
scripts/     runnable experiments (forward sim, identification, inverse problem)
notebooks/   step-by-step tutorial lessons
tests/       gradient verification (correctness gate)
results/     quantitative results (RESULTS.md + data/ CSVs)
pic/         figures
```

## 7. Notes
- Results in `results/RESULTS.md` are **synthetic-data pipeline validation** (placeholder
  parameters), proving the differentiable identification/inverse pipeline is correct — not
  physical measurements of a real robot. Real-data validation is ongoing.
- Internal materials (slides, talk notes) are kept private and are not part of this release.
