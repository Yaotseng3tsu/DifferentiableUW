# DifferentiableUW

A differentiable simulator for underwater-robot dynamics, based on the Fossen
equations of motion and implemented with [NVIDIA Warp](https://github.com/NVIDIA/warp).
Because the whole physics pipeline is differentiable, the simulator can be run in
reverse: from an observed trajectory, gradient-based optimization recovers the
robot's hydrodynamic parameters and the environmental disturbance (water current)
acting on it.

> Research code. The 3-DOF planar model is validated on synthetic data; validation
> on real experimental data is in progress.

## Motivation

Underwater field robotics is limited by data scarcity and a large sim-to-real gap,
especially in modeling hydrodynamic forces and environmental currents. Existing
underwater simulators offer strong rendering but limited, non-differentiable
physics. This project targets the missing piece — a physically grounded, *differentiable*
dynamics model that supports system identification and disturbance estimation.

## Approach

The vehicle is described by the Fossen equations of motion in the body frame,

```
M v̇ + C(v) v + D(v) v + g(η) = τ + τ_env ,      η̇ = J(η) v
```

where `M` includes rigid-body and added mass, `C` is the Coriolis/centripetal
term, `D(v)` is linear plus quadratic hydrodynamic damping, `g(η)` is the restoring
force, `τ` is control input, and `τ_env` is the environmental disturbance.

The equations are implemented as Warp GPU kernels and advanced by a differentiable
time integrator (semi-implicit Euler or RK4). Reverse-mode automatic differentiation
propagates gradients of a trajectory-matching loss back to the physical parameters
or to the disturbance force, which are then optimized with Adam. Development follows
a reduced-dimensionality strategy: a 3-DOF planar model first, then a full 6-DOF
extension.

## Features

- Differentiable 3-DOF Fossen dynamics as NVIDIA Warp GPU kernels
- Semi-implicit Euler and RK4 integrators
- Gradient verification against finite differences
- Hydrodynamic parameter identification from trajectories
- Inverse estimation of current disturbances (constant and time-varying)
- Robustness analysis with respect to measurement noise and initialization

## Results

On synthetic data (see [`results/RESULTS.md`](results/RESULTS.md)):

- analytic gradients match finite differences for all parameters;
- nine hydrodynamic parameters recovered to about 0.6% mean error;
- a constant current force recovered exactly; a time-varying current recovered in
  part, with an observability limit at the trajectory tail;
- identification stays accurate under centimetre-level trajectory noise.

These results validate that the differentiable pipeline is correct; they use
placeholder parameters and are not measurements of a physical robot.

## Getting started

See [`GETTING_STARTED.md`](GETTING_STARTED.md). In short:

```bash
python -m venv .venv && . .venv/Scripts/activate    # Windows
pip install -r requirements.txt
python notebooks/00_warp_basics.py                  # check GPU + Warp
python scripts/run_forward_3dof.py                  # forward simulation
python scripts/identify_params_3dof.py              # parameter identification
```

Requires an NVIDIA GPU with driver ≥ 525 and Python ≥ 3.10. The `warp-lang` wheel
bundles the CUDA runtime, so a separate CUDA Toolkit is not required.

## Repository layout

```
src/         core library
  dynamics/  Fossen equations of motion (3-DOF; 6-DOF in progress)
  integrators.py    differentiable integrators
  diff_rollout.py   taped rollout + trajectory loss
  params.py         identifiable parameter vector
scripts/     runnable experiments (forward sim, identification, inverse problem)
notebooks/   incremental examples
tests/       gradient verification
results/     quantitative results and data
pic/         figures
```

## Status

The 3-DOF differentiable model, parameter identification, and the inverse-disturbance
solver are implemented and validated on synthetic data.

## Future plan

Near term:

- **Validation on real experimental data.** Towing trials — constant-velocity runs to
  isolate hydrodynamic drag, and accelerating/oscillatory runs to excite the added-mass
  terms — together with free-running trials, using camera-tracked trajectories and
  onboard IMU. Identified parameters are compared against the differentiable estimates
  and against classically computed hydrodynamic values.
- **Extension to a fully coupled 6-DOF model.** Adding heave, roll and pitch, the
  gravity/buoyancy restoring forces, and a quaternion-based attitude representation to
  avoid Euler-angle singularities, reusing the existing differentiable pipeline.

Long term:

- Integrating the differentiable dynamics into NVIDIA Isaac Sim for high-fidelity rendering.
- Coupling with a differentiable sonar model toward an end-to-end perception-to-action pipeline.

## References

- T. I. Fossen, *Handbook of Marine Craft Hydrodynamics and Motion Control*, Wiley, 2011.
- NVIDIA Warp — https://github.com/NVIDIA/warp
- J. Song et al., "OceanSim: A GPU-Accelerated Underwater Robot Perception Simulation Framework," IROS 2025.
- R. Newbury et al., "A Review of Differentiable Simulators," *IEEE Access*, 2024.
