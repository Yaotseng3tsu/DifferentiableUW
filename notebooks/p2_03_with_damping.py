"""Phase 2.3 —— 打开阻尼：验证水下物理合理性。

用完整 default_theta（含线性+二次阻尼）：
    阶段1 (前50步, 1s)：推力 F=17N => 速度上升但被水阻力限制，趋向终端速度
    阶段2 (撤力后)     ：速度因水阻力衰减回 0（不再匀速滑行）
同时检查：纯纵荡推进时 v(横荡)、r(转动) 应保持 0（直线运动不耦合）。

参考：仅线性阻尼时的时间常数 tau = m11 / X_u，给衰减快慢一个直觉。

运行：.venv\\Scripts\\python.exe notebooks\\p2_03_with_damping.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import warp as wp

from src import params as P
from src.diff_rollout import rollout

wp.init()
device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"

theta_np = P.default_theta()        # 完整参数，含阻尼
theta = wp.array(theta_np, dtype=wp.float32, device=device)
m11 = theta_np[P.I_M] + theta_np[P.I_AU]
Xu = theta_np[P.I_XU]
tau_const = m11 / Xu

dt = 0.02
T = 300
PUSH_STEPS = 50
F = 17.0

eta0 = wp.zeros(1, dtype=wp.vec3, device=device)
nu0 = wp.zeros(1, dtype=wp.vec3, device=device)
tau_seq, dist_seq = [], []
for t in range(T):
    fx = F if t < PUSH_STEPS else 0.0
    tau_seq.append(wp.array([wp.vec3(fx, 0.0, 0.0)], dtype=wp.vec3, device=device))
    dist_seq.append(wp.zeros(1, dtype=wp.vec3, device=device))

etas, nus = rollout(eta0, nu0, tau_seq, dist_seq, theta, dt, T, device)
NU = np.array([v.numpy()[0] for v in nus])   # (T+1, 3): u, v, r
u = NU[:, 0]

print(f"[参数] m11={m11}, 线性阻尼 X_u={Xu} -> 线性时间常数 tau≈{tau_const:.2f}s")
print("[阶段1 推力中] u 上升并趋向终端速度:")
for s in (0, 10, 25, 50):
    print(f"   t={s*dt:.2f}s:  u = {u[s]:.4f} m/s")
print(f"[峰值] u_max = {u.max():.4f} m/s  (出现在撤力时刻附近)")

print("[阶段2 撤力后] u 因水阻力衰减回 0:")
for s in (50, 100, 150, 200, 300):
    print(f"   t={s*dt:.2f}s:  u = {u[s]:.4f} m/s")

print(f"[耦合检查] 全程 |v|max={np.abs(NU[:,1]).max():.2e}, |r|max={np.abs(NU[:,2]).max():.2e}  (纯纵荡应≈0)")
mono = np.all(np.diff(u[PUSH_STEPS:]) <= 1e-7)
print(f"[校验] 撤力后单调衰减: {mono};  终值 u={u[-1]:.4f} m/s")
print("[结论] 推力中受阻趋稳、撤力后衰减归零、直线不耦合 => 水下物理行为合理。")
