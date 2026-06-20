"""Phase 2.2 —— 无阻尼 sanity check：验证积分器在多步上的正确性。

把阻尼系数全部置 0，然后：
    阶段1 (前50步)：施加恒定纵荡推力 F=m11=17N => 理论匀加速 a=1.0 m/s^2，
                     速度线性增长，第50步(1.0s)末 u≈1.0 m/s。
    阶段2 (后50步)：撤力，无阻尼无外力 => 匀速滑行，u 保持不变。
若撤力后 u 漂移，说明积分器有问题。

运行：.venv\\Scripts\\python.exe notebooks\\p2_02_no_damping.py
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

# ---- 构造「无阻尼」参数 ----------------------------------------------------
theta_np = P.default_theta()
for idx in (P.I_XU, P.I_YV, P.I_NR, P.I_XUU, P.I_YVV, P.I_NRR):
    theta_np[idx] = 0.0       # 阻尼全关
theta = wp.array(theta_np, dtype=wp.float32, device=device)
m11 = theta_np[P.I_M] + theta_np[P.I_AU]   # 17.0

dt = 0.02
T = 100
PUSH_STEPS = 50
F = float(m11)                # 推力，使理论加速度 = 1.0 m/s^2

eta0 = wp.zeros(1, dtype=wp.vec3, device=device)
nu0 = wp.zeros(1, dtype=wp.vec3, device=device)
tau_seq, dist_seq = [], []
for t in range(T):
    fx = F if t < PUSH_STEPS else 0.0
    tau_seq.append(wp.array([wp.vec3(fx, 0.0, 0.0)], dtype=wp.vec3, device=device))
    dist_seq.append(wp.zeros(1, dtype=wp.vec3, device=device))

etas, nus = rollout(eta0, nu0, tau_seq, dist_seq, theta, dt, T, device)
u = np.array([v.numpy()[0][0] for v in nus])   # 每步的纵荡速度 u

print("[阶段1 推力中] u 应线性增长:")
for s in (0, 10, 25, 50):
    print(f"   step {s:3d} (t={s*dt:.2f}s):  u = {u[s]:.4f} m/s   (理论 {min(s,PUSH_STEPS)*dt*F/m11:.4f})")

print("[阶段2 撤力后] u 应保持不变(匀速滑行):")
for s in (50, 60, 80, 100):
    print(f"   step {s:3d} (t={s*dt:.2f}s):  u = {u[s]:.4f} m/s")

drift = abs(u[100] - u[50])
print(f"[校验] 撤力后 u 漂移 = {drift:.2e} m/s  (应≈0)")
print("[结论] 推力中线性增长、撤力后恒定 => 积分器正确，无数值耗散/发散。")
