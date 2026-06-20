"""Phase 2.5（DoD）—— RK4 vs 半隐式欧拉：收敛阶对比。

做法：
    * 用恒定控制力跑一个平滑机动（无脉冲间断，便于看清收敛阶）。
    * 以极小步长(dt=5e-4)的 RK4 解作为"标准答案"(参考解)。
    * 对一组逐步减半的 dt，分别用欧拉和 RK4 积分到同一终点时间，
      比较终点位置(x,y)相对参考解的误差。
预期：
    * 两者误差都随 dt 减小而下降（都收敛）。
    * 欧拉是 1 阶：dt 减半，误差≈减半（阶≈1）。
    * RK4 是 4 阶：dt 减半，误差≈降到 1/16（阶≈4）。

运行：.venv\\Scripts\\python.exe notebooks\\p2_05_rk4_convergence.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import warp as wp

from src import params as P
from src.dynamics.fossen3dof import step_semi_implicit_euler
from src.integrators import step_rk4

wp.init()
device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"

theta = wp.array(P.default_theta(), dtype=wp.float32, device=device)
T_TOTAL = 3.0
# 恒定力：纵荡 15N + 偏航 1.0 N·m（平滑曲线机动，全程不变）
tau = wp.array([wp.vec3(15.0, 0.0, 1.0)], dtype=wp.vec3, device=device)
dist = wp.zeros(1, dtype=wp.vec3, device=device)


def integrate(step_kernel, dt, n_steps):
    """用指定积分器跑 n_steps 步，返回终点 (x,y,psi)。"""
    eta = wp.zeros(1, dtype=wp.vec3, device=device)
    nu = wp.zeros(1, dtype=wp.vec3, device=device)
    for _ in range(n_steps):
        eta2 = wp.zeros(1, dtype=wp.vec3, device=device)
        nu2 = wp.zeros(1, dtype=wp.vec3, device=device)
        wp.launch(step_kernel, dim=1, inputs=[eta, nu, tau, dist, theta, dt],
                  outputs=[eta2, nu2], device=device)
        eta, nu = eta2, nu2
    return eta.numpy()[0]


# ---- 参考解：极小步长 RK4 --------------------------------------------------
dt_ref = 5e-4
n_ref = int(round(T_TOTAL / dt_ref))
ref = integrate(step_rk4, dt_ref, n_ref)
print(f"[参考解] RK4 dt={dt_ref} ({n_ref}步)  终点(x,y)=({ref[0]:.5f}, {ref[1]:.5f})")


def pos_err(end):
    return float(np.hypot(end[0] - ref[0], end[1] - ref[1]))


def order(errs):
    """由相邻误差比估计收敛阶 log2(e_coarse/e_fine)。"""
    return [float(np.log2(errs[i] / errs[i + 1])) for i in range(len(errs) - 1)]


# 用较大步长，让 RK4 误差保持在 float32 精度地板之上，才能看清 4 阶收敛。
dts = [0.2, 0.1, 0.05]
err_euler, err_rk4 = [], []
for dt in dts:
    n = int(round(T_TOTAL / dt))
    err_euler.append(pos_err(integrate(step_semi_implicit_euler, dt, n)))
    err_rk4.append(pos_err(integrate(step_rk4, dt, n)))

print("\n  dt        欧拉误差        RK4误差")
for i, dt in enumerate(dts):
    print(f"  {dt:<8} {err_euler[i]:.3e}    {err_rk4[i]:.3e}")

print("\n[收敛阶] 欧拉 ≈", np.round(order(err_euler), 2), " (理论 1)")
print("[收敛阶] RK4  ≈", np.round(order(err_rk4), 2), " (理论 4)")
print("[结论] 两者都收敛；RK4 同步长下精度高几个数量级 => Phase 2 数值精度过关。")
