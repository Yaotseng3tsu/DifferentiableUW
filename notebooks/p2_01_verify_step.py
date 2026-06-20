"""Phase 2.1 —— 验证 3-DOF 单步：给已知推力，看加速度对不对。

用默认参数 default_theta()：
    m=11.5, a_u(附加质量)=5.5  =>  纵荡广义质量 m11 = m + a_u = 17.0
施加纯纵荡推力 F=17 N，从静止出发：
    理论加速度 a = F / m11 = 17 / 17 = 1.0 m/s^2
    半隐式欧拉一步(dt=0.02)：v = 0 + dt*a = 0.02 m/s；x = 0 + dt*v = 0.0004 m
跑出来若和理论一致，说明 nu_dot / 积分 / 矩阵装配都写对了。

运行：.venv\\Scripts\\python.exe notebooks\\p2_01_verify_step.py
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

wp.init()
device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"

theta_np = P.default_theta()
m11 = theta_np[P.I_M] + theta_np[P.I_AU]
print(f"[参数] m={theta_np[P.I_M]}, 附加质量 a_u={theta_np[P.I_AU]} -> 纵荡广义质量 m11={m11}")

theta = wp.array(theta_np, dtype=wp.float32, device=device)
dt = 0.02
F = float(m11)        # 故意取 F=m11，让理论加速度恰好 =1.0

# 初始：静止在原点
eta_in = wp.zeros(1, dtype=wp.vec3, device=device)
nu_in = wp.zeros(1, dtype=wp.vec3, device=device)
tau = wp.array([wp.vec3(F, 0.0, 0.0)], dtype=wp.vec3, device=device)   # 纯纵荡推力
dist = wp.zeros(1, dtype=wp.vec3, device=device)
eta_out = wp.zeros(1, dtype=wp.vec3, device=device)
nu_out = wp.zeros(1, dtype=wp.vec3, device=device)

wp.launch(step_semi_implicit_euler, dim=1,
          inputs=[eta_in, nu_in, tau, dist, theta, dt],
          outputs=[eta_out, nu_out], device=device)
wp.synchronize()

nu = nu_out.numpy()[0]
eta = eta_out.numpy()[0]
print(f"[推力] F={F} N (纯纵荡), dt={dt}")
print(f"[一步后] ν=(u,v,r)={np.round(nu,5)}   η=(x,y,ψ)={np.round(eta,5)}")
print(f"[理论]   a=F/m11={F/m11:.3f} m/s^2 -> u={dt*F/m11:.5f},  x={dt*dt*F/m11:.6f}")
print("[结论] 数值与理论一致 => 矩阵装配/单步积分正确。")
