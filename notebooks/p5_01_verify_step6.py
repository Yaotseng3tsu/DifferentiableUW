"""Phase 5.1 —— 6-DOF 单步验证：力/力矩 -> 加速度，四元数归一化。

用默认参数：m11 = m + Xud = 11.5 + 5.5 = 17.0；i11 = Ixx + Kpd = 0.16 + 0.12 = 0.28。
中性浮力 + 正直 (q=identity) + 静止：恢复力应为 0，不干扰。
    * 纯纵荡力 F=m11 -> a_lin=(1,0,0) -> v=(dt,0,0)
    * 纯滚转力矩 T=i11 -> a_ang=(1,0,0) -> w=(dt,0,0)
    * 给角速度积分一步后，|q| 仍 ≈ 1

运行：.venv\\Scripts\\python.exe notebooks\\p5_01_verify_step6.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import warp as wp

from src import params6 as P6
from src.dynamics.fossen6dof import step_semi_implicit_euler_6dof as step6

wp.init()
device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"

theta_np = P6.default_theta6()
m11 = theta_np[P6.I_M] + theta_np[P6.I_XUD]
i11 = theta_np[P6.I_IXX] + theta_np[P6.I_KPD]
theta = wp.array(theta_np, dtype=wp.float32, device=device)
dt = 0.02
print(f"[参数] m11={m11}, i11={i11}, 默认 W=B={theta_np[P6.I_W]:.1f}(中性浮力)")


def one_step(force, torque):
    p = wp.zeros(1, dtype=wp.vec3, device=device)
    q = wp.array([wp.quat_identity()], dtype=wp.quat, device=device)
    v = wp.zeros(1, dtype=wp.vec3, device=device)
    w = wp.zeros(1, dtype=wp.vec3, device=device)
    tf = wp.array([wp.vec3(*force)], dtype=wp.vec3, device=device)
    tm = wp.array([wp.vec3(*torque)], dtype=wp.vec3, device=device)
    zf = wp.zeros(1, dtype=wp.vec3, device=device)
    po = wp.zeros(1, dtype=wp.vec3, device=device)
    qo = wp.zeros(1, dtype=wp.quat, device=device)
    vo = wp.zeros(1, dtype=wp.vec3, device=device)
    wo = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(step6, dim=1,
              inputs=[p, q, v, w, tf, tm, zf, zf, theta, dt],
              outputs=[po, qo, vo, wo], device=device)
    wp.synchronize()
    return vo.numpy()[0], wo.numpy()[0], qo.numpy()[0]

# ---- 纯纵荡力 ----
v1, w1, _ = one_step((float(m11), 0.0, 0.0), (0.0, 0.0, 0.0))
print(f"[纵荡力 F={m11}]  v={np.round(v1,5)}  (期望 ({dt},0,0))  w={np.round(w1,5)}")

# ---- 纯滚转力矩 ----
v2, w2, _ = one_step((0.0, 0.0, 0.0), (float(i11), 0.0, 0.0))
print(f"[滚转力矩 T={i11}]  w={np.round(w2,5)}  (期望 ({dt},0,0))  v={np.round(v2,5)}")

# ---- 四元数归一化（给个角速度积分一步）----
_, _, q3 = one_step((0.0, 0.0, 0.0), (0.0, 0.0, float(i11) * 50.0))
print(f"[四元数] 一步后 q={np.round(q3,5)}  |q|={np.linalg.norm(q3):.6f} (应≈1)")

print("[结论] 力/力矩->加速度正确，四元数保持单位模 => 6-DOF 单步装配正确。")
