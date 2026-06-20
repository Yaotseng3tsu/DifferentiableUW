"""Phase 3.1 —— 理解可微分 rollout + 轨迹损失（只做正向，不求梯度）。

目标：搞清楚「损失 L」在衡量什么，以及它对参数的依赖。
    1) 用参数 theta_ref 跑一条轨迹，冻结成"目标轨迹" eta_ref。
    2) 再用同样的 theta_ref 算损失 -> 应≈0（自己和自己比）。
    3) 用偏差 10% 的参数算损失 -> 应>0（参数不对，轨迹就对不上）。
这说明：损失是参数的函数 L(theta)，Phase 3.2 就是要算 dL/dtheta。

运行：.venv\\Scripts\\python.exe notebooks\\p3_01_rollout_loss.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import warp as wp

from src import params as P
from src.diff_rollout import rollout, trajectory_loss

wp.init()
device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"

dt = 0.02
T = 60


def build_inputs():
    eta0 = wp.zeros(1, dtype=wp.vec3, device=device)
    nu0 = wp.zeros(1, dtype=wp.vec3, device=device)
    tau_seq, dist_seq = [], []
    for t in range(T):
        fx = 20.0 if t < 30 else 0.0
        mz = 1.0 if 10 <= t < 30 else 0.0     # 带点转弯，激发更多参数
        tau_seq.append(wp.array([wp.vec3(fx, 0.0, mz)], dtype=wp.vec3, device=device))
        dist_seq.append(wp.zeros(1, dtype=wp.vec3, device=device))
    return eta0, nu0, tau_seq, dist_seq


def compute_loss(theta_np):
    """正向跑一遍并返回标量损失值（不建 tape，不求梯度）。"""
    eta0, nu0, tau_seq, dist_seq = build_inputs()
    theta = wp.array(theta_np.astype(np.float32), dtype=wp.float32, device=device)
    etas, _ = rollout(eta0, nu0, tau_seq, dist_seq, theta, dt, T, device)
    loss_arr = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    trajectory_loss(etas, eta_refs, loss_arr)
    return float(loss_arr.numpy()[0])


# ---- 1) 造目标轨迹（用真值参数）-------------------------------------------
theta_ref = P.default_theta()
eta0, nu0, tau_seq, dist_seq = build_inputs()
theta_ref_w = wp.array(theta_ref, dtype=wp.float32, device=device)
etas_ref, _ = rollout(eta0, nu0, tau_seq, dist_seq, theta_ref_w, dt, T, device)
eta_refs = [wp.array(e.numpy(), dtype=wp.vec3, device=device) for e in etas_ref]
print(f"[目标] 用真值参数跑出 {T} 步轨迹，终点 (x,y)=",
      np.round(etas_ref[-1].numpy()[0][:2], 4))

# ---- 2) 同参数 -> 损失≈0 ---------------------------------------------------
print(f"[损失] 参数 = 真值          L = {compute_loss(theta_ref):.6e}   (应≈0)")

# ---- 3) 偏差参数 -> 损失>0 -------------------------------------------------
for scale in (1.1, 1.3, 0.7):
    print(f"[损失] 参数 = 真值×{scale:<4} L = {compute_loss(theta_ref * scale):.6e}")

print("[结论] 损失 L 是参数 theta 的函数：参数对 -> L=0；越偏 -> L 越大。")
print("       下一步(3.2)就是求 dL/dtheta，告诉优化器每个参数该往哪调。")
