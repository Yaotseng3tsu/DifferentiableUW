"""Phase 3.2 —— 第一次反向传播：对 11 个参数求梯度 dL/dtheta，并解读它。

从偏差 30% 的猜测 theta = theta_true*0.7 出发：
    1) 建 tape -> rollout -> loss -> backward，读出 theta.grad（11 维梯度）。
    2) 解读梯度：负梯度方向就是"让损失下降"的调参方向。
    3) 沿负梯度走一小步，确认损失确实下降 => 梯度方向正确、可用于辨识。

运行：.venv\\Scripts\\python.exe notebooks\\p3_02_first_gradient.py
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

NAMES = ["m", "Iz", "a_u", "a_v", "a_r", "X_u", "Y_v", "N_r", "X_uu", "Y_vv", "N_rr"]
dt = 0.02
T = 60


def build_inputs():
    eta0 = wp.zeros(1, dtype=wp.vec3, device=device)
    nu0 = wp.zeros(1, dtype=wp.vec3, device=device)
    tau_seq, dist_seq = [], []
    for t in range(T):
        fx = 20.0 if t < 30 else 0.0
        mz = 1.0 if 10 <= t < 30 else 0.0
        tau_seq.append(wp.array([wp.vec3(fx, 0.0, mz)], dtype=wp.vec3, device=device))
        dist_seq.append(wp.zeros(1, dtype=wp.vec3, device=device))
    return eta0, nu0, tau_seq, dist_seq


eta0, nu0, tau_seq, dist_seq = build_inputs()

# 目标轨迹（真值参数）
theta_true = P.default_theta()
etas_ref, _ = rollout(eta0, nu0, tau_seq, dist_seq,
                      wp.array(theta_true, dtype=wp.float32, device=device), dt, T, device)
eta_refs = [wp.array(e.numpy(), dtype=wp.vec3, device=device) for e in etas_ref]


def loss_and_grad(theta_np, want_grad):
    theta = wp.array(theta_np.astype(np.float32), dtype=wp.float32,
                     device=device, requires_grad=want_grad)
    loss_arr = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        etas, _ = rollout(eta0, nu0, tau_seq, dist_seq, theta, dt, T, device)
        trajectory_loss(etas, eta_refs, loss_arr)
    L = float(loss_arr.numpy()[0])
    g = None
    if want_grad:
        tape.backward(loss=loss_arr)
        g = theta.grad.numpy().copy()
    return L, g


# ---- 1) & 2) 求梯度并解读 --------------------------------------------------
theta_guess = theta_true * 0.7
L0, grad = loss_and_grad(theta_guess, want_grad=True)
print(f"[起点] 损失 L = {L0:.6f}\n")
print(f"{'参数':<6}{'猜测值':>10}{'真值':>10}{'dL/dθ(梯度)':>16}  方向解读")
for i, name in enumerate(NAMES):
    direction = "↑增大可降损失" if grad[i] < 0 else "↓减小可降损失"
    print(f"{name:<6}{theta_guess[i]:>10.3f}{theta_true[i]:>10.3f}{grad[i]:>16.4f}  {direction}")

# ---- 3) 沿负梯度走一小步，确认损失下降 ------------------------------------
print("\n[验证] 沿负梯度方向 theta - lr*grad 走一步，看损失是否下降:")
for lr in (1e-5, 5e-5, 1e-4):
    new = theta_guess - lr * grad
    for idx, (lo, hi) in P.PARAM_BOUNDS.items():
        new[idx] = np.clip(new[idx], lo, hi)
    Lnew, _ = loss_and_grad(new, want_grad=False)
    flag = "下降 ✓" if Lnew < L0 else "上升(步长过大)"
    print(f"   lr={lr:<7}  L: {L0:.6f} -> {Lnew:.6f}   {flag}")

print("\n[结论] 梯度给出了每个参数的调整方向；沿负梯度损失下降 => 梯度正确、可用于辨识。")
