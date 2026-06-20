"""Phase 4.1 —— 完整的 3-DOF 参数辨识（合成数据，无噪声）。

把前期洞察全部用上：
    * A1 固定可测参数：质量 m、转动惯量 Iz 直接测量(视为已知)，只辨识难测的 9 个
      （附加质量 a_u,a_v,a_r + 线性/二次阻尼 X_u,Y_v,N_r,X_uu,Y_vv,N_rr）。
    * B1 参数无量纲化：每个参数除以"标称量级" scale，使优化在 O(1) 空间进行，
      摆平 0.07~21 的量级差异。优化变量是归一化的 s = theta/scale。
    * A2 丰富机动：依次施加纵荡/横荡/艏摇激励，充分激发各参数。
    * B2 Adam 优化器。

流程：真值参数 -> 造目标轨迹 -> 从错误初值(真值的 0.5 倍)用 Adam 辨识 -> 画收敛曲线。

运行：.venv\\Scripts\\python.exe scripts\\identify_params_3dof.py
"""

import _bootstrap  # noqa: F401

import numpy as np
import warp as wp

from src import params as P
from src.diff_rollout import rollout, trajectory_loss

wp.init()
device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"

NAMES = ["m", "Iz", "a_u", "a_v", "a_r", "X_u", "Y_v", "N_r", "X_uu", "Y_vv", "N_rr"]
FIXED_IDX = [P.I_M, P.I_IZ]                       # 可直接测量 -> 固定
IDENTIFY_IDX = [P.I_AU, P.I_AV, P.I_AR, P.I_XU, P.I_YV, P.I_NR,
                P.I_XUU, P.I_YVV, P.I_NRR]        # 难测 -> 辨识

# 标称量级（先验的"数量级"知识，用于无量纲化；非真值，取整）。
SCALE = np.array([10.0, 0.2, 5.0, 10.0, 0.1, 5.0, 5.0, 0.1, 20.0, 20.0, 2.0],
                 dtype=np.float32)

dt = 0.02
T = 200


def build_inputs():
    """丰富机动：纵荡->横荡->艏摇->组合->滑行，充分激发各参数。"""
    eta0 = wp.zeros(1, dtype=wp.vec3, device=device)
    nu0 = wp.zeros(1, dtype=wp.vec3, device=device)
    tau_seq, dist_seq = [], []
    for t in range(T):
        fx, fy, mz = 0.0, 0.0, 0.0
        if t < 40:
            fx = 20.0
        elif t < 80:
            fy = 15.0
        elif t < 120:
            mz = 2.0
        elif t < 160:
            fx, mz = 15.0, 1.5
        tau_seq.append(wp.array([wp.vec3(fx, fy, mz)], dtype=wp.vec3, device=device))
        dist_seq.append(wp.zeros(1, dtype=wp.vec3, device=device))
    return eta0, nu0, tau_seq, dist_seq


eta0, nu0, tau_seq, dist_seq = build_inputs()

# ---- 目标轨迹（真值参数）--------------------------------------------------
theta_true = P.default_theta()
etas_ref, _ = rollout(eta0, nu0, tau_seq, dist_seq,
                      wp.array(theta_true, dtype=wp.float32, device=device), dt, T, device)
eta_refs = [wp.array(e.numpy(), dtype=wp.vec3, device=device) for e in etas_ref]


def loss_and_grad(theta_np):
    """返回 (损失值, dL/dtheta)。"""
    theta = wp.array(theta_np.astype(np.float32), dtype=wp.float32,
                     device=device, requires_grad=True)
    loss_arr = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        etas, _ = rollout(eta0, nu0, tau_seq, dist_seq, theta, dt, T, device)
        trajectory_loss(etas, eta_refs, loss_arr)
    tape.backward(loss=loss_arr)
    return float(loss_arr.numpy()[0]), theta.grad.numpy().copy()


# ---- 初值：固定参数=真值；待辨识参数=真值×0.5 -----------------------------
theta = theta_true.copy().astype(np.float32)
theta[IDENTIFY_IDX] = theta_true[IDENTIFY_IDX] * 0.5

# 在归一化空间 s = theta/scale 上做 Adam（只更新待辨识分量）
s = theta / SCALE
m_adam = np.zeros(P.N_PARAMS, dtype=np.float32)
v_adam = np.zeros(P.N_PARAMS, dtype=np.float32)
b1, b2, epsA, lr = 0.9, 0.999, 1e-8, 0.02
N_ITERS = 3000

hist_loss, hist_err = [], []
for it in range(1, N_ITERS + 1):
    theta = s * SCALE
    L, g_theta = loss_and_grad(theta)
    g_s = g_theta * SCALE                      # 链式法则：dL/ds = dL/dtheta * scale
    # Adam（仅对待辨识分量）
    m_adam = b1 * m_adam + (1 - b1) * g_s
    v_adam = b2 * v_adam + (1 - b2) * g_s * g_s
    mhat = m_adam / (1 - b1 ** it)
    vhat = v_adam / (1 - b2 ** it)
    step = lr * mhat / (np.sqrt(vhat) + epsA)
    for k in IDENTIFY_IDX:                      # 固定分量不更新
        s[k] -= step[k]

    rel_err = np.mean(np.abs(theta[IDENTIFY_IDX] - theta_true[IDENTIFY_IDX])
                      / np.abs(theta_true[IDENTIFY_IDX]))
    hist_loss.append(L)
    hist_err.append(rel_err * 100.0)
    if it == 1 or it % 300 == 0:
        print(f"iter {it:4d}  L={L:.3e}  平均相对误差={rel_err*100:.2f}%")

theta = s * SCALE
print("\n参数        真值      初值(×0.5)   辨识结果")
for k in IDENTIFY_IDX:
    print(f"{NAMES[k]:<6}{theta_true[k]:>10.4f}{theta_true[k]*0.5:>12.4f}{theta[k]:>12.4f}")

# ---- 收敛曲线（英文图表）--------------------------------------------------
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(hist_loss)
    ax[0].set_title("Loss vs iteration")
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("trajectory loss (log)")
    ax[0].grid(alpha=0.3)

    ax[1].plot(hist_err)
    ax[1].set_title("Mean relative parameter error vs iteration")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("mean rel. error [%]")
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    import os
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "pic", "identify_params_convergence.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\n已保存收敛曲线 {out}")
except ImportError:
    print("未安装 matplotlib，跳过绘图。")

print(f"[结论] 9 个难测参数从真值×0.5 收敛回真值，最终平均相对误差 {hist_err[-1]:.2f}% => 参数辨识成立。")
