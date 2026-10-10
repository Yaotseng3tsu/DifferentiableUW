"""Phase 5.4 —— 6-DOF 参数辨识（合成数据）。

固定可测参数（m, 惯量, W, B, zb），辨识 18 个难测参数（附加质量6 + 线性阻尼6 + 二次阻尼6）。
无量纲化(/SCALE6) + 丰富机动(依次激发 surge/sway/heave/roll/pitch/yaw) + Adam。

运行：.venv\\Scripts\\python.exe scripts\\identify_params_6dof.py
"""

import os
import _bootstrap  # noqa: F401

import numpy as np
import warp as wp

from src import params6 as P6
from src.diff_rollout6 import rollout6, trajectory_loss6

wp.init()
device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"

NAMES = ["m", "Ixx", "Iyy", "Izz", "Xud", "Yvd", "Zwd", "Kpd", "Mqd", "Nrd",
         "Xu", "Yv", "Zw", "Kp", "Mq", "Nr", "Xuu", "Yvv", "Zww", "Kpp", "Mqq", "Nrr",
         "W", "B", "zb"]
dt = 0.02
T = 120


def build_inputs():
    p0 = wp.zeros(1, dtype=wp.vec3, device=device)
    q0 = wp.array([wp.quat_identity()], dtype=wp.quat, device=device)
    v0 = wp.zeros(1, dtype=wp.vec3, device=device)
    w0 = wp.zeros(1, dtype=wp.vec3, device=device)
    tau_f, tau_m, zf = [], [], []
    for t in range(T):
        ff, mm = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        if t < 20:    ff[0] = 20.0
        elif t < 40:  ff[1] = 15.0
        elif t < 60:  ff[2] = 15.0
        elif t < 80:  mm[0] = 1.5
        elif t < 100: mm[1] = 1.5
        else:         mm[2] = 1.5
        tau_f.append(wp.array([wp.vec3(*ff)], dtype=wp.vec3, device=device))
        tau_m.append(wp.array([wp.vec3(*mm)], dtype=wp.vec3, device=device))
        zf.append(wp.zeros(1, dtype=wp.vec3, device=device))
    return p0, q0, v0, w0, tau_f, tau_m, zf


P0, Q0, V0, W0, TAU_F, TAU_M, ZF = build_inputs()

theta_true = P6.default_theta6()
ps, qs, _, _ = rollout6(P0, Q0, V0, W0, TAU_F, TAU_M, ZF, ZF,
                        wp.array(theta_true, dtype=wp.float32, device=device), dt, T, device)
p_refs = [wp.array(p.numpy(), dtype=wp.vec3, device=device) for p in ps]
q_refs = [wp.array(q.numpy(), dtype=wp.quat, device=device) for q in qs]


def loss_and_grad(theta_np):
    theta = wp.array(theta_np.astype(np.float32), dtype=wp.float32,
                     device=device, requires_grad=True)
    loss_arr = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        ps2, qs2, _, _ = rollout6(P0, Q0, V0, W0, TAU_F, TAU_M, ZF, ZF, theta, dt, T, device)
        trajectory_loss6(ps2, qs2, p_refs, q_refs, loss_arr, w_att=1.0)
    tape.backward(loss=loss_arr)
    return float(loss_arr.numpy()[0]), theta.grad.numpy().copy()


IDX = P6.IDENTIFY_IDX6
theta = theta_true.copy().astype(np.float32)
theta[IDX] = theta_true[IDX] * 0.5                  # 错误初值
s = theta / P6.SCALE6
m_a = np.zeros(P6.N_PARAMS6, dtype=np.float32)
v_a = np.zeros(P6.N_PARAMS6, dtype=np.float32)
b1, b2, epsA, lr = 0.9, 0.999, 1e-8, 0.02
N_ITERS = 2500

hist_loss, hist_err = [], []
for it in range(1, N_ITERS + 1):
    theta = s * P6.SCALE6
    L, g = loss_and_grad(theta)
    g_s = g * P6.SCALE6
    m_a = b1 * m_a + (1 - b1) * g_s
    v_a = b2 * v_a + (1 - b2) * g_s * g_s
    step = lr * (m_a / (1 - b1 ** it)) / (np.sqrt(v_a / (1 - b2 ** it)) + epsA)
    for k in IDX:
        s[k] -= step[k]
    rel = np.mean(np.abs(theta[IDX] - theta_true[IDX]) / np.abs(theta_true[IDX]))
    hist_loss.append(L); hist_err.append(rel * 100)
    if it == 1 or it % 250 == 0:
        print(f"iter {it:4d}  L={L:.3e}  平均相对误差={rel*100:.2f}%")

theta = s * P6.SCALE6
print("\n参数        真值      初值(×0.5)   辨识结果")
for k in IDX:
    print(f"{NAMES[k]:<6}{theta_true[k]:>10.4f}{theta_true[k]*0.5:>12.4f}{theta[k]:>12.4f}")

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(hist_loss); ax[0].set_title("6-DOF: loss vs iteration")
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("trajectory loss (log)"); ax[0].grid(alpha=0.3)
    ax[1].plot(hist_err); ax[1].set_title("6-DOF: mean relative parameter error")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("mean rel. error [%]"); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "pic", "identify_params6_convergence.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\n已保存收敛曲线 {out}")
except ImportError:
    print("未安装 matplotlib，跳过绘图。")

print(f"[结论] 18 个难测参数从真值×0.5 收敛，最终平均相对误差 {hist_err[-1]:.2f}% => 6-DOF 参数辨识成立。")
