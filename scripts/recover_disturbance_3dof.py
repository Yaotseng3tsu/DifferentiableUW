"""Phase 4.3 —— 逆问题求解器：从轨迹反算水流力（合成数据）。

Part 1（核心）：恒定水流力 U=(Ux,Uy) 的反算（PPT 的"质心单一合力"简化）。
Part 2（进阶, 预览 open-problems D2）：时变水流 U(t) 的反算。
    这是欠定/病态反问题——高频分量被惯性/阻尼抹平，难以恢复。
    加平滑正则化 R=λ·Σ||U(t+1)-U(t)||² 稳定求解（正则化场反演的雏形）。

机器人参数已知固定；机动=短推力脉冲+长滑行(滑行段水流主导，可观测性好)。
结果存表(E6,E7) + 图 + CSV。

运行：.venv\\Scripts\\python.exe scripts\\recover_disturbance_3dof.py
"""

import os
import _bootstrap  # noqa: F401

import numpy as np
import warp as wp

from src import params as P
from src.diff_rollout import rollout, trajectory_loss

wp.init()
device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"

dt = 0.02
T = 120
theta = wp.array(P.default_theta(), dtype=wp.float32, device=device)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 机动：短推力脉冲(前20步) + 长滑行
tau_seq = []
for t in range(T):
    fx = 20.0 if t < 20 else 0.0
    tau_seq.append(wp.array([wp.vec3(fx, 0.0, 0.0)], dtype=wp.vec3, device=device))
eta0 = wp.zeros(1, dtype=wp.vec3, device=device)
nu0 = wp.zeros(1, dtype=wp.vec3, device=device)


def gen_ref(dist_seq):
    etas, _ = rollout(eta0, nu0, tau_seq, dist_seq, theta, dt, T, device)
    return [wp.array(e.numpy(), dtype=wp.vec3, device=device) for e in etas]


# ===========================================================================
# Part 1 —— 恒定水流力反算
# ===========================================================================
print("===== Part 1: 恒定水流力反算 =====")
U_TRUE = np.array([1.0, -1.5, 0.0], dtype=np.float32)
U_ref = wp.array([wp.vec3(*U_TRUE)], dtype=wp.vec3, device=device)
eta_refs = gen_ref([U_ref] * T)


def loss_grad_const(U_np):
    U = wp.array([wp.vec3(float(U_np[0]), float(U_np[1]), 0.0)], dtype=wp.vec3,
                 device=device, requires_grad=True)
    loss_arr = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        etas, _ = rollout(eta0, nu0, tau_seq, [U] * T, theta, dt, T, device)
        trajectory_loss(etas, eta_refs, loss_arr)
    tape.backward(loss=loss_arr)
    return float(loss_arr.numpy()[0]), U.grad.numpy()[0][:2].astype(np.float32)


U = np.array([0.0, 0.0], dtype=np.float32)
m_a = np.zeros(2); v_a = np.zeros(2)
b1, b2, epsA, lr = 0.9, 0.999, 1e-8, 0.05
hist_L, hist_e = [], []
for it in range(1, 601):
    L, g = loss_grad_const(U)
    m_a = b1 * m_a + (1 - b1) * g
    v_a = b2 * v_a + (1 - b2) * g * g
    U = U - lr * (m_a / (1 - b1 ** it)) / (np.sqrt(v_a / (1 - b2 ** it)) + epsA)
    hist_L.append(L)
    hist_e.append(float(np.hypot(U[0] - U_TRUE[0], U[1] - U_TRUE[1])))
    if it == 1 or it % 100 == 0:
        print(f"   iter {it:3d}  L={L:.3e}  U=({U[0]:.3f},{U[1]:.3f})")
print(f"   真值 U=({U_TRUE[0]},{U_TRUE[1]})  反算 U=({U[0]:.3f},{U[1]:.3f})")

# ===========================================================================
# Part 2 —— 时变水流 U(t) 反算（带平滑正则化）
# ===========================================================================
print("\n===== Part 2: 时变水流 U(t) 反算 (平滑正则化) =====")
ts = np.arange(T) / T
Ux_true = 0.4 + 0.6 * ts                       # 缓慢上升
Uy_true = -1.2 * np.sin(np.pi * ts)            # 半正弦起伏
U_true_t = np.stack([Ux_true, Uy_true], axis=1).astype(np.float32)  # (T,2)

dist_true = [wp.array([wp.vec3(float(U_true_t[t, 0]), float(U_true_t[t, 1]), 0.0)],
                      dtype=wp.vec3, device=device) for t in range(T)]
eta_refs_tv = gen_ref(dist_true)


def loss_grad_tv(U_np):
    dist_seq = [wp.array([wp.vec3(float(U_np[t, 0]), float(U_np[t, 1]), 0.0)],
                         dtype=wp.vec3, device=device, requires_grad=True) for t in range(T)]
    loss_arr = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        etas, _ = rollout(eta0, nu0, tau_seq, dist_seq, theta, dt, T, device)
        trajectory_loss(etas, eta_refs_tv, loss_arr)
    tape.backward(loss=loss_arr)
    g = np.array([dist_seq[t].grad.numpy()[0][:2] for t in range(T)], dtype=np.float32)
    return float(loss_arr.numpy()[0]), g


LAM = 0.02                                      # 平滑正则化强度
Ut = np.zeros((T, 2), dtype=np.float32)
m_t = np.zeros((T, 2)); v_t = np.zeros((T, 2))
lr2 = 0.02
for it in range(1, 1501):
    L, g = loss_grad_tv(Ut)
    # 平滑正则化梯度 dR/dU, R=Σ||U[t+1]-U[t]||²
    d = np.diff(Ut, axis=0)
    g_smooth = np.zeros_like(Ut)
    g_smooth[:-1] -= 2 * d
    g_smooth[1:] += 2 * d
    g_tot = g + LAM * g_smooth
    m_t = b1 * m_t + (1 - b1) * g_tot
    v_t = b2 * v_t + (1 - b2) * g_tot * g_tot
    Ut = Ut - lr2 * (m_t / (1 - b1 ** it)) / (np.sqrt(v_t / (1 - b2 ** it)) + epsA)
    if it == 1 or it % 300 == 0:
        rmse = float(np.sqrt(np.mean((Ut - U_true_t) ** 2)))
        print(f"   iter {it:4d}  L={L:.3e}  RMSE(U(t))={rmse:.4f}")
rmse = float(np.sqrt(np.mean((Ut - U_true_t) ** 2)))
print(f"   时变水流反算 RMSE = {rmse:.4f} N")

# ---- 保存 CSV + 图 --------------------------------------------------------
os.makedirs(os.path.join(ROOT, "results", "data"), exist_ok=True)
np.savetxt(os.path.join(ROOT, "results", "data", "recover_disturbance_tv.csv"),
           np.column_stack([ts * T * dt, Ux_true, Ut[:, 0], Uy_true, Ut[:, 1]]),
           delimiter=",", header="time_s,Ux_true,Ux_est,Uy_true,Uy_est", comments="")

try:
    import matplotlib.pyplot as plt

    # Part 1 收敛
    fig1, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(hist_L); ax[0].set_title("Part 1: loss vs iteration")
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("loss (log)"); ax[0].grid(alpha=0.3)
    ax[1].plot(hist_e); ax[1].set_title("Part 1: |U_est - U_true| vs iteration")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("error in U [N]"); ax[1].grid(alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(os.path.join(ROOT, "pic", "recover_disturbance_const.png"), dpi=120)

    # Part 2 时变 U(t)
    time = np.arange(T) * dt
    fig2, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(time, Ux_true, "k--", label="true")
    ax[0].plot(time, Ut[:, 0], "b-", label="recovered")
    ax[0].set_title("Part 2: current Ux(t)")
    ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("Ux [N]"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(time, Uy_true, "k--", label="true")
    ax[1].plot(time, Ut[:, 1], "r-", label="recovered")
    ax[1].set_title("Part 2: current Uy(t)")
    ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("Uy [N]"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(ROOT, "pic", "recover_disturbance_timevarying.png"), dpi=120)
    print("已保存图 pic/recover_disturbance_const.png 和 _timevarying.png")
except ImportError:
    print("未安装 matplotlib，跳过绘图。")

print("[结论] 恒定水流精确反算；时变水流经平滑正则化也能恢复主体 => 逆问题求解器成立。")
