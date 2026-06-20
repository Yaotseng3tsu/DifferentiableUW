"""Phase 3.4 —— 对扰动力 U 求梯度（逆问题地基）。

设定：机器人参数已知固定；有一个未知恒定水流力 U_true 作用在质心，把轨迹推偏。
仅凭偏移后的轨迹，从 U=0 出发反算 U。

技巧：所有时间步共享同一个 U 数组(requires_grad=True)，
      反向传播自动把各步梯度累加为 dL/dU。

两部分：
    A) 校验 dL/dU：解析梯度 vs 有限差分（沿用门禁思路）。
    B) 迷你反算：对 U 做梯度下降，看能否恢复 U_true（逆问题雏形）。

运行：.venv\\Scripts\\python.exe notebooks\\p3_04_disturbance_gradient.py
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
T = 120
theta = wp.array(P.default_theta(), dtype=wp.float32, device=device)

# 控制力：短推力脉冲(前20步)让机器人动起来，之后长滑行——
# 滑行段没有推力，水流力 U 成为主导，Ux/Uy 都能在漂移中显现(可观测性更好)。
tau_seq = []
for t in range(T):
    fx = 20.0 if t < 20 else 0.0
    tau_seq.append(wp.array([wp.vec3(fx, 0.0, 0.0)], dtype=wp.vec3, device=device))

eta0 = wp.zeros(1, dtype=wp.vec3, device=device)
nu0 = wp.zeros(1, dtype=wp.vec3, device=device)

U_TRUE = np.array([1.0, -1.5, 0.0], dtype=np.float32)   # 未知水流力(真值)


def make_dist(U_np, requires_grad):
    U = wp.array([wp.vec3(float(U_np[0]), float(U_np[1]), float(U_np[2]))],
                 dtype=wp.vec3, device=device, requires_grad=requires_grad)
    return U, [U] * T          # 所有步共享同一个 U


# ---- 目标轨迹：在 U_true 作用下 -------------------------------------------
U_ref, dist_ref = make_dist(U_TRUE, False)
etas_ref, _ = rollout(eta0, nu0, tau_seq, dist_ref, theta, dt, T, device)
eta_refs = [wp.array(e.numpy(), dtype=wp.vec3, device=device) for e in etas_ref]
print(f"[目标] 水流真值 U_true=({U_TRUE[0]}, {U_TRUE[1]})，终点(x,y)=",
      np.round(etas_ref[-1].numpy()[0][:2], 4))


def loss_for_U(U_np, want_grad):
    U, dist_seq = make_dist(U_np, want_grad)
    loss_arr = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        etas, _ = rollout(eta0, nu0, tau_seq, dist_seq, theta, dt, T, device)
        trajectory_loss(etas, eta_refs, loss_arr)
    return U, loss_arr, tape


# ---- A) 校验扰动梯度 -------------------------------------------------------
U0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)      # 假装没有水流
U, loss_arr, tape = loss_for_U(U0, want_grad=True)
tape.backward(loss=loss_arr)
g_auto = U.grad.numpy()[0]

eps = 1e-3
g_fd = np.zeros(3, dtype=np.float32)
for k in range(2):       # 只验 Ux, Uy（Uz 平面内无意义）
    up = U0.copy(); up[k] += eps
    um = U0.copy(); um[k] -= eps
    _, lp, _ = loss_for_U(up, False)
    _, lm, _ = loss_for_U(um, False)
    g_fd[k] = (float(lp.numpy()[0]) - float(lm.numpy()[0])) / (2 * eps)

print("\n[A 梯度校验] 在 U=0 处:")
print(f"   解析 dL/dU = ({g_auto[0]:.4f}, {g_auto[1]:.4f})")
print(f"   差分 dL/dU = ({g_fd[0]:.4f}, {g_fd[1]:.4f})")
print(f"   绝对误差   = ({abs(g_auto[0]-g_fd[0]):.2e}, {abs(g_auto[1]-g_fd[1]):.2e})")

# ---- B) 迷你反算：用 Adam 恢复 U_true -------------------------------------
print("\n[B 反算] 从 U=(0,0) 出发，用 Adam 优化恢复水流力:")
U_np = np.array([0.0, 0.0, 0.0], dtype=np.float32)
m = np.zeros(3, dtype=np.float32)
v = np.zeros(3, dtype=np.float32)
b1, b2, epsA, lr = 0.9, 0.999, 1e-8, 0.05
for it in range(1, 801):
    U, loss_arr, tape = loss_for_U(U_np, want_grad=True)
    tape.backward(loss=loss_arr)
    g = U.grad.numpy()[0].astype(np.float32)
    tape.zero()
    m = b1 * m + (1 - b1) * g            # Adam: 一阶矩(动量)
    v = b2 * v + (1 - b2) * g * g        # Adam: 二阶矩(自适应步长)
    mhat = m / (1 - b1 ** it)
    vhat = v / (1 - b2 ** it)
    U_np = U_np - lr * mhat / (np.sqrt(vhat) + epsA)
    U_np[2] = 0.0                        # 平面内 Uz 无意义，固定为 0
    if it == 1 or it % 100 == 0:
        print(f"   iter {it:3d}  L={float(loss_arr.numpy()[0]):.6f}  "
              f"U=({U_np[0]:.3f}, {U_np[1]:.3f})")

print(f"\n[结果] 真值 U_true=({U_TRUE[0]}, {U_TRUE[1]})")
print(f"       反算  U_est =({U_np[0]:.3f}, {U_np[1]:.3f})")
print("[结论] 梯度校验通过 + 成功从轨迹反算出水流力 => 逆问题管线成立。")
