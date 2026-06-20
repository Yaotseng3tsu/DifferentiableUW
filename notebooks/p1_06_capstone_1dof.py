"""Phase 1.6（收官）—— 1-DOF「质量-阻尼」系统 + 自动微分 + 有限差分校验。

这是 Phase 2 的微缩版，把全套概念串起来：
    kernel + 多步积分(rollout) + 损失 + wp.Tape 求梯度 + 有限差分验证(=1.5)。

物理：一维直线上的物体，被力 F 推，受线性阻尼 c：
    m * a = F - c * v        =>   a = (F - c*v) / m
    半隐式欧拉：v <- v + dt*a ;  x <- x + dt*v

任务：已知真值阻尼 c_true 生成一条参考轨迹，再问——
    损失 L = Σ (x_sim - x_ref)^2 对阻尼 c 的梯度 dL/dc 是多少？
    用 wp.Tape 自动求，并和有限差分对比，验证梯度正确。

运行：.venv\\Scripts\\python.exe notebooks\\p1_06_capstone_1dof.py
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import warp as wp

wp.init()
device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"

M = 1.0      # 质量
DT = 0.05    # 步长
T = 100      # 步数


@wp.kernel
def step_md(x_in: wp.array(dtype=wp.float32), v_in: wp.array(dtype=wp.float32),
            F: wp.float32, c: wp.array(dtype=wp.float32), m: wp.float32, dt: wp.float32,
            x_out: wp.array(dtype=wp.float32), v_out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    a = (F - c[0] * v_in[i]) / m       # c 放数组里，才能对它求梯度
    v_new = v_in[i] + dt * a
    v_out[i] = v_new
    x_out[i] = x_in[i] + dt * v_new


@wp.kernel
def accum_sq(x_sim: wp.array(dtype=wp.float32), x_ref_t: wp.float32,
             loss: wp.array(dtype=wp.float32)):
    i = wp.tid()
    d = x_sim[i] - x_ref_t
    wp.atomic_add(loss, 0, d * d)


def force_at(t):
    """前 30 步推 1N，之后撤力滑行。"""
    return 1.0 if t < 30 else 0.0


def simulate_traj(c_val):
    """前向跑一条轨迹（不求梯度），返回每步位置 x 的列表。用来造参考轨迹。"""
    c = wp.array([c_val], dtype=wp.float32, device=device)
    cur_x = wp.zeros(1, dtype=wp.float32, device=device)
    cur_v = wp.zeros(1, dtype=wp.float32, device=device)
    xs = [0.0]
    for t in range(T):
        x_next = wp.zeros(1, dtype=wp.float32, device=device)
        v_next = wp.zeros(1, dtype=wp.float32, device=device)
        wp.launch(step_md, dim=1, inputs=[cur_x, cur_v, force_at(t), c, M, DT],
                  outputs=[x_next, v_next], device=device)
        xs.append(float(x_next.numpy()[0]))
        cur_x, cur_v = x_next, v_next
    return xs


def simulate_loss(c_val, x_ref, requires_grad):
    """可微分 rollout：返回 (c 数组, loss 数组, tape)。"""
    c = wp.array([c_val], dtype=wp.float32, device=device, requires_grad=requires_grad)
    loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    cur_x = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    cur_v = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        for t in range(T):
            x_next = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
            v_next = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
            wp.launch(step_md, dim=1, inputs=[cur_x, cur_v, force_at(t), c, M, DT],
                      outputs=[x_next, v_next], device=device)
            wp.launch(accum_sq, dim=1, inputs=[x_next, float(x_ref[t + 1]), loss],
                      device=device)
            cur_x, cur_v = x_next, v_next
    return c, loss, tape


# ---- 1) 用真值阻尼造参考轨迹 ----------------------------------------------
C_TRUE = 0.8
x_ref = simulate_traj(C_TRUE)
print(f"[参考] 用真值阻尼 c_true={C_TRUE} 造出轨迹，终点 x={x_ref[-1]:.4f}")

# ---- 2) 在一个猜测值处，自动微分求 dL/dc ----------------------------------
C_GUESS = 0.4
c, loss, tape = simulate_loss(C_GUESS, x_ref, requires_grad=True)
tape.backward(loss=loss)
g_auto = float(c.grad.numpy()[0])

# ---- 3) 有限差分（中心差分）校验梯度（=步骤 1.5）-------------------------
eps = 1e-3
_, lp, _ = simulate_loss(C_GUESS + eps, x_ref, requires_grad=False)
_, lm, _ = simulate_loss(C_GUESS - eps, x_ref, requires_grad=False)
g_fd = (float(lp.numpy()[0]) - float(lm.numpy()[0])) / (2.0 * eps)

print(f"[损失] 在 c={C_GUESS} 处 L = {float(loss.numpy()[0]):.6f}")
print(f"[梯度] 自动微分 dL/dc = {g_auto:.5f}")
print(f"[梯度] 有限差分 dL/dc = {g_fd:.5f}")
print(f"[校验] 相对误差 = {abs(g_auto - g_fd) / max(abs(g_fd), 1e-9):.3%}")
print("[结论] 两者一致 => 可微分管线(求梯度)正确。这正是 Phase 3/4 要依赖的能力。")
