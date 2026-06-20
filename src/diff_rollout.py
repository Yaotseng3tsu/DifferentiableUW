"""可微分多步 rollout + 轨迹损失（Phase 3 的核心）。

把单步 kernel 在 Python 循环里连续 launch，并全部包在 `wp.Tape` 上下文内，
就能让自动微分贯穿整段轨迹。每一步的状态保存在独立数组中（不就地覆盖），
这是 Warp 反向传播能正确工作的前提。

典型用法：
    tape = wp.Tape()
    with tape:
        etas, nus = rollout(eta0, nu0, tau_seq, dist_seq, theta, dt, T)
        loss = trajectory_loss(etas, eta_ref, loss_arr)
    tape.backward(loss=loss_arr)     # 之后 theta.grad / dist_seq.grad 即为梯度
"""

import warp as wp

from src.dynamics.fossen3dof import step_semi_implicit_euler


def alloc_state(batch: int, device, requires_grad: bool = True):
    """分配一对 (eta, nu) 状态数组。"""
    eta = wp.zeros(batch, dtype=wp.vec3, device=device, requires_grad=requires_grad)
    nu = wp.zeros(batch, dtype=wp.vec3, device=device, requires_grad=requires_grad)
    return eta, nu


def rollout(eta0, nu0, tau_seq, dist_seq, theta, dt, num_steps, device):
    """从初始状态积分 num_steps 步，返回每一步的 (eta, nu) 数组列表。

    参数
    ----
    eta0, nu0 : wp.array(vec3)         初始位姿/速度，形状 (batch,)
    tau_seq   : list[wp.array(vec3)]   每步控制力，长度 num_steps
    dist_seq  : list[wp.array(vec3)]   每步扰动力，长度 num_steps
    theta     : wp.array(float32)      参数向量（requires_grad=True 时可辨识）
    dt        : float                  时间步长
    num_steps : int
    返回
    ----
    etas, nus : list[wp.array(vec3)]，长度 num_steps+1（含初始状态）
    """
    batch = eta0.shape[0]
    etas = [eta0]
    nus = [nu0]
    for t in range(num_steps):
        eta_next, nu_next = alloc_state(batch, device, requires_grad=True)
        wp.launch(
            step_semi_implicit_euler,
            dim=batch,
            inputs=[etas[t], nus[t], tau_seq[t], dist_seq[t], theta, dt],
            outputs=[eta_next, nu_next],
            device=device,
        )
        etas.append(eta_next)
        nus.append(nu_next)
    return etas, nus


@wp.kernel
def accumulate_pose_mse(
    eta_sim: wp.array(dtype=wp.vec3),
    eta_ref: wp.array(dtype=wp.vec3),
    loss: wp.array(dtype=wp.float32),
):
    """把某一步的位姿平方误差累加进标量 loss[0]（用 atomic_add 以支持 batch 求和）。"""
    i = wp.tid()
    d = eta_sim[i] - eta_ref[i]
    wp.atomic_add(loss, 0, wp.dot(d, d))


def trajectory_loss(etas, eta_refs, loss_arr):
    """L = sum_t ||eta_sim(t) - eta_ref(t)||^2，结果写入 loss_arr[0]。

    eta_refs : list[wp.array(vec3)]，与 etas 对齐的真值轨迹。
    loss_arr : wp.array(float32, shape=1, requires_grad=True)，需事先 zero。
    """
    device = loss_arr.device
    for t in range(len(eta_refs)):
        wp.launch(
            accumulate_pose_mse,
            dim=etas[t].shape[0],
            inputs=[etas[t], eta_refs[t], loss_arr],
            device=device,
        )
    return loss_arr
