"""6-DOF 可微多步 rollout + 轨迹损失（Phase 5）。

与 3-DOF 的 diff_rollout 同构，只是状态是 (p, q, v, w) 四组。
损失 = 位置 MSE + 姿态误差（四元数测地：1 - (q·q_ref)²，自动处理双重覆盖）。
"""

import warp as wp

from src.dynamics.fossen6dof import step_semi_implicit_euler_6dof as step6


def rollout6(p0, q0, v0, w0, tau_f_seq, tau_m_seq, dist_f_seq, dist_m_seq,
             theta, dt, num_steps, device):
    """积分 num_steps 步，返回 (ps, qs, vs, ws)，各长度 num_steps+1。"""
    batch = p0.shape[0]
    ps, qs, vs, ws = [p0], [q0], [v0], [w0]
    for t in range(num_steps):
        po = wp.zeros(batch, dtype=wp.vec3, device=device, requires_grad=True)
        qo = wp.zeros(batch, dtype=wp.quat, device=device, requires_grad=True)
        vo = wp.zeros(batch, dtype=wp.vec3, device=device, requires_grad=True)
        wo = wp.zeros(batch, dtype=wp.vec3, device=device, requires_grad=True)
        wp.launch(step6, dim=batch,
                  inputs=[ps[t], qs[t], vs[t], ws[t],
                          tau_f_seq[t], tau_m_seq[t], dist_f_seq[t], dist_m_seq[t],
                          theta, dt],
                  outputs=[po, qo, vo, wo], device=device)
        ps.append(po); qs.append(qo); vs.append(vo); ws.append(wo)
    return ps, qs, vs, ws


@wp.kernel
def accumulate_pose_mse6(p_sim: wp.array(dtype=wp.vec3),
                         p_ref: wp.array(dtype=wp.vec3),
                         q_sim: wp.array(dtype=wp.quat),
                         q_ref: wp.array(dtype=wp.quat),
                         w_att: wp.float32,
                         loss: wp.array(dtype=wp.float32)):
    i = wp.tid()
    dp = p_sim[i] - p_ref[i]
    d = wp.dot(dp, dp)
    qs = q_sim[i]
    qr = q_ref[i]
    qd = qs[0] * qr[0] + qs[1] * qr[1] + qs[2] * qr[2] + qs[3] * qr[3]
    d = d + w_att * (1.0 - qd * qd)       # 姿态测地误差
    wp.atomic_add(loss, 0, d)


def trajectory_loss6(ps, qs, p_refs, q_refs, loss_arr, w_att=1.0):
    """L = Σ_t [ ‖p_sim-p_ref‖² + w_att·(1-(q_sim·q_ref)²) ]。"""
    device = loss_arr.device
    for t in range(len(p_refs)):
        wp.launch(accumulate_pose_mse6, dim=ps[t].shape[0],
                  inputs=[ps[t], p_refs[t], qs[t], q_refs[t], float(w_att), loss_arr],
                  device=device)
    return loss_arr
