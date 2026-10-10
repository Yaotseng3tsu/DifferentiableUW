"""Phase 5.3 关卡 —— 6-DOF 解析梯度 vs 有限差分校验。

和 3-DOF 的 test_gradients 同思路：对一个激发全部 6 轴的机动，
比较 wp.Tape 反传的 dL/dtheta 与中心差分，逐个辨识参数用联合容差判定。

运行：.venv\\Scripts\\python.exe tests\\test_gradients6.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import warp as wp

from src import params6 as P6
from src.diff_rollout6 import rollout6, trajectory_loss6

wp.init()
DEVICE = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
DT = 0.02
T = 60

NAMES = ["m", "Ixx", "Iyy", "Izz", "Xud", "Yvd", "Zwd", "Kpd", "Mqd", "Nrd",
         "Xu", "Yv", "Zw", "Kp", "Mq", "Nr", "Xuu", "Yvv", "Zww", "Kpp", "Mqq", "Nrr",
         "W", "B", "zb"]


def build_inputs():
    p0 = wp.zeros(1, dtype=wp.vec3, device=DEVICE)
    q0 = wp.array([wp.quat_identity()], dtype=wp.quat, device=DEVICE)
    v0 = wp.zeros(1, dtype=wp.vec3, device=DEVICE)
    w0 = wp.zeros(1, dtype=wp.vec3, device=DEVICE)
    tau_f, tau_m, zf = [], [], []
    for t in range(T):
        ff = [0.0, 0.0, 0.0]
        mm = [0.0, 0.0, 0.0]
        if t < 10:
            ff[0] = 20.0          # surge
        elif t < 20:
            ff[1] = 15.0          # sway
        elif t < 30:
            ff[2] = 15.0          # heave
        elif t < 40:
            mm[0] = 1.0           # roll
        elif t < 50:
            mm[1] = 1.0           # pitch
        else:
            mm[2] = 1.0           # yaw
        tau_f.append(wp.array([wp.vec3(*ff)], dtype=wp.vec3, device=DEVICE))
        tau_m.append(wp.array([wp.vec3(*mm)], dtype=wp.vec3, device=DEVICE))
        zf.append(wp.zeros(1, dtype=wp.vec3, device=DEVICE))
    return p0, q0, v0, w0, tau_f, tau_m, zf


P0, Q0, V0, W0, TAU_F, TAU_M, ZF = build_inputs()


def gen_ref(theta_np):
    theta = wp.array(theta_np, dtype=wp.float32, device=DEVICE)
    ps, qs, _, _ = rollout6(P0, Q0, V0, W0, TAU_F, TAU_M, ZF, ZF, theta, DT, T, DEVICE)
    p_refs = [wp.array(p.numpy(), dtype=wp.vec3, device=DEVICE) for p in ps]
    q_refs = [wp.array(q.numpy(), dtype=wp.quat, device=DEVICE) for q in qs]
    return p_refs, q_refs


def loss_for(theta_np, p_refs, q_refs, requires_grad):
    theta = wp.array(theta_np.astype(np.float32), dtype=wp.float32,
                     device=DEVICE, requires_grad=requires_grad)
    loss_arr = wp.zeros(1, dtype=wp.float32, device=DEVICE, requires_grad=True)
    tape = wp.Tape()
    with tape:
        ps, qs, _, _ = rollout6(P0, Q0, V0, W0, TAU_F, TAU_M, ZF, ZF, theta, DT, T, DEVICE)
        trajectory_loss6(ps, qs, p_refs, q_refs, loss_arr, w_att=1.0)
    return theta, loss_arr, tape


def main():
    theta_ref = P6.default_theta6()
    theta_test = theta_ref.copy()
    theta_test[P6.IDENTIFY_IDX6] = theta_ref[P6.IDENTIFY_IDX6] * 1.1   # 偏离以产生非零损失
    p_refs, q_refs = gen_ref(theta_ref)

    theta, loss_arr, tape = loss_for(theta_test, p_refs, q_refs, True)
    tape.backward(loss=loss_arr)
    g_an = theta.grad.numpy().copy()

    eps = 1e-3
    g_fd = np.zeros_like(theta_test)
    for k in P6.IDENTIFY_IDX6:
        tp = theta_test.copy(); tp[k] += eps
        tm = theta_test.copy(); tm[k] -= eps
        _, lp, _ = loss_for(tp, p_refs, q_refs, False)
        _, lm, _ = loss_for(tm, p_refs, q_refs, False)
        g_fd[k] = (float(lp.numpy()[0]) - float(lm.numpy()[0])) / (2 * eps)

    atol, rtol = 1e-3, 5e-2
    print(f"{'参数':<6}{'解析梯度':>14}{'差分梯度':>14}{'绝对误差':>12}  判定")
    ok_all = True
    for k in P6.IDENTIFY_IDX6:
        ae = abs(g_an[k] - g_fd[k])
        ok = ae < atol + rtol * abs(g_fd[k])
        ok_all = ok_all and ok
        print(f"{NAMES[k]:<6}{g_an[k]:>14.5f}{g_fd[k]:>14.5f}{ae:>12.2e}  {'OK' if ok else 'FAIL'}")
    assert ok_all, "6-DOF 梯度校验失败"
    print("OK —— 6-DOF 梯度门禁通过。")


if __name__ == "__main__":
    main()
