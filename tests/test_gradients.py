"""Phase 3 关卡 —— 解析梯度 vs 有限差分梯度校验。

这是整个项目的“门禁”：如果 wp.Tape 反向传播得到的 dL/dtheta 与
数值有限差分不一致，那么 Phase 4 的参数辨识与逆问题都不可信。

运行：
    python -m pytest tests/test_gradients.py -v
或直接：
    python tests/test_gradients.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台中文输出不报错

import numpy as np
import warp as wp

from src import params as P
from src.diff_rollout import rollout, trajectory_loss

wp.init()
DEVICE = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"

DT = 0.02
T = 60
BATCH = 1


def _build_inputs():
    eta0 = wp.zeros(BATCH, dtype=wp.vec3, device=DEVICE)
    nu0 = wp.zeros(BATCH, dtype=wp.vec3, device=DEVICE)
    tau_seq, dist_seq = [], []
    for t in range(T):
        fx = 20.0 if t < 30 else 0.0
        tau_seq.append(wp.array(np.array([[fx, 0.0, 1.0]], dtype=np.float32),
                                dtype=wp.vec3, device=DEVICE))
        dist_seq.append(wp.zeros(BATCH, dtype=wp.vec3, device=DEVICE))
    return eta0, nu0, tau_seq, dist_seq


def _loss_for(theta_np, eta_refs, eta0, nu0, tau_seq, dist_seq, requires_grad):
    theta = wp.array(theta_np.astype(np.float32), dtype=wp.float32,
                     device=DEVICE, requires_grad=requires_grad)
    loss_arr = wp.zeros(1, dtype=wp.float32, device=DEVICE, requires_grad=True)
    tape = wp.Tape()
    with tape:
        etas, _ = rollout(eta0, nu0, tau_seq, dist_seq, theta, DT, T, DEVICE)
        trajectory_loss(etas, eta_refs, loss_arr)
    return theta, loss_arr, tape


def test_gradient_matches_finite_difference():
    eta0, nu0, tau_seq, dist_seq = _build_inputs()

    # 参考轨迹：用稍微扰动过的参数生成，使损失非零。
    theta_ref = P.default_theta()
    theta_test = theta_ref * 1.1
    ref_theta = wp.array(theta_ref, dtype=wp.float32, device=DEVICE)
    etas_ref, _ = rollout(eta0, nu0, tau_seq, dist_seq, ref_theta, DT, T, DEVICE)
    eta_refs = [wp.array(e.numpy(), dtype=wp.vec3, device=DEVICE) for e in etas_ref]

    # 解析梯度（反向传播）。
    theta, loss_arr, tape = _loss_for(theta_test, eta_refs, eta0, nu0,
                                      tau_seq, dist_seq, requires_grad=True)
    tape.backward(loss=loss_arr)
    grad_analytic = theta.grad.numpy().copy()

    # 有限差分（中心差分）。
    eps = 1e-3
    grad_fd = np.zeros_like(theta_test)
    for k in range(P.N_PARAMS):
        tp = theta_test.copy(); tp[k] += eps
        tm = theta_test.copy(); tm[k] -= eps
        _, lp, _ = _loss_for(tp, eta_refs, eta0, nu0, tau_seq, dist_seq, False)
        _, lm, _ = _loss_for(tm, eta_refs, eta0, nu0, tau_seq, dist_seq, False)
        grad_fd[k] = (lp.numpy()[0] - lm.numpy()[0]) / (2.0 * eps)

    # 联合容差判据（绝对 + 相对），等价于 numpy.allclose 的思路：
    #   通过条件: |解析 - 差分| < atol + rtol*|差分|
    # 这样对“接近 0 的梯度”用绝对容差兜底（float32 有限差分在这里只能估到 ~0），
    # 对“较大的梯度”用相对容差，避免纯相对误差在 0 附近爆炸。
    names = ["m", "Iz", "a_u", "a_v", "a_r", "X_u", "Y_v", "N_r", "X_uu", "Y_vv", "N_rr"]
    atol, rtol = 1e-3, 5e-2
    abs_err = np.abs(grad_analytic - grad_fd)
    tol = atol + rtol * np.abs(grad_fd)
    ok = abs_err < tol

    print(f"{'参数':<6}{'解析梯度':>14}{'差分梯度':>14}{'绝对误差':>12}  判定")
    for k, nm in enumerate(names):
        print(f"{nm:<6}{grad_analytic[k]:>14.5f}{grad_fd[k]:>14.5f}"
              f"{abs_err[k]:>12.2e}  {'OK' if ok[k] else 'FAIL'}")
    assert np.all(ok), (
        f"梯度校验失败的参数: {[names[k] for k in range(len(ok)) if not ok[k]]}")


if __name__ == "__main__":
    test_gradient_matches_finite_difference()
    print("OK —— 梯度校验通过。")
