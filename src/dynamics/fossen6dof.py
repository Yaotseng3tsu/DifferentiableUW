"""6-DOF 完全耦合 Fossen 动力学 —— Phase 5 占位骨架。

待 3-DOF 的可微分 pipeline（Phase 3）与逆问题（Phase 4）验证通过后再实现。

设计预案：
    状态 eta = (x, y, z, phi, theta, psi)，nu = (u, v, w, p, q, r)
    M, C(nu), D(nu) 为 6x6；恢复力 g(eta) 来自重力/浮力之差。
    姿态优先用四元数（wp.quat）以避免欧拉角奇异性（见 README §10）。
    复用 diff_rollout 的 rollout/loss 框架，仅替换 step kernel 与状态维度。
"""

# TODO(Phase 5): 实现 6-DOF step kernel 与 6x6 矩阵装配。
