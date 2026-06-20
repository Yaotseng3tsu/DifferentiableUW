"""3-DOF（平面）Fossen 动力学 —— NVIDIA Warp 实现。

运动方程（体坐标系）：
    eta_dot = J(psi) * nu
    M * nu_dot + C(nu) * nu + D(nu) * nu = tau + tau_dist

状态：
    eta = (x, y, psi)   —— 惯性系位姿
    nu  = (u, v, r)     —— 体坐标系速度（纵荡/横荡/艏摇角速度）

简化假设（Phase 2 起步版本）：
    * 质心在原点 (x_g = 0)，因此 M 为对角阵，无横荡-艏摇耦合项；
      x_g 耦合留待后续按需加入（见 README §10）。
    * 水平面内恢复力 g(eta) = 0。

所有计算都写成 Warp kernel / func，从而整段 rollout 可被 wp.Tape 记录、可微分。
"""

import warp as wp

from src import params as P

# 把 params.py 里的 Python 整数下标提升为 Warp 编译期常量，便于在 kernel 中索引 theta。
I_M = wp.constant(P.I_M)
I_IZ = wp.constant(P.I_IZ)
I_AU = wp.constant(P.I_AU)
I_AV = wp.constant(P.I_AV)
I_AR = wp.constant(P.I_AR)
I_XU = wp.constant(P.I_XU)
I_YV = wp.constant(P.I_YV)
I_NR = wp.constant(P.I_NR)
I_XUU = wp.constant(P.I_XUU)
I_YVV = wp.constant(P.I_YVV)
I_NRR = wp.constant(P.I_NRR)


@wp.func
def nu_dot_3dof(
    nu: wp.vec3,
    tau: wp.vec3,
    dist: wp.vec3,
    theta: wp.array(dtype=wp.float32),
):
    """给定当前体速度、控制力、扰动力与参数，返回加速度 nu_dot = M^{-1}(...)。"""
    u = nu[0]
    v = nu[1]
    r = nu[2]

    # 广义质量（含附加质量），对角阵。
    m11 = theta[I_M] + theta[I_AU]
    m22 = theta[I_M] + theta[I_AV]
    m33 = theta[I_IZ] + theta[I_AR]

    # 科氏 / 向心力 C(nu)*nu （3-DOF，对角 M 情形）。
    c0 = -m22 * v * r
    c1 = m11 * u * r
    c2 = (m22 - m11) * u * v

    # 阻尼力 D(nu)*nu = (线性 + 二次)*速度。
    d0 = (theta[I_XU] + theta[I_XUU] * wp.abs(u)) * u
    d1 = (theta[I_YV] + theta[I_YVV] * wp.abs(v)) * v
    d2 = (theta[I_NR] + theta[I_NRR] * wp.abs(r)) * r

    a0 = (tau[0] + dist[0] - c0 - d0) / m11
    a1 = (tau[1] + dist[1] - c1 - d1) / m22
    a2 = (tau[2] + dist[2] - c2 - d2) / m33
    return wp.vec3(a0, a1, a2)


@wp.func
def eta_dot_3dof(psi: wp.float32, nu: wp.vec3):
    """运动学：把体坐标系速度旋转到惯性系。eta_dot = J(psi) * nu。"""
    c = wp.cos(psi)
    s = wp.sin(psi)
    return wp.vec3(c * nu[0] - s * nu[1], s * nu[0] + c * nu[1], nu[2])


@wp.kernel
def step_semi_implicit_euler(
    eta_in: wp.array(dtype=wp.vec3),
    nu_in: wp.array(dtype=wp.vec3),
    tau: wp.array(dtype=wp.vec3),
    dist: wp.array(dtype=wp.vec3),
    theta: wp.array(dtype=wp.float32),
    dt: wp.float32,
    eta_out: wp.array(dtype=wp.vec3),
    nu_out: wp.array(dtype=wp.vec3),
):
    """半隐式（辛）欧拉单步：先更新速度，再用新速度更新位姿。

    数组的第 0 维是 batch（可同时仿真多个机器人/多组参数）。
    输入与输出用不同数组（不就地覆盖），以便 wp.Tape 正确记录梯度。
    """
    i = wp.tid()
    a = nu_dot_3dof(nu_in[i], tau[i], dist[i], theta)
    nu_new = nu_in[i] + dt * a
    psi = eta_in[i][2]
    eta_new = eta_in[i] + dt * eta_dot_3dof(psi, nu_new)
    nu_out[i] = nu_new
    eta_out[i] = eta_new
