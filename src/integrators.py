"""可微分时间积分器。

step_semi_implicit_euler 在 dynamics/fossen3dof.py（1 阶，开发默认）。
本文件实现 RK4（4 阶），用于精度对比与高精度仿真。

RK4 思路：把状态 y=(eta, nu) 看成一个整体，导数 F(y)=(eta_dot, nu_dot)。
一步内做 4 次求值再加权：
    k1 = F(y)
    k2 = F(y + dt/2 * k1)
    k3 = F(y + dt/2 * k2)
    k4 = F(y + dt   * k3)
    y_new = y + dt/6 * (k1 + 2 k2 + 2 k3 + k4)
控制力 tau / 扰动 dist 在一步内视为常数（标准做法）。

注意：kernel 全程用 Warp 写，因此 RK4 同样可被 wp.Tape 记录、可微分。
"""

import warp as wp

# 复用 fossen3dof 里的两个 @wp.func（导数计算），保证 RK4 与欧拉用同一套物理。
from src.dynamics.fossen3dof import eta_dot_3dof, nu_dot_3dof


@wp.kernel
def step_rk4(
    eta_in: wp.array(dtype=wp.vec3),
    nu_in: wp.array(dtype=wp.vec3),
    tau: wp.array(dtype=wp.vec3),
    dist: wp.array(dtype=wp.vec3),
    theta: wp.array(dtype=wp.float32),
    dt: wp.float32,
    eta_out: wp.array(dtype=wp.vec3),
    nu_out: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    eta0 = eta_in[i]
    nu0 = nu_in[i]
    t = tau[i]
    d = dist[i]

    # --- k1：在起点求导 ---
    k1_nu = nu_dot_3dof(nu0, t, d, theta)
    k1_eta = eta_dot_3dof(eta0[2], nu0)

    # --- k2：在半步(用 k1)处求导 ---
    nu_a = nu0 + 0.5 * dt * k1_nu
    eta_a = eta0 + 0.5 * dt * k1_eta
    k2_nu = nu_dot_3dof(nu_a, t, d, theta)
    k2_eta = eta_dot_3dof(eta_a[2], nu_a)

    # --- k3：在半步(用 k2)处求导 ---
    nu_b = nu0 + 0.5 * dt * k2_nu
    eta_b = eta0 + 0.5 * dt * k2_eta
    k3_nu = nu_dot_3dof(nu_b, t, d, theta)
    k3_eta = eta_dot_3dof(eta_b[2], nu_b)

    # --- k4：在整步(用 k3)处求导 ---
    nu_c = nu0 + dt * k3_nu
    eta_c = eta0 + dt * k3_eta
    k4_nu = nu_dot_3dof(nu_c, t, d, theta)
    k4_eta = eta_dot_3dof(eta_c[2], nu_c)

    # --- 加权平均 ---
    nu_out[i] = nu0 + (dt / 6.0) * (k1_nu + 2.0 * k2_nu + 2.0 * k3_nu + k4_nu)
    eta_out[i] = eta0 + (dt / 6.0) * (k1_eta + 2.0 * k2_eta + 2.0 * k3_eta + k4_eta)
