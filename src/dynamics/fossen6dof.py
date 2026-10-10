"""6-DOF 完全三维 Fossen 动力学 —— NVIDIA Warp 实现（Phase 5）。

状态：
    p = (x,y,z)   位置（惯性/NED 系）
    q             姿态四元数 (body->inertial)，wp.quat (x,y,z,w)
    v = (u,v,w)   体坐标线速度
    w = (p,q,r)   体坐标角速度

运动方程（块对角 M，r_g=0）：
    M_lin a_lin = f_ctrl + f_dist + f_restore - f_coriolis - f_damp
    M_ang a_ang = m_ctrl + m_dist + m_restore - m_coriolis - m_damp
    ṗ = R(q) v ,   q̇ = ½ q ⊗ (0, w) ，每步归一化 q。

坐标约定：NED，z 向下；重力 +z，浮力 -z。
所有计算写成 Warp kernel，整段 rollout 可被 wp.Tape 记录、可微。
"""

import warp as wp

from src import params6 as P6

# 把下标提升为 Warp 编译期常量
I_M = wp.constant(P6.I_M)
I_IXX = wp.constant(P6.I_IXX); I_IYY = wp.constant(P6.I_IYY); I_IZZ = wp.constant(P6.I_IZZ)
I_XUD = wp.constant(P6.I_XUD); I_YVD = wp.constant(P6.I_YVD); I_ZWD = wp.constant(P6.I_ZWD)
I_KPD = wp.constant(P6.I_KPD); I_MQD = wp.constant(P6.I_MQD); I_NRD = wp.constant(P6.I_NRD)
I_XU = wp.constant(P6.I_XU); I_YV = wp.constant(P6.I_YV); I_ZW = wp.constant(P6.I_ZW)
I_KP = wp.constant(P6.I_KP); I_MQ = wp.constant(P6.I_MQ); I_NR = wp.constant(P6.I_NR)
I_XUU = wp.constant(P6.I_XUU); I_YVV = wp.constant(P6.I_YVV); I_ZWW = wp.constant(P6.I_ZWW)
I_KPP = wp.constant(P6.I_KPP); I_MQQ = wp.constant(P6.I_MQQ); I_NRR = wp.constant(P6.I_NRR)
I_W = wp.constant(P6.I_W); I_B = wp.constant(P6.I_B); I_ZB = wp.constant(P6.I_ZB)


@wp.func
def restoring_6dof(q: wp.quat, theta: wp.array(dtype=wp.float32)):
    """重力+浮力的恢复力/力矩（体坐标）。返回 (f_res, m_res) 打包的 spatial_vector。

    NED: 重力 (0,0,W) 作用在质心(原点)；浮力 (0,0,-B) 作用在浮心 (0,0,-zb)。
    """
    W = theta[I_W]
    B = theta[I_B]
    zb = theta[I_ZB]
    fg_b = wp.quat_rotate_inv(q, wp.vec3(0.0, 0.0, W))    # 重力在体坐标
    fb_b = wp.quat_rotate_inv(q, wp.vec3(0.0, 0.0, -B))   # 浮力在体坐标
    f_res = fg_b + fb_b
    r_b = wp.vec3(0.0, 0.0, -zb)                              # 浮心位置
    m_res = wp.cross(r_b, fb_b)                               # 重力在原点无力矩
    return wp.spatial_vector(f_res[0], f_res[1], f_res[2], m_res[0], m_res[1], m_res[2])


@wp.kernel
def step_semi_implicit_euler_6dof(
    p_in: wp.array(dtype=wp.vec3),
    q_in: wp.array(dtype=wp.quat),
    v_in: wp.array(dtype=wp.vec3),
    w_in: wp.array(dtype=wp.vec3),
    tau_f: wp.array(dtype=wp.vec3),
    tau_m: wp.array(dtype=wp.vec3),
    dist_f: wp.array(dtype=wp.vec3),
    dist_m: wp.array(dtype=wp.vec3),
    theta: wp.array(dtype=wp.float32),
    dt: wp.float32,
    p_out: wp.array(dtype=wp.vec3),
    q_out: wp.array(dtype=wp.quat),
    v_out: wp.array(dtype=wp.vec3),
    w_out: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    v = v_in[i]
    w = w_in[i]
    q = q_in[i]

    # 广义质量（含附加质量），块对角
    m11 = theta[I_M] + theta[I_XUD]
    m22 = theta[I_M] + theta[I_YVD]
    m33 = theta[I_M] + theta[I_ZWD]
    i11 = theta[I_IXX] + theta[I_KPD]
    i22 = theta[I_IYY] + theta[I_MQD]
    i33 = theta[I_IZZ] + theta[I_NRD]

    Mv = wp.vec3(m11 * v[0], m22 * v[1], m33 * v[2])
    Iw = wp.vec3(i11 * w[0], i22 * w[1], i33 * w[2])

    # 科氏/向心力（6-DOF，块对角 M）
    f_cor = wp.cross(w, Mv)
    m_cor = wp.cross(v, Mv) + wp.cross(w, Iw)

    # 阻尼（对角，线性+二次）
    f_damp = wp.vec3((theta[I_XU] + theta[I_XUU] * wp.abs(v[0])) * v[0],
                     (theta[I_YV] + theta[I_YVV] * wp.abs(v[1])) * v[1],
                     (theta[I_ZW] + theta[I_ZWW] * wp.abs(v[2])) * v[2])
    m_damp = wp.vec3((theta[I_KP] + theta[I_KPP] * wp.abs(w[0])) * w[0],
                     (theta[I_MQ] + theta[I_MQQ] * wp.abs(w[1])) * w[1],
                     (theta[I_NR] + theta[I_NRR] * wp.abs(w[2])) * w[2])

    # 恢复力
    g = restoring_6dof(q, theta)
    f_res = wp.vec3(g[0], g[1], g[2])
    m_res = wp.vec3(g[3], g[4], g[5])

    tf = tau_f[i] + dist_f[i]
    tm = tau_m[i] + dist_m[i]

    a_lin = wp.vec3((tf[0] + f_res[0] - f_cor[0] - f_damp[0]) / m11,
                    (tf[1] + f_res[1] - f_cor[1] - f_damp[1]) / m22,
                    (tf[2] + f_res[2] - f_cor[2] - f_damp[2]) / m33)
    a_ang = wp.vec3((tm[0] + m_res[0] - m_cor[0] - m_damp[0]) / i11,
                    (tm[1] + m_res[1] - m_cor[1] - m_damp[1]) / i22,
                    (tm[2] + m_res[2] - m_cor[2] - m_damp[2]) / i33)

    # 半隐式欧拉：先更新速度
    v_new = v + dt * a_lin
    w_new = w + dt * a_ang

    # 再用新速度更新位姿
    p_out[i] = p_in[i] + dt * wp.quat_rotate(q, v_new)          # ṗ = R(q) v
    wq = wp.quat(w_new[0], w_new[1], w_new[2], 0.0)
    q_dot = wp.mul(q, wq) * 0.5                                  # q̇ = ½ q ⊗ (0,w)
    q_out[i] = wp.normalize(q + q_dot * dt)                      # 归一化保持单位四元数
    v_out[i] = v_new
    w_out[i] = w_new
