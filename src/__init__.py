"""DifferentiableUW —— 基于 Fossen 模型的可微分水下机器人动力学（NVIDIA Warp 实现）。

子模块：
    dynamics/   —— Fossen 运动方程（3-DOF / 6-DOF）的 Warp kernel
    integrators —— 可微分的时间积分器（半隐式欧拉、RK4）
    diff_rollout—— 记录在 wp.Tape 上的多步 rollout + 轨迹损失
    params      —— 可辨识参数向量 θ 的布局、默认值与边界
"""
