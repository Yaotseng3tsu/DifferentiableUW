"""6-DOF 可辨识参数向量 theta6 的布局、默认值与标称量级。

简化假设（Phase 5 起步版本，和 3-DOF 起步一致）：
    * 质心在原点 r_g=0，附加质量对角 => 广义质量矩阵 M 为块对角（线/角解耦）。
    * 浮心在质心正上方 (0,0,-zb)（NED: z 向下，-z 为上）。
全耦合（r_g≠0、非对角附加质量）留作后续精度升级。

坐标约定：NED（North-East-Down），z 向下；姿态用四元数(body->inertial)。
"""

import numpy as np

# ---- theta6 各分量下标 ------------------------------------------------------
I_M = 0        # 质量 m
I_IXX = 1      # 转动惯量
I_IYY = 2
I_IZZ = 3
I_XUD = 4      # 附加质量 (surge)  = -X_udot，取正
I_YVD = 5      # (sway)
I_ZWD = 6      # (heave)
I_KPD = 7      # 附加惯量 (roll)
I_MQD = 8      # (pitch)
I_NRD = 9      # (yaw)
I_XU = 10      # 线性阻尼
I_YV = 11
I_ZW = 12
I_KP = 13
I_MQ = 14
I_NR = 15
I_XUU = 16     # 二次阻尼
I_YVV = 17
I_ZWW = 18
I_KPP = 19
I_MQQ = 20
I_NRR = 21
I_W = 22       # 重量 W = m g
I_B = 23       # 浮力 B
I_ZB = 24      # 浮心相对质心的垂直距离 zb (>0: 浮心在上，稳定)

N_PARAMS6 = 25

G = 9.81

# 辨识时的 9+... 分组：可直接测量的固定，难测的辨识
FIXED_IDX6 = [I_M, I_IXX, I_IYY, I_IZZ, I_W, I_B, I_ZB]       # 测量/标定
IDENTIFY_IDX6 = [I_XUD, I_YVD, I_ZWD, I_KPD, I_MQD, I_NRD,     # 附加质量 6
                 I_XU, I_YV, I_ZW, I_KP, I_MQ, I_NR,           # 线性阻尼 6
                 I_XUU, I_YVV, I_ZWW, I_KPP, I_MQQ, I_NRR]     # 二次阻尼 6 (共18)


def default_theta6() -> np.ndarray:
    """一组 BlueROV2 量级的默认参数（占位真值）。"""
    t = np.zeros(N_PARAMS6, dtype=np.float32)
    t[I_M] = 11.5
    t[I_IXX] = 0.16; t[I_IYY] = 0.16; t[I_IZZ] = 0.16
    t[I_XUD] = 5.5;  t[I_YVD] = 12.7; t[I_ZWD] = 14.6
    t[I_KPD] = 0.12; t[I_MQD] = 0.12; t[I_NRD] = 0.12
    t[I_XU] = 4.0;   t[I_YV] = 6.0;   t[I_ZW] = 5.0
    t[I_KP] = 0.07;  t[I_MQ] = 0.07;  t[I_NR] = 0.07
    t[I_XUU] = 18.0; t[I_YVV] = 21.0; t[I_ZWW] = 36.0
    t[I_KPP] = 1.5;  t[I_MQQ] = 1.5;  t[I_NRR] = 1.5
    t[I_W] = t[I_M] * G           # 112.8 N
    t[I_B] = t[I_M] * G           # 中性浮力（默认），正浮力时略大于 W
    t[I_ZB] = 0.02                # 浮心在质心上方 2 cm -> 有扶正力矩
    return t


# 无量纲化标称量级（用于辨识，见 B1）；取整的先验数量级。
SCALE6 = np.array([
    10.0, 0.2, 0.2, 0.2,                 # m, Ixx, Iyy, Izz
    5.0, 10.0, 10.0, 0.1, 0.1, 0.1,      # 附加质量
    5.0, 5.0, 5.0, 0.1, 0.1, 0.1,        # 线性阻尼
    20.0, 20.0, 30.0, 2.0, 2.0, 2.0,     # 二次阻尼
    100.0, 100.0, 0.05,                  # W, B, zb
], dtype=np.float32)
