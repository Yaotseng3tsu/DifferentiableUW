"""可辨识参数向量 theta 的布局、默认值与边界（3-DOF）。

设计要点
--------
把所有“可被梯度辨识”的水动力参数打平成一个一维 float 数组 `theta`，
原因是 Warp 的自动微分是对 **wp.array 输入** 求梯度的：只要把 theta 做成
`wp.array(..., requires_grad=True)`，反向传播后 `theta.grad` 里就直接是
dL/dtheta —— 这正是 Phase 4 参数辨识要用的量。

下面这些整数常量是 theta 里各分量的下标（索引），在 kernel 与 Python 两侧共用，
避免“魔法数字”。
"""

import numpy as np

# ---- theta 各分量的下标 ----------------------------------------------------
I_M = 0  # 质量 m
I_IZ = 1  # 绕 z 轴转动惯量 I_z
I_AU = 2  # 纵荡附加质量 a_u   (= -X_udot, 取正值)
I_AV = 3  # 横荡附加质量 a_v   (= -Y_vdot)
I_AR = 4  # 艏摇附加惯量 a_r   (= -N_rdot)
I_XU = 5  # 纵荡线性阻尼 X_u   (取正，阻力方向)
I_YV = 6  # 横荡线性阻尼 Y_v
I_NR = 7  # 艏摇线性阻尼 N_r
I_XUU = 8  # 纵荡二次阻尼 X_|u|u
I_YVV = 9  # 横荡二次阻尼 Y_|v|v
I_NRR = 10  # 艏摇二次阻尼 N_|r|r

N_PARAMS = 11  # theta 的长度


def default_theta() -> np.ndarray:
    """返回一组物理上合理的默认参数（量级参考 BlueROV2 级小型 ROV）。

    这些只是占位的初值/真值，正式数值在 Phase 2 标定时再确定。
    """
    theta = np.zeros(N_PARAMS, dtype=np.float32)
    theta[I_M] = 11.5  # kg
    theta[I_IZ] = 0.16  # kg·m^2
    theta[I_AU] = 5.5  # kg
    theta[I_AV] = 12.7  # kg（横向附加质量通常更大）
    theta[I_AR] = 0.12  # kg·m^2
    theta[I_XU] = 4.0  # N·s/m
    theta[I_YV] = 6.0
    theta[I_NR] = 0.07  # N·m·s
    theta[I_XUU] = 18.0  # N·s^2/m^2
    theta[I_YVV] = 21.0
    theta[I_NRR] = 1.5
    return theta


# 参数辨识时的边界（下界, 上界）—— 给优化器做投影/裁剪用，全部为非负物理量。
PARAM_BOUNDS = {
    I_M: (1.0, 100.0),
    I_IZ: (0.01, 10.0),
    I_AU: (0.0, 50.0),
    I_AV: (0.0, 100.0),
    I_AR: (0.0, 10.0),
    I_XU: (0.0, 100.0),
    I_YV: (0.0, 100.0),
    I_NR: (0.0, 50.0),
    I_XUU: (0.0, 500.0),
    I_YVV: (0.0, 500.0),
    I_NRR: (0.0, 100.0),
}
