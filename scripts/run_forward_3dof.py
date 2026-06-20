"""Phase 2 —— 3-DOF 前向仿真演示。

施加一段纵荡推力脉冲，然后让机器人在阻尼作用下滑行，画出轨迹与速度曲线。
预期行为：推力期间加速 -> 撤力后速度因阻尼指数衰减（物理合理性检查）。

运行：
    python scripts/run_forward_3dof.py
"""

import _bootstrap  # noqa: F401  （把仓库根加入 sys.path）

import numpy as np
import warp as wp

from src import params as P
from src.diff_rollout import rollout

wp.init()
device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"

# ---- 仿真设置 --------------------------------------------------------------
dt = 0.02
T = 500                      # 总步数（10 秒）
batch = 1

theta = wp.array(P.default_theta(), dtype=wp.float32, device=device)

eta0 = wp.zeros(batch, dtype=wp.vec3, device=device)   # 从原点静止出发
nu0 = wp.zeros(batch, dtype=wp.vec3, device=device)

# 控制力序列（一个转弯机动，便于看出曲线轨迹与转动耦合）：
#   纵荡推力 fx：前 120 步(2.4s)给 20N，之后撤力滑行
#   偏航力矩 mz：第 40~100 步给 2.0 N·m，让机器人边走边转
tau_seq = []
dist_seq = []
for t in range(T):
    fx = 20.0 if t < 120 else 0.0
    mz = 2.0 if 40 <= t < 100 else 0.0
    tau_seq.append(wp.array(np.array([[fx, 0.0, mz]], dtype=np.float32),
                            dtype=wp.vec3, device=device))
    dist_seq.append(wp.zeros(batch, dtype=wp.vec3, device=device))   # 暂无扰动

etas, nus = rollout(eta0, nu0, tau_seq, dist_seq, theta, dt, T, device)

# ---- 取回轨迹并可视化 ------------------------------------------------------
eta_hist = np.array([e.numpy()[0] for e in etas])    # (T+1, 3): x, y, psi
nu_hist = np.array([v.numpy()[0] for v in nus])      # (T+1, 3): u, v, r
time = np.arange(T + 1) * dt

try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(eta_hist[:, 0], eta_hist[:, 1], "-b")
    ax[0].plot(eta_hist[0, 0], eta_hist[0, 1], "go", label="start")
    ax[0].plot(eta_hist[-1, 0], eta_hist[-1, 1], "rs", label="end")
    ax[0].set_title("Trajectory (x-y)")
    ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("y [m]")
    ax[0].axis("equal"); ax[0].legend()

    ax[1].plot(time, nu_hist[:, 0], label="u (surge)")
    ax[1].plot(time, nu_hist[:, 1], label="v (sway)")
    ax[1].plot(time, nu_hist[:, 2], label="r (yaw rate)")
    ax[1].set_title("Body-frame velocity vs time")
    ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("u,v [m/s]  /  r [rad/s]")
    ax[1].legend()

    fig.tight_layout()
    import os
    _out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "pic", "forward_3dof.png")
    os.makedirs(os.path.dirname(_out), exist_ok=True)
    fig.savefig(_out, dpi=120)
    print("已保存图像", _out)
except ImportError:
    print("未安装 matplotlib，跳过绘图。")

print("最终纵荡速度 u =", nu_hist[-1, 0], "（应已衰减接近 0）")
