"""生成「普通梯度下降 vs Adam」优化路径对比图，保存到 pic/。

损失面：L(x,y) = 0.5*(x^2 + B*y^2)，一个强各向异性的"扁长山谷"
（y 方向比 x 方向陡 B 倍，模拟 3.4 里参数量级差异大的病态情形）。最小值在原点。
两个优化器从同一起点出发、跑同样步数：
    * 普通 GD：单一学习率。为了在陡的 y 方向不发散，lr 必须很小，
               于是在缓的 x 方向龟速爬行；同时在 y 方向来回横跳。
    * Adam   ：每维自适应步长 + 动量，较直地奔向谷底。
图中文字全英文。

运行：.venv\\Scripts\\python.exe notebooks\\make_optimizer_diagram.py
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pic")
os.makedirs(PIC_DIR, exist_ok=True)

# ---- 配置 ------------------------------------------------------------------
B = 50.0                       # 各向异性：y 方向曲率是 x 的 50 倍（病态）
START = np.array([9.0, 3.0])
N = 50                         # 两者相同步数
LR_GD = 0.036                  # GD: 受 y 方向稳定性限制(必须 < 2/B=0.04)，只能很小
LR_ADAM = 0.4                  # Adam: 自适应，可用大得多的学习率


def grad(p):
    return np.array([p[0], B * p[1]])


def run_gd(lr, n):
    p = START.copy()
    path = [p.copy()]
    for _ in range(n):
        p = p - lr * grad(p)
        path.append(p.copy())
    return np.array(path)


def run_adam(lr, n, b1=0.9, b2=0.999, eps=1e-8):
    p = START.copy()
    m = np.zeros(2)
    v = np.zeros(2)
    path = [p.copy()]
    for t in range(1, n + 1):
        g = grad(p)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        p = p - lr * mhat / (np.sqrt(vhat) + eps)
        path.append(p.copy())
    return np.array(path)


gd = run_gd(LR_GD, N)
adam = run_adam(LR_ADAM, N)

# ---- 等高线 + 两条路径 -----------------------------------------------------
xs = np.linspace(-10, 10, 400)
ys = np.linspace(-3.5, 3.5, 400)
X, Y = np.meshgrid(xs, ys)
Z = 0.5 * (X ** 2 + B * Y ** 2)

fig, ax = plt.subplots(figsize=(11, 5.6))
ax.contour(X, Y, Z, levels=np.linspace(2, 300, 24), colors="lightgray", linewidths=0.8)

ax.plot(gd[:, 0], gd[:, 1], "-o", color="tab:red", ms=3, lw=1.4,
        label=f"Plain GD (lr={LR_GD}, {N} steps)")
ax.plot(adam[:, 0], adam[:, 1], "-o", color="tab:blue", ms=3, lw=1.4,
        label=f"Adam (lr={LR_ADAM}, {N} steps)")
ax.plot(*START, "ks", ms=11, label="start")
ax.plot(0, 0, "g*", ms=20, label="minimum (target)")

ax.set_title("Plain GD vs Adam on an ill-conditioned (anisotropic) valley\n"
             f"L(x,y) = 0.5 (x^2 + {int(B)} y^2)   — condition number = {int(B)}")
ax.set_xlabel("parameter x  (gentle direction: GD crawls)")
ax.set_ylabel("parameter y  (steep direction: GD zig-zags)")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.2)        # 不用等比例，让图填满、纵向放大看清横跳

gd_d = float(np.hypot(*gd[-1]))
adam_d = float(np.hypot(*adam[-1]))
ax.text(0.0, -3.15,
        f"after {N} steps -> distance to minimum:   GD = {gd_d:.2f}   |   Adam = {adam_d:.3f}",
        ha="center", fontsize=10, style="italic",
        bbox=dict(boxstyle="round", fc="white", ec="gray"))

fig.tight_layout()
out = os.path.join(PIC_DIR, "optimizer_gd_vs_adam.png")
fig.savefig(out, dpi=130)
print("saved:", out)
print(f"GD 终点距谷底 = {gd_d:.3f},  Adam 终点距谷底 = {adam_d:.4f}")
