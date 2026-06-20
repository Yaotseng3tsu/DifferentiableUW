"""生成「Warp 原理」示意图，保存到 pic/warp_principle.png。

两部分：
  ① How Warp runs: Python kernel -> JIT 生成 CUDA C++ -> 编译 -> GPU 并行执行
  ② Why differentiable: 前向(参数->轨迹->损失) + wp.Tape 反向(梯度) -> Adam 更新
     => 用于辨识水动力参数 / 估计水流扰动
图中文字全英文。

运行：.venv\\Scripts\\python.exe notebooks\\make_warp_diagram.py
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pic")
os.makedirs(PIC_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(12.5, 7.6))
ax.set_xlim(0, 12.5)
ax.set_ylim(0, 8)
ax.axis("off")


def box(cx, cy, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.08", fc=fc, ec="black", lw=1.3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=10.5)


def arrow(x1, y1, x2, y2, color="black", rad=0.0, lw=2.2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}", mutation_scale=20))


# ===== 标题 =====
ax.text(6.25, 7.7, "How NVIDIA Warp works", ha="center", fontsize=16, fontweight="bold")

# ===== ① 运行流水线 =====
ax.text(0.2, 6.95, "(1) How Warp runs:  write in Python  ->  compile  ->  execute on GPU",
        ha="left", fontsize=12, fontweight="bold", color="tab:blue")
topY = 6.0
bw, bh = 2.55, 1.0
cxs = [1.55, 4.55, 7.55, 10.7]
texts = ["Python kernel\n@wp.kernel",
         "Warp JIT:\ngenerate CUDA C++",
         "Compile to\nGPU machine code",
         "Parallel run on GPU\n(1000s of threads)"]
for cx, t in zip(cxs, texts):
    box(cx, topY, bw, bh, t, "#dbe7f6")
for i in range(3):
    arrow(cxs[i] + bw / 2, topY, cxs[i + 1] - bw / 2, topY, color="tab:blue")

# ===== 分隔 =====
ax.plot([0.2, 12.3], [5.15, 5.15], color="lightgray", lw=1, ls="--")

# ===== ② 可微闭环 =====
ax.text(0.2, 4.75, "(2) Why differentiable:  gradients turn simulation into an inverse tool",
        ha="left", fontsize=12, fontweight="bold", color="tab:red")
midY = 3.6
cxs2 = [1.7, 4.85, 7.85, 10.6]
box(cxs2[0], midY, 2.7, 1.05, "Parameters θ\n& disturbance U", "#fff2cc")
box(cxs2[1], midY, 2.9, 1.05, "Differentiable rollout\n(Fossen dynamics)", "#d9ead3")
box(cxs2[2], midY, 2.4, 1.05, "Simulated\ntrajectory", "#d9ead3")
box(cxs2[3], midY, 2.7, 1.05, "Loss L\n||sim - observed||²", "#fce5cd")

# 前向箭头（蓝）
arrow(cxs2[0] + 1.35, midY, cxs2[1] - 1.45, midY, color="tab:blue")
arrow(cxs2[1] + 1.45, midY, cxs2[2] - 1.2, midY, color="tab:blue")
arrow(cxs2[2] + 1.2, midY, cxs2[3] - 1.35, midY, color="tab:blue")
ax.text(6.25, midY + 0.75, "forward pass", ha="center", fontsize=9.5, color="tab:blue", style="italic")

# 反向箭头（红，向下绕回）
arrow(cxs2[3], midY - 0.55, cxs2[0], midY - 0.55, color="tab:red", rad=-0.32, lw=2.6)
ax.text(6.25, 1.45,
        "wp.Tape backward:  gradients  ∂L/∂θ , ∂L/∂U   →   Adam updates θ, U",
        ha="center", fontsize=10.5, color="tab:red", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="tab:red", lw=1))

# ===== 结论条 =====
ax.add_patch(FancyBboxPatch((1.0, 0.35), 10.5, 0.7, boxstyle="round,pad=0.08",
                            fc="#efe1ff", ec="black", lw=1.2))
ax.text(6.25, 0.7,
        "=>  identify hydrodynamic parameters   &   estimate current disturbance",
        ha="center", fontsize=11.5, fontweight="bold")

fig.tight_layout()
out = os.path.join(PIC_DIR, "warp_principle.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved:", out)
