"""生成「欧拉 vs RK4 如何采样斜率」的示意图，保存到 pic/ 文件夹。

用放大差异的例子：dy/dt = y, y0=1, 真解 y=e^t；步长 h=1（故意取大）。
    Euler 只用起点斜率 k1 -> 估计 2.000（明显偏低，真值 2.718）
    RK4 采样 4 个斜率加权平均 -> 估计 2.708（几乎落在真解上）
图中所有文字均为英文。

运行：.venv\\Scripts\\python.exe notebooks\\make_integrator_diagram.py
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")          # 不弹窗，直接存文件
import matplotlib.pyplot as plt

PIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pic")
os.makedirs(PIC_DIR, exist_ok=True)

# ---- ODE: dy/dt = y, 真解 e^t ----------------------------------------------
h = 1.0
t = np.linspace(0.0, h, 200)
true = np.exp(t)


def f(tt, yy):
    return yy


y0 = 1.0
k1 = f(0.0, y0)                                # 1.00  起点斜率
ym1 = y0 + 0.5 * h * k1; k2 = f(0.5 * h, ym1)  # 1.50  中点(用k1)
ym2 = y0 + 0.5 * h * k2; k3 = f(0.5 * h, ym2)  # 1.75  中点(用k2,修正)
ye = y0 + h * k3;        k4 = f(h, ye)         # 2.75  终点(用k3)
euler_est = y0 + h * k1                         # 2.000
rk4_est = y0 + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)  # 2.708

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True, sharey=True)
for ax in (axL, axR):
    ax.plot(t, true, "k--", lw=1.5, label="true solution  y=e^t")
    ax.plot(0.0, y0, "ko", ms=7)
    ax.set_xlabel("time")
    ax.set_ylabel("state y")
    ax.grid(alpha=0.3)

# ---- 左：欧拉（单斜率）----------------------------------------------------
axL.plot([0.0, h], [y0, euler_est], "-", color="tab:blue", lw=2.5,
         label="Euler step (uses k1 only)")
axL.plot(h, euler_est, "o", color="tab:blue", ms=10,
         label=f"Euler estimate = {euler_est:.3f}")
axL.plot(h, true[-1], "s", color="green", ms=9, label=f"true = {true[-1]:.3f}")
axL.annotate("", xy=(h, true[-1]), xytext=(h, euler_est),
             arrowprops=dict(arrowstyle="<->", color="red", lw=2))
axL.text(h - 0.03, 0.5 * (euler_est + true[-1]), "error  ", color="red",
         ha="right", va="center", fontweight="bold")
axL.annotate(f"slope k1 = {k1:.2f}", xy=(0.14, y0 + 0.14 * k1 + 0.05),
             color="tab:blue", fontsize=10)
axL.set_title("Euler (1st order): a single slope at the start")
axL.legend(loc="upper left", fontsize=9)


# ---- 右：RK4（四斜率加权）-------------------------------------------------
def slope_seg(ax, t0, yp, slope, color, label, half=0.20):
    ax.plot([t0 - half, t0 + half], [yp - half * slope, yp + half * slope],
            "-", color=color, lw=3)
    ax.plot(t0, yp, "o", color=color, ms=7)
    ax.annotate(label, xy=(t0, yp), xytext=(t0 + 0.04, yp - 0.20),
                color=color, fontsize=9)


slope_seg(axR, 0.0, y0, k1, "tab:blue", f"k1={k1:.2f}  @start")
slope_seg(axR, 0.5 * h, ym1, k2, "tab:orange", f"k2={k2:.2f}  @mid")
slope_seg(axR, 0.5 * h, ym2, k3, "tab:purple", f"k3={k3:.2f}  @mid (refined)")
slope_seg(axR, h, ye, k4, "tab:brown", f"k4={k4:.2f}  @end")
axR.plot(h, rk4_est, "*", color="red", ms=18, label=f"RK4 estimate = {rk4_est:.3f}")
axR.plot(h, true[-1], "s", color="green", ms=9, label=f"true = {true[-1]:.3f}")
axR.set_title("RK4 (4th order): weighted average of 4 sampled slopes")
axR.legend(loc="upper left", fontsize=9)

fig.suptitle("How Euler vs RK4 sample the slope over one step   (dy/dt = y, step h = 1)",
             fontsize=13)
fig.text(0.5, 0.01,
         "Note: semi-implicit Euler uses the same single slope, but updates position with the NEWLY "
         "computed velocity (v += dt*a; x += dt*v_new), giving better long-term energy stability.",
         ha="center", fontsize=9, style="italic")
fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))

out = os.path.join(PIC_DIR, "integrators_euler_vs_rk4.png")
fig.savefig(out, dpi=130)
print("saved:", out)
