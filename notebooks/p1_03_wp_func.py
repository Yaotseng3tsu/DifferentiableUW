"""Phase 1.3 —— @wp.func 子函数：把可复用的计算抽出来。

学习目标：理解 @wp.func 和 @wp.kernel 的区别与配合。
    * @wp.kernel：能被 wp.launch 启动，是 GPU 任务的入口。
    * @wp.func  ：不能单独 launch，只能被 kernel（或其它 func）调用，会返回值。
fossen3dof.py 里的 nu_dot_3dof / eta_dot_3dof 就是 func，被 step kernel 调用。

运行：.venv\\Scripts\\python.exe notebooks\\p1_03_wp_func.py
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import warp as wp

wp.init()
device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


# ---- 一个可复用的子函数：动能 0.5*m*|v|^2 ----------------------------------
@wp.func
def kinetic_energy(m: wp.float32, vel: wp.vec3):
    return 0.5 * m * wp.dot(vel, vel)   # func 有返回值


# ---- kernel 调用上面的 func，对每个粒子算动能 -----------------------------
@wp.kernel
def energy_kernel(m: wp.float32, vels: wp.array(dtype=wp.vec3),
                  out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = kinetic_energy(m, vels[i])  # 复用：每个线程都调用同一个 func


vels = wp.array([wp.vec3(3.0, 4.0, 0.0),    # |v|=5 -> KE=0.5*2*25=25
                 wp.vec3(1.0, 0.0, 0.0),    # |v|=1 -> KE=0.5*2*1 =1
                 wp.vec3(0.0, 0.0, 2.0)],   # |v|=2 -> KE=0.5*2*4 =4
                dtype=wp.vec3, device=device)
out = wp.zeros(3, dtype=wp.float32, device=device)
wp.launch(energy_kernel, dim=3, inputs=[2.0, vels], outputs=[out], device=device)
wp.synchronize()
print("[动能] 三个粒子 (m=2) =", out.numpy())   # [25. 1. 4.]
print("[要点] func 写一次、到处复用；kernel 才是入口。物理代码就靠 func 把方程拆成可读的块。")
