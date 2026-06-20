"""Phase 1.1 —— Warp 三件套：kernel / array / launch / tid。

学习目标：理解 GPU 并行计算的最基本骨架。
    * wp.array   —— GPU 上的数组（类似 GPU 版 numpy）
    * @wp.kernel —— 在 GPU 上并行运行的函数
    * wp.launch  —— 启动 kernel，开 N 个线程
    * wp.tid()   —— 当前线程编号，用来索引数组的「第 i 个元素」

运行：
    .venv\\Scripts\\python.exe notebooks\\p1_01_kernel_basics.py
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import warp as wp

wp.init()
device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
print("[设备]", device)


# ---- 演示 A：每个线程把「自己的编号」写进数组，直观看到并行 ----------------
@wp.kernel
def write_thread_id(out: wp.array(dtype=wp.int32)):
    i = wp.tid()        # 第 i 个线程
    out[i] = i          # 把编号写到第 i 个格子


n = 8
ids = wp.zeros(n, dtype=wp.int32, device=device)
wp.launch(write_thread_id, dim=n, inputs=[ids], device=device)  # 开 n 个线程
wp.synchronize()        # 等 GPU 算完
print("[演示A] 每个线程写自己的编号 =", ids.numpy())   # [0 1 2 3 4 5 6 7]


# ---- 演示 B：逐元素计算 out[i] = a*x[i] + b ---------------------------------
@wp.kernel
def affine(a: wp.float32, x: wp.array(dtype=wp.float32),
           b: wp.float32, out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = a * x[i] + b


x = wp.array(np.array([0, 1, 2, 3, 4], dtype=np.float32), device=device)
out = wp.zeros(5, dtype=wp.float32, device=device)
wp.launch(affine, dim=5, inputs=[2.0, x, 10.0], outputs=[out], device=device)
wp.synchronize()
print("[演示B] 2*x + 10 =", out.numpy())               # [10 12 14 16 18]

# numpy <-> warp 的桥梁：
#   wp.array(np_array)  把 CPU 数据搬到 GPU
#   gpu_array.numpy()   把 GPU 结果搬回 CPU 看
print("[要点] dim=N 开 N 个线程；每个线程用 wp.tid() 拿到自己的 i，各算各的格子。")
