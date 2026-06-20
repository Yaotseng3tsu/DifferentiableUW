"""Phase 1 —— Warp 基础冒烟测试。

目标：确认 Warp 能看到 GPU，并跑通最基本的 kernel 编程模型
（@wp.kernel / wp.array / wp.launch）。

运行：
    python notebooks/00_warp_basics.py
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台中文输出不报错

import numpy as np
import warp as wp

wp.init()
print("可用设备：", wp.get_devices())          # 期望看到 cuda:0
device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
print("使用设备：", device)


@wp.kernel
def saxpy(a: wp.float32, x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32)):
    """经典 SAXPY：y[i] = a * x[i] + y[i]。"""
    i = wp.tid()
    y[i] = a * x[i] + y[i]


n = 8
x = wp.array(np.arange(n, dtype=np.float32), device=device)
y = wp.zeros(n, dtype=wp.float32, device=device)

wp.launch(saxpy, dim=n, inputs=[2.0, x, y], device=device)
wp.synchronize()

print("结果 y =", y.numpy())                   # 期望 [0,2,4,6,8,10,12,14]
print("OK —— Warp kernel 运行成功。")
