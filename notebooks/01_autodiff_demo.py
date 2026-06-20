"""Phase 1 —— Warp 自动微分（wp.Tape）端到端示例 + 有限差分校验。

这是整个项目最关键的一课：理解 Warp 如何对 **wp.array 输入** 求梯度。
模式：
    1) 把要求梯度的输入设 requires_grad=True
    2) 在 `with tape:` 内做前向计算，把标量损失写入一个长度 1 的数组
    3) tape.backward(loss=loss_arr)
    4) 从 input.grad 读取梯度，并与有限差分对比

运行：
    python notebooks/01_autodiff_demo.py
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台中文输出不报错

import numpy as np
import warp as wp

wp.init()
device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


@wp.kernel
def sum_of_squares(x: wp.array(dtype=wp.float32), loss: wp.array(dtype=wp.float32)):
    """loss[0] = sum_i x[i]^2，解析梯度 d loss / d x[i] = 2 x[i]。"""
    i = wp.tid()
    wp.atomic_add(loss, 0, x[i] * x[i])


n = 5
x_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
x = wp.array(x_np, dtype=wp.float32, device=device, requires_grad=True)
loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

tape = wp.Tape()
with tape:
    wp.launch(sum_of_squares, dim=n, inputs=[x, loss], device=device)

tape.backward(loss=loss)

grad_analytic = x.grad.numpy()
grad_finite_diff = 2.0 * x_np            # 理论值

print("loss            =", loss.numpy()[0])      # 期望 55
print("解析梯度 (Warp)  =", grad_analytic)         # 期望 [2,4,6,8,10]
print("理论梯度 2*x     =", grad_finite_diff)
err = np.max(np.abs(grad_analytic - grad_finite_diff))
print("最大误差         =", err)
assert err < 1e-4, "梯度校验失败！"
print("OK —— 自动微分与理论梯度一致。")
