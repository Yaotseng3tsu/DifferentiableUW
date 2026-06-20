"""Phase 1.2 —— 向量与矩阵类型：wp.vec3 / wp.mat33。

学习目标：会用 Warp 的向量/矩阵，因为机器人状态 (u,v,r) 就是一个 vec3，
而坐标系旋转 J(psi) 就是一个 mat33。这一步直接对应 fossen3dof.py 里的写法。

运行：.venv\\Scripts\\python.exe notebooks\\p1_02_vec_mat.py
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import warp as wp

wp.init()
device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


# ---- 演示 A：向量运算（点积 / 叉积 / 模长） --------------------------------
@wp.kernel
def vec_ops(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3),
            dotp: wp.array(dtype=wp.float32),
            crossp: wp.array(dtype=wp.vec3),
            norm_a: wp.array(dtype=wp.float32)):
    i = wp.tid()
    dotp[i] = wp.dot(a[i], b[i])      # 点积（标量）
    crossp[i] = wp.cross(a[i], b[i])  # 叉积（向量）
    norm_a[i] = wp.length(a[i])       # 模长


a = wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
b = wp.array([wp.vec3(0.0, 1.0, 0.0)], dtype=wp.vec3, device=device)
dotp = wp.zeros(1, dtype=wp.float32, device=device)
crossp = wp.zeros(1, dtype=wp.vec3, device=device)
norm_a = wp.zeros(1, dtype=wp.float32, device=device)
wp.launch(vec_ops, dim=1, inputs=[a, b], outputs=[dotp, crossp, norm_a], device=device)
wp.synchronize()
print("[向量] a·b =", dotp.numpy()[0], " a×b =", crossp.numpy()[0],
      " |a| =", norm_a.numpy()[0])    # 0.0  [0,0,1]  1.0


# ---- 演示 B：用 mat33 做平面旋转（这就是 3-DOF 的 J(psi)）------------------
@wp.kernel
def rotate2d(psi: wp.float32, v: wp.array(dtype=wp.vec3),
             out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    c = wp.cos(psi)
    s = wp.sin(psi)
    # 绕 z 轴的旋转矩阵（按行填 9 个元素）
    R = wp.mat33(c, -s, 0.0,
                 s,  c, 0.0,
                 0.0, 0.0, 1.0)
    out[i] = R * v[i]                 # 矩阵 * 向量


v = wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
out = wp.zeros(1, dtype=wp.vec3, device=device)
wp.launch(rotate2d, dim=1, inputs=[float(np.pi / 2.0), v], outputs=[out], device=device)
wp.synchronize()
print("[旋转] 把 (1,0,0) 转 90° =", np.round(out.numpy()[0], 6))  # ≈ (0,1,0)
print("[要点] 机器人速度 (u,v,r) 用 vec3；坐标系变换 J(psi) 用 mat33，与 fossen3dof.py 一致。")
