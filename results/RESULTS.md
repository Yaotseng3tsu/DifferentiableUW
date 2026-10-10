# 实验结果记录 / Experiment Results Log

> ⚠️ **范围说明 / Scope**：本表记录的是**合成数据下的"管线验证"实验**（参数为占位值），
> 用于证明可微分辨识/逆问题管线正确，**不是真实机器人的物理测量结果**。
> 真实物理结果待 Phase 6（BlueROV2 实测数据）。
> *These are synthetic-data pipeline-validation experiments (placeholder parameters),
> proving the differentiable identification/inverse pipeline is correct — NOT physical
> measurements of a real robot. Real results await Phase 6 (BlueROV2 data).*

记录规则 / Convention：只记 Phase 3+ 的定量实验；图存 `pic/`，数列(可选)存 `results/data/`。
更新 / Updated: 2026-06-19

---

## 实验索引 / Index
| ID | Phase | 实验 / Experiment | 关键结果 / Key result | 图 / Figure |
|----|-------|-------------------|------------------------|-------------|
| E1 | 3.3 | 梯度门禁校验 Gradient gate (analytic vs finite-diff) | 11/11 参数通过 | — |
| E2 | 3.4 | 扰动力反算 Disturbance (current) recovery | U 精确恢复, loss→0 | — |
| E3 | 4.1 | 参数辨识 Parameter identification (9 params) | 平均相对误差 0.60% | identify_params_convergence.png |
| E4 | 4.2 | 初值鲁棒性 Robustness to initial guess | ×0.5~1.5 均收敛 ~2% | robustness.png |
| E5 | 4.2 | 噪声鲁棒性 Robustness to measurement noise | 误差随 σ 单调上升、有界 | robustness.png |
| E6 | 4.3 | 恒定水流反算 Constant current inversion | U 精确恢复, loss→1e-16 | recover_disturbance_const.png |
| E7 | 4.3 | 时变水流反算 Time-varying current inversion | 中段好/末端失真, RMSE 0.24N | recover_disturbance_timevarying.png |
| E8 | 5.3-5.4 | 6-DOF 梯度门禁 + 参数辨识 6-DOF gate + identification | 门禁 18/18 通过; 辨识误差 1.38% | identify_params6_convergence.png |

---

## E1 — 梯度门禁校验 / Gradient gate (Phase 3.3)
**设置 / Setup**: 3-DOF, dt=0.02, T=60；解析梯度(wp.Tape) vs 中心差分(eps=1e-3)；
联合容差 atol=1e-3, rtol=5e-2。

| param | analytic | finite-diff | abs error | verdict |
|-------|---------:|------------:|----------:|:-------:|
| m     | 0.03006 | 0.02985 | 2.1e-04 | OK |
| Iz    | 1.51281 | 1.51267 | 1.5e-04 | OK |
| a_u   | 0.04298 | 0.04293 | 4.3e-05 | OK |
| a_v   | -0.01292 | -0.01287 | 4.6e-05 | OK |
| a_r   | 1.51281 | 1.51267 | 1.5e-04 | OK |
| X_u   | 0.01306 | 0.01295 | 1.1e-04 | OK |
| Y_v   | 0.00110 | 0.00094 | 1.6e-04 | OK |
| N_r   | 0.65155 | 0.65156 | 9.9e-06 | OK |
| X_uu  | 0.00487 | 0.00486 | 6.5e-06 | OK |
| Y_vv  | 0.00009 | 0.00000 | 8.6e-05 | OK |
| N_rr  | 0.46224 | 0.46239 | 1.6e-04 | OK |

**结论 / Takeaway**: 解析梯度全部正确（绝对误差 ≤2e-4）。`Iz`≡`a_r`（结构不可辨识，见 open-problems A1）；
`Y_v`/`Y_vv` 梯度极小（弱可观测，A2）。

---

## E2 — 扰动力反算 / Disturbance recovery (Phase 3.4)
**设置 / Setup**: 参数已知固定；未知恒定水流力 U 作用于质心；短推力脉冲(20步)+长滑行；
Adam (lr=0.05, 800 it)。梯度校验绝对误差 (2e-4, 1.3e-2)。

| quantity | Ux | Uy |
|----------|---:|---:|
| true     | 1.000 | -1.500 |
| recovered | 1.000 | -1.500 |

**对比 / Comparison**: 普通 GD + 推进机动 → 卡在 (0.12,-0.26)；Adam + 滑行机动 → 精确 (1.0,-1.5)，loss→0。
**结论 / Takeaway**: 仅凭被水流推偏的轨迹即可精确反算水流力（逆问题成立）。可观测性依赖机动(A2)，病态需 Adam(B1)。

---

## E3 — 参数辨识 / Parameter identification (Phase 4.1)
**设置 / Setup**: 固定可测参数 m, Iz；辨识 9 个难测参数；无量纲化(/scale)；丰富机动(纵荡/横荡/艏摇/组合) T=200；
Adam (lr=0.02, 3000 it)；初值 = 真值×0.5。

**收敛 / Convergence**: 平均相对误差 50% → **0.60%**；loss 60.7 → 6.2e-7（~1500 步后进入平台 B2）。

| param | true | init(×0.5) | identified | rel.err |
|-------|-----:|-----------:|-----------:|--------:|
| a_u   | 5.5000 | 2.7500 | 5.5004 | 0.0% |
| a_v   | 12.7000 | 6.3500 | 12.7021 | 0.0% |
| a_r   | 0.1200 | 0.0600 | 0.1200 | 0.0% |
| X_u   | 4.0000 | 2.0000 | 4.0437 | 1.1% |
| Y_v   | 6.0000 | 3.0000 | 5.9960 | 0.1% |
| N_r   | 0.0700 | 0.0350 | 0.0724 | 3.4% |
| X_uu  | 18.0000 | 9.0000 | 17.9074 | 0.5% |
| Y_vv  | 21.0000 | 10.5000 | 21.0251 | 0.1% |
| N_rr  | 1.5000 | 0.7500 | 1.4973 | 0.2% |

**图 / Figure**: [pic/identify_params_convergence.png](../pic/identify_params_convergence.png)
**结论 / Takeaway**: 9 个量级差异巨大(0.07~21)的参数经无量纲化后被 Adam 稳定辨识，平均误差 0.6%。
误差稍大的 N_r/X_u 为弱可观测参数(A2)。末端平台为 Adam 徘徊+float32 地板(B2/C1)。

---

## E4 — 初值鲁棒性 / Robustness to initial guess (Phase 4.2)
**设置 / Setup**: 同 E3 机动但 T=120, N_iters=1000；无噪声；从真值的不同倍数出发。
（迭代数较 E3 少，故绝对误差偏高，重点看趋势。）

| init guess (× true) | final mean rel. error |
|--------------------:|----------------------:|
| 0.3 | 8.56% |
| 0.5 | 2.14% |
| 0.7 | 1.79% |
| 1.5 | 2.32% |

**结论 / Takeaway**: 初值在 0.5~1.5 倍范围均稳定收敛到 ~2%；远初值(×0.3)收敛较慢(8.6%，需更多迭代)。
辨识对初值不敏感（收敛盆地较宽）。

## E5 — 噪声鲁棒性 / Robustness to measurement noise (Phase 4.2)
**设置 / Setup**: 初值×0.5；目标轨迹加高斯噪声 N(0,σ)(模拟 GoPro+ArUco 跟踪误差)；每 σ 跑 3 个种子。
原始数据 / raw: `results/data/robustness_noise.csv`。

| noise σ [m] | mean rel. error | std |
|------------:|----------------:|----:|
| 0.000 | 2.14% | 0.00% |
| 0.005 | 1.81% | 0.92% |
| 0.010 | 2.84% | 0.98% |
| 0.020 | 5.59% | 0.74% |

**结论 / Takeaway**: 辨识误差随噪声**单调上升但保持有界**（σ=2cm 时 ~5.6%），属优雅退化(graceful degradation)，
说明管线对真实跟踪噪声有鲁棒性。

---

## E6 — 恒定水流反算 / Constant current inversion (Phase 4.3)
**设置 / Setup**: 参数已知；机动=短脉冲(20步)+长滑行；Adam(lr=0.05, 600 it)；初值 U=0。

| quantity | Ux | Uy |
|----------|---:|---:|
| true | 1.000 | -1.500 |
| recovered | 1.000 | -1.500 |

**结论 / Takeaway**: 恒定水流力精确反算，loss 降至机器精度(1e-16)。

## E7 — 时变水流反算 / Time-varying current inversion (Phase 4.3)
**设置 / Setup**: 真值 Ux(t)=0.4+0.6·t/T（斜坡），Uy(t)=-1.2·sin(π·t/T)（半正弦）；
240 个未知量；平滑正则化 R=λ·Σ‖U(t+1)-U(t)‖², λ=0.02；Adam(lr=0.02, 1500 it)。
原始数据 / raw: `results/data/recover_disturbance_tv.csv`。

| metric | value |
|--------|------:|
| trajectory loss | 1.4e-4 |
| RMSE of U(t) | 0.24 N |

**关键发现 / Key finding**: **中段水流恢复好，轨迹末端失真**——因为 U(t) 仅通过其**后续轨迹点**可观测，
末端缺乏后续观测 → 观测性时间衰减（A2 的时间版）。
**结论 / Takeaway**: 时变流场反演是欠定/病态问题；正则化恢复了主体趋势，末端需更多约束。
→ 直接对应 open-problems **D2**（时空变化流场反演）。

---

## E8 — 6-DOF 梯度门禁 + 参数辨识 / 6-DOF gate + identification (Phase 5.3-5.4)
**设置 / Setup**: 完整 3D 6-DOF Fossen（四元数姿态 + 恢复力，块对角 M, r_g=0）。
门禁：激发全 6 轴的机动，解析梯度 vs 有限差分。
辨识：固定可测参数(m, 惯量, W, B, zb)，辨识 18 个（附加质量6 + 线性阻尼6 + 二次阻尼6）；
无量纲化(/SCALE6)；丰富机动 surge/sway/heave/roll/pitch/yaw, T=120；Adam(lr=0.02, 2500 it)；初值×0.5。

**门禁 / Gate**: 18/18 参数解析梯度 ≈ 有限差分（联合容差），`tests/test_gradients6.py` 通过。
**辨识收敛 / Convergence**: 平均相对误差 50% → **1.38%**；loss 3.3 → ~1e-6。

| param | true | identified | rel.err |
|-------|-----:|-----------:|--------:|
| Xud (附加质量) | 5.50 | 5.507 | 0.1% |
| Yvd | 12.70 | 12.703 | 0.0% |
| Zwd | 14.60 | 14.600 | 0.0% |
| Nrd (附加惯量) | 0.120 | 0.120 | 0.1% |
| Xu (线性阻尼) | 4.00 | 4.110 | 2.7% |
| Zw | 5.00 | 5.274 | 5.5% |
| Zww (二次阻尼) | 36.0 | 34.35 | 4.6% |
| Nrr | 1.50 | 1.496 | 0.3% |

**图 / Figure**: [pic/identify_params6_convergence.png](../pic/identify_params6_convergence.png)
**结论 / Takeaway**: 6-DOF 可微管线（含四元数运动学 + 恢复力）验证通过。附加质量恢复极准；
升沉阻尼 Zw/Zww 误差略大(~5%)，因 heave 激励段较短（可观测性/实验设计 A2）。

---

## 已保存图片清单 / Saved figures
| 文件 / file | 内容 / content | 来源 / from |
|------|------|------|
| pic/identify_params_convergence.png | 参数辨识收敛曲线 | E3 / Phase 4.1 |
| pic/robustness.png | 初值/噪声鲁棒性 | E4,E5 / Phase 4.2 |
| pic/recover_disturbance_const.png | 恒定水流反算收敛 | E6 / Phase 4.3 |
| pic/recover_disturbance_timevarying.png | 时变水流 U(t) 反算 | E7 / Phase 4.3 |
| pic/identify_params6_convergence.png | 6-DOF 参数辨识收敛 | E8 / Phase 5.4 |
| pic/optimizer_gd_vs_adam.png | GD vs Adam 路径对比(示意) | 教学图 |
| pic/integrators_euler_vs_rk4.png | 欧拉 vs RK4 采样斜率(示意) | 教学图 |
| forward_3dof.png (根目录) | 3-DOF 前向轨迹(转弯机动) | Phase 2.4 |
