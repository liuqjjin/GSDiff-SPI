# GSDiff-SPI 完整理论与代码讲解

> 从第一性原理出发，面向完全初学者。每个公式都解释物理含义，
> 每段代码都标注对应的数学。
> 本版对应 2026-07 升级后的代码状态：64×64 / T=20 / K=2500 / M=1000，
> ADMM(80×50) + warmup/transition，TV 与视频扩散两种可插拔先验。

---

## 第1章 你在解决什么问题？

### 1.1 单像素相机的物理原理

普通相机有百万像素的传感器阵列，每个像素独立测量光强。
单像素相机（Single-Pixel Imaging, SPI）只有**一个**传感器——
每次测量只输出**一个数字**：整个场景的总光强。

那怎么恢复 2D 图像？答案是**编码调制**：

1. 用一个已知的空间图案（pattern）P_k 去调制场景
2. 单像素探测器测量调制后的总光强
3. 换一个 pattern，再测一次
4. 重复 K 次，用 K 个数字反算出图像

数学上，第 k 次测量：

    y_k = Σ_{i,j} P_k(i,j) · x(i,j) + noise

**对应代码**：`gsdiff/forward/spi.py` 的 `measure()`。
默认 pattern 是 U[0,1] 随机图案（`data/patterns.py` 还提供
bernoulli / hadamard 三种排序 / fourier / S 矩阵等族，见第4章的适配域讨论）。

### 1.2 噪声模型

两种加噪约定（`data/simulation.py → _add_noise`）：

    (a) snr_db 约定：   σ² = var(y) / 10^(snr_db/10)     ← 相对 AC 方差
    (b) σ_abs 约定：    y ← y + N(0, σ_abs²)             ← 探测器参照绝对噪声

注意 (a) 里用的是 **var(y)**（AC 方差）而不是 DC 均值——
y 的 DC 分量约为 H·W·0.5·μ_pixel，很大但不含任何结构信息；
真正携带信息的是 y 围绕均值的**起伏**。若按 DC 定义 SNR，
"30 dB" 实际上是淹没信号的噪声。
(b) 是 2026-07 新增的：跨图案族公平对比**必须**用它（第7章解释为什么）。

### 1.3 动态场景为什么难？

测量是串行的——第 1 个 pattern 在 t=0 投射，第 2500 个在 t=1。
如果物体在移动，不同 pattern"看到"的是不同时刻的场景：

    y_k = Σ_{i,j} P_k(i,j) · x(i,j, t_k) + noise
                                    ↑
                              不同 k 对应不同时刻！

本项目把连续时间离散为 T=20 帧，pattern 到帧的分配是：

    ppf = ⌈K/T⌉ = 125,     frame_idx[k] = k // ppf

即前 125 个 pattern 打在第 0 帧上，依此类推。一个重要推论：
**pattern 的显示顺序就是时间轴**——这对有序基（如 Hadamard 排序基）
与运动的交互至关重要（第4章）。另有 `time_assignment_mode: interpolation`
选项，把 y_k 定义为相邻两帧的线性插值内积（`measure_interpolated`）。

### 1.4 为什么 DGI 只能得到运动模糊？

传统重建（DGI，差分鬼成像，`data/dgi.py`）假设场景静止，
本质上是把所有测量对同一张图做相关反演：

    x_dgi ∝ ⟨y_k · P_k⟩_k − ⟨y_k⟩⟨P_k⟩

代入动态场景 y_k = ⟨P_k, x(t_k)⟩，随机 pattern 之间近似不相关，
期望意义下得到的是**时间平均**：

    x_dgi ≈ (1/K) Σ_k x(t_k)  =  运动轨迹上所有时刻场景的平均

物体平移 8 像素 + 旋转 0.3 rad 的场景，平均出来就是一张拖影模糊图。
DGI 因此扮演两个角色：

1. **PSNR 下界基线**（运动模糊的代价有多大）
2. **初始化信息源**（第2章的 dgi_adaptive 初始化）

### 1.5 本项目的解法一句话

用极少参数把"场景"和"运动"分开建模——
M=1000 个 2D 高斯描述**规范场景**（canonical scene），
2–3 个标量描述**刚体运动**，二者联合可微地拟合全部 K 个测量。
未知数从像素视频的 T·H·W = 81920 压到约 6000，
再用先验（TV 或视频扩散模型）补足剩余的欠定性。

---

## 第2章 2D Gaussian Splatting——怎么表示场景

### 2.1 模型

规范场景是 M 个带方向 2D 高斯的叠加：

    s(u) = Σ_{m=1}^{M}  a_m · exp(-½ (u - μ_m)ᵀ Σ_m⁻¹ (u - μ_m))

当前默认 **M = 1000**（autoresearch-v3 campaign 唯一被接受的超参改动：
500→1000 带来 31.48→35.10 dB——在初始化与 ρ 延拓修好之后，
容量才是约束瓶颈）。

### 2.2 每个高斯的 6 个参数

| 参数 | 符号 | 存储 | 约束手段 |
|------|------|------|---------|
| 幅度 | a_m | `raw_amps` | softplus(raw) > 0 |
| 中心 | μ_m=[y,x] | `centers` [M,2] | 自由 |
| 尺度 | (s_y, s_x) | `log_scales` [M,2] | exp(log_s) > 0，可选 min_scale 下钳 |
| 角度 | θ_m | `angles` [M] | 自由 |

**为什么用 log_scales？** 宽度必须为正。直接优化 s 时梯度可能把它推成负数；
取 log 后 exp(log_s) 恒正，且乘性尺度在 log 域是加性的，优化更均匀。

**为什么用 softplus(raw_amps)？** 亮度非负。softplus(x)=log(1+eˣ)
是光滑的"截断到正数"，初始化 raw=0 → a≈0.69。

### 2.3 协方差矩阵

    Σ_m = R(θ_m) · diag(s_y², s_x²) · R(θ_m)ᵀ,
    R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]

**直觉**：s_y=1, s_x=3, θ=45° 就是一个长轴沿 45°、长宽比 3:1 的椭圆光斑。
**对应代码**：`gaussian2d.py → get_covariances()`。

### 2.4 渲染：稠密解析求值

对每个像素坐标 u（整数网格 0..H-1 × 0..W-1）：

    1. diff = u - μ_m                          [M, N, 2]   (N = H·W)
    2. quad = diffᵀ Σ_m⁻¹ diff                 [M, N]      (Mahalanobis 距离)
    3. vals = a_m · exp(-½·quad)               [M, N]
    4. img  = ReLU( Σ_m vals )                 [1,1,H,W]

**关键性质**：exp、矩阵求逆、einsum 全部可微，autograd 给出对全部
6M 个参数的**精确**梯度——这是与光栅化(tile-based rasterization)近似
渲染的本质区别。复杂度 O(M·H·W)，在 64×64、M=1000 规模下完全够快
（~280 s / 4000 步），因此**不做** CUDA 光栅化，精确性是卖点。
求逆前加 1e-6·I 防奇异。
**对应代码**：`gaussian2d.py → render()` 与 `forward/spi.py → _render_frame()`
（同一套数学，后者作用于运动变换后的参数）。

### 2.5 min_scale 钳制——抗混叠

`get_scales()` 支持 `s = exp(log_s).clamp(min=min_scale)`。理由：

SPI 的测量是**积分型**的（⟨P, I⟩ 对全图求和）。一个 σ < 0.3 px 的
亚像素高斯在整数网格上采样时，其离散求和严重偏离连续积分
2π·a·s_y·s_x——同一个连续高斯，中心落在像素中心和落在像素间隙，
离散能量可以差数倍。优化器会利用这种混叠"作弊"：
用亚像素尖峰拼凑测量值而不形成真实结构。把尺度钳在约 0.3 px 以上，
离散采样才忠实于连续模型。这是数值卫生措施，不是新机制
（配置 `scene.min_scale`，默认 0 = 关闭，需要时开）。

### 2.6 初始化：dgi_adaptive 及其修复的平移塌缩

三种模式（`scene.init_mode`）：

- `random`：中心均匀撒在 [0,H)×[0,W)，幅度全 0.69；
- `dgi`：在 random 基础上做**全局幅度缩放**，使渲染均值 ≈ DGI 均值
  （`init_from_image`，用精确 softplus 逆 log(expm1(·)) 反解 raw_amps。
  注意一个已修过的真实 bug：`dgi_reconstruct` 返回 z-score 图像，均值≈0，
  直接喂给幅度匹配会把场景塌成全黑——必须先 `normalize_01`）；
- `dgi_adaptive`（**当前默认**，Image-GS 风格，`init_adaptive`）：
  中心按概率 ∝ |∇DGI| + 0.2·均匀底（梯度大处=边缘处多放高斯），
  幅度取中心处 DGI 局部亮度（下钳 0.05），再叠加 `dgi` 的全局幅度缩放。

**为什么重要**：实验发现纯平移场景 @SNR25 在 random 初始化下
3 个种子里塌 2 个（11.4 dB，v_x 偏差 ~4 px，塌缩发生在 warmup 期，
与先验无关——是场景/运动联合景观的坏盆地问题）。dgi_adaptive
把它完全治好（25.08±0.19 dB），对旋转场景中性偏好（27.78±0.30）。
机制直觉：初始高斯已经大致落在"模糊物体"上，θ-step 早期
数据梯度不再需要靠错误的速度估计去"搬运"整团高斯。

### 2.7 参数量

M=1000 × 6 = 6000 个场景参数 + 2–3 个运动参数。
相对像素视频（81920 未知数）压缩约 13 倍；相对 K=2500 个测量，
场景仍是欠定的（先验负责补），但**运动是极度超定的**
（3 个 DOF 对 2500 个方程）——这就是速度能恢复到 ~0.01 px 精度的原因。

---

## 第3章 SE(2) 运动模型及其精确可传输的推广

### 3.1 核心定理：高斯族在仿射变换下封闭

设仿射映射 T(x) = A·x + b（A 可逆）。把一团高斯"搬运"过去，
新旧坐标满足 x' = T(x)，则对任意 x'：

    x' - T(μ) = A·(x - μ)
    ⇒ (x'-μ')ᵀ (AΣAᵀ)⁻¹ (x'-μ') = (x-μ)ᵀ Σ⁻¹ (x-μ)

即变换后**仍是高斯**，只需

    μ' = A·μ + b,     Σ' = A·Σ·Aᵀ

这意味着：不需要像素空间的 warp/grid_sample 插值——只改 μ 和 Σ
再重新渲染，梯度对运动参数是**精确**的。这条封闭性也划定了
运动模型推广的自然边界：**凡是仿射的都可以精确传输，非仿射的不行**。

（幅度约定：代码保持 a_m 不变。旋转时 |det A|=1 无歧义；
开启仿射分量时 det≠1，保持 a_m 相当于"保峰值亮度"而非"保光通量"
的约定——对反射式成像这是合理选择。）

### 3.2 基础 SE(2)

所有 M 个高斯**共享**一组刚体运动参数，t ∈ [0,1]：

    μ_m(t) = R(ω·t) · (μ_m − c) + c + v·t
    Σ_m(t) = R(ω·t) · Σ_m · R(ω·t)ᵀ

- v = [v_y, v_x]：t∈[0,1] 内的总位移（可学习，初始 0）
- ω：总旋转角（可学习，初始 0，`enable_rotation` 开关）
- c = **((H−1)/2, (W−1)/2)**：旋转中心（不可学习）

**规范中心为什么是 (H−1)/2 而不是 H/2？** 像素网格是 0..H−1，
其几何中心是 (H−1)/2。GT 仿真用 `scipy.ndimage.rotate` 生成，
该函数绕的正是这个中心——两侧约定必须逐位一致，否则
学到的 ω 会混进一个伪平移分量。

### 3.3 推广一：时间多项式（poly_degree=2，匀加速）

    d(t)     = v·t + a·t²          (a = [a_y, a_x]，+2 DOF)
    angle(t) = ω·t + β·t²          (β 角加速度，+1 DOF)

位移与角度只是 t 的函数，代入 3.2 的公式，每个时刻仍是一次仿射
变换——传输依然精确。GT 仿真器同步支持 `data.gt_accel / gt_beta`。

### 3.4 推广二：对称仿射分量（enable_affine）

线性部分推广为

    A(t) = R(angle(t)) · (I + t·L_sym),
    L_sym = [[l_yy, l_yx], [l_yx, l_xx]]      (+3 DOF：双轴缩放+剪切)

**为什么限制 L 为对称？** 极分解：任何可逆矩阵 = 旋转 × 对称正定。
旋转自由度已经由 ω 承担，如果 L 再含反对称（旋转）分量，
ω 与 L 之间会出现规范冗余（同一运动有无穷多参数化），
优化景观退化。对称化把冗余精确切除。
协方差传输同样精确：Σ(t) = A(t)·Σ·A(t)ᵀ。
**对应代码**：`se2.py → _A(), _displacement(), transform_centers(),
transform_covariances()`。

### 3.5 为什么不做每高斯独立运动 / 非刚性形变场？

可辨识性叙事会被破坏：共享运动是 3–9 个 DOF 对 2500 个测量的
超定问题；每高斯独立运动是 2M~3M 个 DOF，必须引入额外的
形变正则（违反奥卡姆），且最强竞争者（MC3-SPI、TMA-SPI、Monin 2021）
也都是刚体假设。非刚性列为 future work。

可学习 DOF 汇总：2 (v) [+1 ω] [+2 a, +1 β] [+3 L]，最小 2、最大 9。

---

## 第4章 z-score 损失——不变性、规范自由度与适配域

### 4.1 定义

    f(θ) = ½ · MSE( zscore(ŷ), zscore(y) ),
    zscore(x) = (x − mean(x)) / (std(x) + ε)

ŷ 与 y **各自独立**归一化（不是共用一组 mean/std）。
**对应代码**：`solver/admm.py` 与 `solver/sgd.py` 的 `zscore()` /
`_data_loss()`（另有 `loss_norm: target_std` 的固定尺度变体做对照）。

### 4.2 为什么需要它：尺度不变 + DC 不变

对任意 a>0, b：zscore(a·y + b) = zscore(y)。于是：

1. **DC 不变**：测量的 DC 分量 ≈ H·W·0.5·μ_pixel（数千量级）
   被 mean 减除自动消掉——它不携带结构信息，却主导原始 MSE。
2. **尺度不变**：初始化时 ŷ 与 y 的绝对尺度可以差几个量级，
   独立 z-score 让优化从第一步就在"比较波形"而不是"追绝对亮度"。

### 4.3 代价：规范自由度（gauge freedom）

不变性意味着损失**看不到**解的某些方向：场景整体亮度乘 a、
加一个匀值偏置 b，损失完全不变。所以重建只确定到一个
全局正仿射变换。这与评估约定是配套的：per-frame PSNR 先对每帧做
`normalize_01`（逐帧仿射规范化）再比较——评估忽略的规范
恰好覆盖了损失不约束的规范。

一个由此派生的细节（train.py）：GT-free 指标有两个版本，
`eval_residual`（整段测量全局 z-score）和 `eval_residual_pf`
（**逐帧** z-score）。后者去掉的逐帧仿射规范与 PSNR 评估的规范
对齐；全局版本会惩罚"帧间通量轨迹"的偏差——而强先验往往正是
用这个自由度去换取形状质量的，用它选型会与论文指标错位。

### 4.4 适配域局限：有序正交基下的梯度集中（实测结论）

z-score MSE 在**i.i.d. 随机图案**下工作得很好，但对
**有序正交基**（Hadamard cake-cutting/Walsh/natural、Fourier、S 矩阵）
存在结构性失效。机制：

- `frame_idx[k] = k//ppf` 使每帧只看到基的一个 125 维**连续切片**，
  有序基意味着"低频系数只在早期帧出现"；
- 正交基下 y 就是场景的变换系数，自然图像的系数动态范围约 5 个
  数量级；z-score 只做一次全局仿射，消不掉系数间的量级差；
- MSE 的梯度权重 ∝ 残差幅度 → 梯度质量集中在少数巨大低频系数上，
  系数长尾得不到学习信号。

判决性实验（pattern 基准 v2，无噪声 hadamard_cc）：运动恢复**完美**
（~0.04 px / 0.01），场景却停在 12.17 dB——证明失效与噪声无关、
与运动估计无关，就是场景梯度分配问题。时间调度
（sequential/stratified/random）也救不了它。

实测排名：**Bernoulli 0/1 是最优图案族（28.31±0.67 dB）**，
比 U[0,1] 随机高约 +2 dB（每光子 AC 对比度翻倍，且是 DMD 硬件原生格式）。
论文措辞：本方法的测量域 z-score MSE 适配 i.i.d. 随机掩模；
有序正交基需要系数白化或逐帧全谱采样（future work）。

---

## 第5章 ADMM——变量分裂、warmup/transition 与 HQS 消融

### 5.1 要解的问题与变量分裂

    min_θ  f(θ) + g(R(θ))        f = z-score 数据保真, g = 先验

R(θ) 是渲染出的视频 [T,1,H,W]。引入辅助变量 z（视频域张量）：

    min_{θ,z}  f(θ) + g(z)    s.t.  R(θ) = z

分裂的意义：g 落在**像素视频域**上，它的"求解器"可以是
Chambolle 精确 prox，也可以是一个学习的去噪器——θ 那边完全不用知道。

### 5.2 增广 Lagrangian（Boyd 2011 scaled form）与符号推导

    L_ρ(θ, z, u) = f(θ) + g(z) + (ρ/2)‖R(θ) − z + u‖² − (ρ/2)‖u‖²

三步迭代，符号从展开 ‖R(θ) − z + u‖² 直接读出：

    θ-step:  min_θ f(θ) + (ρ/2)‖R(θ) − (z − u)‖²    ← target = z − u
    z-step:  z = prox_{g/ρ}( R(θ) + u )              ← input = R(θ) + u
    u-step:  u ← u + R(θ) − z

**符号搞反的后果**（历史 bug）：target 写成 z+u 时，TV 去掉的高频
经 u 被原样加回，z-step 的效果被抵消，prim_res 指数增长、ADMM 发散。
现在的约定在 `solver/admm.py` 文件头和 CLAUDE.md 里双重固定。

### 5.3 θ-step 的实际形式（不精确内环）

θ-step 不是精确解，而是 **n_inner=50 步 Adam**：

    loss = f(θ) + λ_soft·TV_3D(R(θ)) + (ρ/2)·MSE(R(θ), z−u)

- `admm_soft_tv_weight=0.006` 的软 TV **不能设 0**（实验验证会明显掉点）：
  它在 θ 流形内部提供逐步平滑梯度，与 z-step 的先验不重复；
  3D 版本带 `temporal_tv_weight=0.05` 的时间差分项（0.02 太弱 0.08 过平滑）。
- 优化器**跨外环持久**（Adam 动量/二阶矩不重置），
  CosineAnnealingLR 以 T_max = num_outer×num_inner = 4000 全程退火，
  与 SGD 基线的总步数预算严格对齐。
- 梯度裁剪 5.0。

### 5.4 warmup 与 transition（关键机制）

z、u 初始化为 0。如果第一轮就带上 (ρ/2)‖R(θ)−(z−u)‖²，
θ 会被 z=0 锚向全黑场景，速度参数在错误的场景上无法收敛。因此：

    outer ≤ n_warmup      : θ-step 只用 f + soft TV（≡ SGD 损失），
                            z/u 完全不更新
    outer = n_warmup + 1  : 【transition 轮】θ-step 仍不带一致性项
                            （z 还不合法），但轮末执行 z_step/u_step
                            —— 用当前渲染初始化 z 和 u
    outer ≥ n_warmup + 2  : 完整 ADMM，target = z − u

transition 轮是一个被实验验证过的修补：修好之前，刚出 warmup 的
θ-step 会去追一个未初始化的 z−u，造成一次无谓的塌陷。
当前默认 `admm_n_warmup=20`（num_outer=80）。两个极端：
n_warmup = num_outer 时 z-step 永不执行，ADMM 退化为
"4000 步 SGD"；n_warmup < num_outer 才是真 ADMM，先验才生效。

### 5.5 ρ 单调延拓

出 warmup 后每个外环 `ρ ← ρ_growth · ρ`，当前默认 0.1 × 1.1^k。
含义：早期约束松（先验话语权大、θ 可以大幅重排），后期约束紧
（R(θ) 被拉向 z，迭代趋于定点）。这不是随意调参：Chan et al. 2017
证明**有界去噪器 + ρ 单调递增**的 PnP-ADMM 收敛到不动点——
单调延拓正是我们在非凸情形下唯一有理论支撑的 ρ 策略
（residual-balancing 自适应 ρ 预设精确 prox，PnP 下失效，不采用）。
实测 rho_growth 1.05→1.1（配合 dgi_adaptive 初始化）带来 +5.3 dB
（27.59→32.93，3 种子）——晚期锚定力度是被低估过的关键量。

### 5.6 HQS 消融：对偶变量 u 到底贡献了什么

`solver.hqs: true` 令 u ≡ 0（半二次分裂）：θ-step 直追 z，
z-step 输入是裸 R(θ)。ADMM 与 HQS 的**唯一**算法差异就是 u。
3 种子实测（SNR25 旋转场景）：

| 先验 | ADMM | HQS (u≡0) | 判定 |
|------|------|-----------|------|
| TV (Chambolle 精确 prox) | 26.32±0.40 | 26.24±0.32 | 打平（噪声内） |
| 扩散 (PnP 去噪器) | 27.59±0.24 | 26.60±0.19 | **ADMM +0.99 dB** |

**机制（Bregman 记忆）**：u-step 展开是 u_K = Σ_k (R(θ_k) − z_k)，
u 是历史约束违反的**累积和**。精确 prox 无系统偏差，违反项均值为零，
u 没东西可记——HQS 打平。而 PnP 去噪器有**系统性偏置** b
（总是抹掉某些真实细节），每轮 z ≈ v − b，u 就单调累积 +b；
θ-step 的 target = z − u 于是被反向补偿，定点处去噪器的偏置被 u
精确抵消。一句话：**对偶变量是纠正有偏先验的记忆项，
先验无偏时它是多余的**。这是论文选 ADMM 而非 HQS 的实测归因。

### 5.7 诚实的收敛性声明

必须写清楚三个非标准之处：

1. **f 非凸**：R(θ) 对 θ 高度非线性（exp、旋转、softplus），
   z-score 归一化进一步破坏凸性；
2. **θ-step 不精确**：50 步 Adam，不是 argmin；
3. **PnP 模式没有显式 g(z)**：去噪器不是任何函数的精确 prox。

因此 Boyd 2011 的凸收敛理论**不适用**。能够诚实引用的是
Chan et al. 2017（有界去噪器 + ρ 单调延拓 → 不动点收敛）与
Ryu et al. 2019（去噪器收缩性条件）。工程上用三个诊断量监控
（`admm.py → step()` 的 info dict）：

    prim_res = ‖R(θ) − z‖²          原始残差（应下降或先升后降）
    dual_res = ρ·‖z_k − z_{k−1}‖²   对偶残差代理
    u_norm   = mean|u|               无界增长 = 流形不可行警报
                                     （θ 流形根本表达不了 z 要求的图像）

### 5.8 SGD 基线与 RED-diff 变体

SGD（`solver/sgd.py`）：单环 Adam 直接最小化 f + λ·TV_3D(R(θ))，
TV 梯度直接反传过渲染管线，无 z/u。lr_motion=0.15 是 lr_scene 的
~17 倍（运动 DOF 少但必须先收敛），CosineAnnealingLR 同预算。
真 ADMM 修好 transition 后稳定优于 SGD 约 +3 dB。
`red_weight > 0` 时 SGD 加 RED-diff 式单环扩散正则
½‖V − D_σ(V)‖²（D_σ 断开梯度、σ 沿同款退火）——
"扩散先验 × 参数化渲染器"的标准单环耦合基线，用于对照表完整性。
`freeze_motion: true` 则给出静止场景拟合的运动模糊下界基线。

---

## 第6章 PnP 扩散先验——Tweedie、独立 σ 退火与两次"文献建议被否决"

### 6.1 去噪器进 z-step（PnP 替换）

把 5.2 的 z-step 中的 prox_{g/ρ} **替换**为学习的高斯去噪器：

    z = clamp( D_σ_k( R(θ) + u ), [0,1] )

这是 Venkatakrishnan et al. 2013 的 Plug-and-Play 模板：不存在
显式 g(z) 使 D_σ 恰为其 prox——所以第5.7节的收敛声明必须弱化。
去噪器是 `prior/unet3d.py` 的轻量 3D UNet（~2.8M 参数，
输入 [1,1,T,H,W] 单通道灰度视频，FiLM 式 log σ 条件化，ε-预测目标），
在 ~5000 个合成 SE(2) 运动视频上训练（EDM 式 log-线性噪声表
σ∈[0.002, 0.5]，`prior/noise_schedule.py`）。

### 6.2 Tweedie 单步去噪的推导

训练时加噪：v = x₀ + σ·ε，ε~N(0,I)。Tweedie 公式给出后验均值：

    E[x₀ | v] = v + σ² · ∇_v log p_σ(v)

而 ε-预测网络学的是 E[ε|v]，score 与它的关系是
∇_v log p_σ(v) = −E[ε|v]/σ（高斯平滑密度对 v 求导直接得到）。代入：

    D_σ(v) = v − σ · ε̂_θ(v, σ)        ← 单步 Tweedie（denoise_steps=1，默认）

多步选项（denoise_steps>1）：DDIM 从 σ 沿阶梯降到 σ_min，
每步 x₀ 预测 + 重噪到下一级；阶梯间距 `ddim_spacing: linear|log`。
实测单步 Tweedie 就是最优（autoresearch v3 确认 denoise_steps=1 局部最优）。

### 6.3 独立 log-线性 σ 退火（本项目最重要的一个设计决定）

σ 不从 ρ 推导，而是走独立的 log-线性时间表：

    σ_k = exp( (1−k/N)·log σ_start + (k/N)·log σ_end ),
    N = num_outer − admm_n_warmup = 60,   默认 0.3 → 0.05

**反面教材（历史 bug）**：早期版本用 σ = √(tv_weight/ρ)，
把 σ 压到 0.03–0.14——在这个量级去噪器近似恒等映射，
扩散先验实测与弱 TV 不可区分。TV 系数与扩散先验强度本无关系。

**文献锚定**：独立噪声退火正是 2025-26 的最佳实践——DAPS 的
解耦噪声退火、DPIR/DiffPIR 的 σ 阶梯、Taming-ADMM、FlowADMM
全部把去噪强度与 ρ 解耦。而且"忽略 ρ"有一个严格的等价解释：
若坚持把 D_σ_k 读作 prox_{λ_k g/ρ_k}（对应去噪强度 σ_k=√(λ_k/ρ_k)），
自由选 σ_k 就等价于**隐式的迭代变权先验** λ_k = ρ_k·σ_k²。
一句话把"经验选择"变成"带文献锚的迭代变权正则化设计"。
σ 全程钳在训练区间 [0.002, 0.5] 内，网络永不越界查询。
簿记：train.py 构造后必须调 `prior.set_n_steps(N)`，
否则 _n_steps=1，σ 一次调用就跳到 σ_end。

### 6.4 两个文献建议被受控实验否决（都要写进论文）

1. **重加噪（renoise）**：z = D_σ(v + σ·ε)。动机（DDfire/FlowADMM）：
   naive PnP 的已知失败模式是 off-manifold 查询——v 偏离干净流形的
   方式既非高斯也不在 σ 量级。实测 **−2.9 dB**（24.67±0.51）。
   解释：R(θ)+u 是参数化高斯渲染，本来就光滑、近流形，
   注入噪声反而破坏 θ-step 的锚定目标。M=1000 下复测仅回到中性，
   仍不采用。**教训：on-manifold 修正是为像素域优化设计的，
   参数化渲染器天然规避了该失败模式。**
2. **σ_end → 0.01–0.02**：退火收敛理论要求 σ→0（先验最终退场）。
   实测 **−3.0 dB**（24.61±0.46），在 M=1000 新工作点复测再次被拒
   （−2.2 dB）——稳健结论。保持 σ_end=0.05：终态仍保留温和去噪，
   因为 θ 流形 + 有限内环的定点并不满足"数据项最终完全接管"的前提。

### 6.5 实测收益与波动性

扩散 > TV：SNR25 下 +1.35 dB（3 种子）。注意扩散路径跨 GPU
非确定性 ~1 dB（TV 路径 ±0.02 dB）——**一切扩散先验结论必须
多种子报告**。`DiffusionPrior.energy()` 返回的空间 TV 只是监控量
（复用 ADMM 日志通路），与优化无关；`proximal(x, weight)` 的
weight 参数仅为与 TVPrior 接口兼容，**被忽略**。

---

## 第7章 评估方法论——GT-free 选型的三条实验教训

### 7.1 指标体系

- **per-frame PSNR**：每帧 normalize_01 后对 GT 求 PSNR，再取均值
  （规范约定见 4.3）；
- **运动误差**：|v_est − v_gt|、|ω_est − ω_gt|（好解 ~0.01 px / 0.002 rad）；
- **GT-free 残差**：真实实验没有 GT，超参选型必须有不看 GT 的指标。

### 7.2 教训一：训练内 holdout 会破坏被测系统

第一版 GT-free 指标是 `holdout_mod=10`：抽走 10% 测量不参与训练、
只做验证。结果：SNR25+旋转+扩散的场景/运动联合问题是**双峰刀刃盆地**，
移除 10% 测量就足以翻转盆地——run14 种子42 从 27.34 掉到
10.15 dB（运动发散）。第一次 autoresearch campaign 整体作废
（quarantine 在 results/ar_v1_holdout_destab/）。
**规则：本问题上永远不用 holdout_mod 做选型。**

修复：`data.holdout_extra=250` —— **非侵入评估集**。用独立 RNG 流
RandomState(seed+9999) 另生成 250 个 U[0,1] pattern 在 GT 帧上测量
（同噪声模型），训练数据逐位不变（已验证 bit-identical）。
它是纯探针：`eval_residual`（全局 z-score）与 `eval_residual_pf`
（逐帧 z-score，与 PSNR 规范对齐，见 4.3）写入 results.json。

### 7.3 教训二：均值聚合 = 塌缩彩票

双峰问题下"多种子取均值"被单个塌缩种子劫持——第二次 campaign
因此作废（results/ar_v2_meanlottery/）。现行协议
（`scripts/autoresearch.py`）：

    metric = mean_{运动情形}( median_{种子}( eval_residual ) )
    接受条件：metric 改善 且 塌缩计数不增
    塌缩判定：eval_residual > 0.05（GT-free 警报，实测 100% 分离——
              塌缩盆地 ~0.22 量级 vs 正常 ~0.003–0.01 量级）

逐情形**中位数**对少数塌缩稳健；塌缩计数守卫防止"中位数变好
但塌得更频繁"的假接受。Ke=400 验证过 eval_residual 与 PSNR
排序一致（0.0028↔32.4 dB vs 0.0038↔27.3 dB）；早先观察到的
"残差-PSNR 反相关"是均值聚合的伪影，不是指标的问题。

### 7.4 教训三：跨图案族对比必须用探测器参照噪声

AC 方差 SNR 约定（1.2 节 (a)）对图案族**不公平**：有序正交基的
var(y) 被少数巨大低频系数主导，同样标称 "25 dB" 换算出的绝对噪声
σ 是随机图案的 ~4 倍（实测 1.77 vs 0.45）。若用 (a) 做跨族基准，
有序基被双重惩罚（梯度集中 + 更大绝对噪声），结论被污染。
`data.noise_sigma_abs` 的探测器参照约定 (b) 把噪声定义在物理层面
（同一探测器同一 σ），是跨族基准的**必需**设置——pattern 基准 v2
即由此得出第4.4节的"结构性失效与噪声无关"的干净结论。

### 7.5 多种子协议

种子 {7, 11, 42} 报 mean±std（`scripts/run_multiseed.py` →
results/<exp>/summary.json）；扩散实验因 ~1 dB 跨 GPU 波动
建议 ≥5 种子。选型永远用 GT-free 指标，PSNR 只用于报告。

---

## 第8章 文件地图与数据流

### 8.1 文件地图

```
gsdiff/
├── scene/gaussian2d.py     第2章：2DGS 渲染 + dgi/dgi_adaptive 初始化 + min_scale
├── motion/se2.py           第3章：SE(2) + poly_degree=2 + enable_affine（精确传输）
├── forward/spi.py          第1章：渲染视频 → 内积测量（uniform / interpolation）
├── prior/
│   ├── tv.py               TVPrior (2D Chambolle) + TVPrior3D（各向同性 3D）
│   ├── diffusion.py        第6章：PnP 去噪器，独立 σ 退火，Tweedie/DDIM
│   ├── unet3d.py           ~2.8M 参数 3D UNet ε-预测网络
│   └── noise_schedule.py   EDM 式 log-线性 σ 表（训练 + 推理钳制）
├── solver/admm.py          第5章：三步 ADMM + warmup/transition + hqs 开关
├── solver/sgd.py           第5.8：Adam 基线 + red_weight + freeze_motion
├── data/simulation.py      GT 生成、两种噪声约定、非侵入评估集
├── data/patterns.py        图案族 + 排序 + 时间调度（第4.4 的实验对象）
├── data/dgi.py             DGI 基线（下界 + 初始化源）
└── utils.py                seed / config / PSNR / I/O

scripts/
├── generate_video_dataset.py  扩散先验训练集（[N,T,H,W] SE(2) 视频）
├── train_diffusion_prior.py   UNet3D ε-预测训练
├── run_multiseed.py           多种子 mean±std 聚合（7.5）
└── autoresearch.py            坐标下降超参搜索，中位数+塌缩守卫（7.3）

train.py                    入口：数据→DGI→初始化→求解→评估→保存
configs/default.yaml        当前工作点（下表）
```

先验是插件：`train.py` 按 `solver.prior_type` 实例化 TVPrior(3D) 或
DiffusionPrior，ADMMSolver 只调 `prior.proximal(x, weight)`，
不知道拿到的是哪个（唯一例外：diffusion 需要 `set_n_steps`）。

### 8.2 当前默认工作点（configs/default.yaml，2026-07-18 定稿）

    64×64, T=20, K=2500(+250 eval), pattern=random U[0,1]
    （Bernoulli 0/1 实测更优 +2 dB，切换待定）, SNR25
    M=1000, init=dgi_adaptive, ADMM 80×50, warmup=20
    rho=0.1, rho_growth=1.1, soft_tv=0.006, temporal_tv=0.05(3DTV)
    prior=diffusion, Tweedie-1, σ: 0.3→0.05, lr=(0.009, 0.15)

轨迹：27.59（旧基线）→ 32.93（dgi_adaptive + rho_growth 1.1）
→ **35.10 dB**（M=1000），全部 3 种子 + GT-free 选型验证。

### 8.3 一轮 ADMM 外环的数据流

```
θ = {centers, log_scales, angles, raw_amps, v, ω [, a, β, L]}
  │
  ├─ θ-step × n_inner=50:
  │    A(t), d(t)  ──SE(2)/仿射──▶  μ_m(t), Σ_m(t)      [T,M,·]
  │    稠密渲染   ──────────────▶  V = R(θ)             [T,1,64,64]
  │    ⟨P_k, V_f(k)⟩ ───────────▶  ŷ                    [2500]
  │    loss = ½MSE(zs(ŷ), zs(y)) + 0.006·TV3D(V)
  │           + (ρ/2)·MSE(V, z−u)          ← warmup/transition 时无此项
  │    backward → clip 5.0 → Adam → cosine
  │
  ├─ z-step:  z = prior.proximal(V + u, λ/ρ)
  │             TV:  Chambolle 精确 prox（2D 逐帧 / 3D 联合）
  │             扩散: clamp(D_σ_k(V+u))，σ_k 独立退火，weight 忽略
  ├─ u-step:  u ← u + V − z                 （hqs 时 u≡0）
  └─ ρ ← 1.1·ρ；记录 prim_res / dual_res / u_norm / σ
```

### 8.4 健康指标速查

| 指标 | 健康 | 异常及含义 |
|------|------|-----------|
| loss_data 初值 | 0.5–1.5 | >10：z-score 没生效 |
| loss_data 终值 | <0.05 | >0.5：未收敛 |
| prim_res | 下降或先升后降 | 指数增长：ADMM 符号错（5.2） |
| u_norm | 有界 | 单调无界：θ 流形不可行（5.7） |
| velocity | →GT | 不动/发散：lr_motion 或初始化问题（2.6） |
| eval_residual | <0.05 | >0.05：塌缩盆地警报（7.3） |

