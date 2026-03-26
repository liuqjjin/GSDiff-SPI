# GSDiff-SPI完整理论与代码讲解

> 从第一性原理出发，面向完全初学者。每个公式都解释物理含义，
> 每段代码都标注对应的数学。

---

## 第1章 你在解决什么问题？

### 1.1 单像素相机的物理原理

普通相机有百万像素的传感器阵列，每个像素独立测量光强。
单像素相机（Single-Pixel Imaging, SPI）只有**一个**传感器——
每次测量只输出**一个数字**：整个场景的总光强。

那怎么恢复2D图像？答案是**编码调制**：

1. 用一个已知的空间图案（pattern）P_k 去调制场景
2. 单像素探测器测量调制后的总光强
3. 换一个pattern，再测一次
4. 重复K次，用K个数字反算出图像

数学上，第k次测量：

    y_k = Σ_{i,j} P_k(i,j) · x(i,j) + noise

其中 P_k(i,j) 是pattern在像素(i,j)处的值（0或1），
x(i,j) 是场景在该像素的亮度。

**对应代码**：`gsdiff/forward/spi.py` 中的 `measure()` 方法。

### 1.2 动态场景为什么难？

测量是串行的——第1个pattern在时刻t=0投射，第100个在t=0.1。
如果物体在移动，不同pattern"看到"的是不同时刻的场景：

    y_k = Σ_{i,j} P_k(i,j) · x(i,j, t_k) + noise
                                    ↑
                              不同k对应不同时刻！

传统方法假设x不变，用 x_hat = S^{-1} y 反演。
但如果x在变化，这个反演会得到一张"运动模糊"的混合图像。

**这就是DGI（差分鬼成像）的结果**——它不知道物体在动，
所以输出一张模糊图。

**对应代码**：`gsdiff/data/dgi.py`

---

## 第2章 2D Gaussian Splatting——怎么表示场景

### 2.1 为什么不直接用像素？

28×28图像有784个像素。如果要恢复10帧视频，
就有 10 × 784 = 7840 个未知数，但只有784个测量值。
方程组严重欠定（未知数远多于方程），无法求解。

解决思路：用**更少的参数**描述同样的图像。

### 2.2 高斯光斑

一个2D高斯函数就是一个"钟形光斑"：

    G_m(u) = a_m · exp(-½ (u - μ_m)ᵀ Σ_m⁻¹ (u - μ_m))

想象一下：在纸上放一个圆形的光源，中间最亮，
向边缘逐渐变暗。这就是一个2D高斯。

M个高斯叠加可以近似任何图像：

    s(u) = Σ_{m=1}^{M} G_m(u)    ← 这就是canonical scene

### 2.3 每个高斯的6个参数

| 参数 | 符号 | 含义 | 代码变量 |
|------|------|------|---------|
| 幅度 | a_m | 这个光斑有多亮 | `raw_amps` → softplus保正 |
| 中心Y | μ_{m,y} | 光斑在图像中的行位置 | `centers[:,0]` |
| 中心X | μ_{m,x} | 光斑在图像中的列位置 | `centers[:,1]` |
| 宽度Y | s_{m,y} | 光斑沿局部Y轴有多宽 | `exp(log_scales[:,0])` |
| 宽度X | s_{m,x} | 光斑沿局部X轴有多宽 | `exp(log_scales[:,1])` |
| 角度 | θ_m | 光斑的旋转角度 | `angles` |

**为什么用log_scales？** 宽度必须是正数。如果直接优化s，
梯度可能把它推到负数。取log后，exp(log_s)永远>0。

**为什么用softplus(raw_amps)？** 同理，亮度必须非负。
softplus(x) = log(1 + exp(x)) 是一个光滑的"截断到正数"函数。

### 2.4 协方差矩阵

宽度和角度组合成协方差矩阵：

    Σ_m = R(θ_m) · diag(s_y², s_x²) · R(θ_m)ᵀ

其中R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]] 是旋转矩阵。

**直觉**：如果s_y = 1, s_x = 3, θ = 45°，
这个高斯就是一个椭圆，长轴沿45°方向，长宽比3:1。

**对应代码**：`gaussian2d.py` 中的 `get_covariances()` 方法。

### 2.5 渲染：从参数到像素

"渲染"就是：给定M个高斯的参数，计算28×28个像素每一个的亮度。

对每个像素坐标(i,j)：
  1. 计算它到每个高斯中心的距离 diff = (i,j) - μ_m
  2. 用Mahalanobis距离 q = diff^T Σ^{-1} diff 衡量"加权距离"
  3. 高斯值 = a_m · exp(-0.5 · q)
  4. 所有M个高斯的值加起来 → 该像素的亮度

**对应代码**：`gaussian2d.py` 中的 `render()` 方法。

**为什么可微？** 因为exp、矩阵乘法、求和全是可微操作。
PyTorch的autograd会自动算出"如果μ_m往右移1个像素，
第(5,10)号像素的亮度变化多少"——这就是梯度。

### 2.6 参数数量

M=256个高斯 × 6个参数 = 1536个参数。
远少于 10帧 × 784像素 = 7840个未知数。
这就是为什么2DGS能在欠定系统中工作。

---

## 第3章 SE(2)运动模型——物体怎么动

### 3.1 什么是SE(2)

SE(2) = Special Euclidean group in 2D = 二维刚体运动群。
包含两种变换：
- **旋转**：绕某个中心旋转角度ω
- **平移**：在x和y方向移动(v_y, v_x)

### 3.2 运动方程

在时刻t（t从0到1），物体的位置变换为：

    μ_m(t) = R(ω·t) · (μ_m − c) + c + v·t

其中：
- R(ω·t) 是旋转矩阵（角度=ω·t弧度）
- c = (H/2, W/2) 是图像中心（旋转中心）
- v = (v_y, v_x) 是平移速度
- μ_m 是canonical（参考时刻）的高斯中心

**直觉**：先把高斯中心移到旋转中心(c)附近做旋转，
再移回去，再加上平移。

协方差也要一起旋转：

    Σ_m(t) = R(ω·t) · Σ_m · R(ω·t)ᵀ

**对应代码**：`se2.py` 中的 `transform_centers()` 和
`transform_covariances()` 方法。

### 3.3 为什么这个定理这么重要？（定理3.1）

高斯函数在仿射变换下仍然是高斯！

这意味着：
- 不需要在像素空间做插值/变形（你旧代码的grid_sample方法）
- 只需修改μ和Σ，然后重新渲染
- 梯度是精确的，不是近似的

### 3.4 运动参数

只有2-3个标量：
- v_y, v_x: 平移速度（代码：`velocity`）
- ω: 角速度（代码：`omega`）

所有M个高斯**共享**这些参数。这就是"共享运动模型"的含义。

---

## 第4章 SPI前向模型——完整链路

整个流程：

    高斯参数(1536个) + 运动参数(3个)
        ↓ 定理3.1：变换μ和Σ
    T帧视频 [T, 1, H, W]
        ↓ 与pattern内积
    K个测量值 [K]

### 4.1 渲染视频

对每个时刻 t_k（k=0,...,T-1）：
1. 用SE(2)变换所有高斯的μ和Σ
2. 渲染该时刻的帧

**对应代码**：`spi.py` 中的 `render_video()` 方法。

### 4.2 计算测量值

每个pattern对应一帧：

    y_k = Σ_{i,j} P_k(i,j) · frame[f(k)](i,j)

其中 f(k) = frame_idx[k] 告诉你第k个pattern属于哪一帧。

**对应代码**：`spi.py` 中的 `measure()` 方法。

### 4.3 z-score归一化

原始测量值 y_k ≈ 40（28×28图像，一半pattern是1，亮度约0.5）。
如果不归一化，MSE loss ≈ 40² = 1600，太大了。

z-score: y_norm = (y - mean(y)) / std(y)

归一化后 y_norm ∈ [-3, 3]，MSE ≈ 2.0（两个标准正态的距离）。

**关键**：y_pred和y_target要**各自独立**z-score，
不能用同一个mean/std。原因：初始化时两者的尺度完全不同。

**对应代码**：`admm.py` 和 `sgd.py` 中的 `zscore()` 函数。

---

## 第5章 ADMM求解器——怎么恢复参数

### 5.1 我们要解的优化问题

已知：测量值y和pattern P_k
未知：高斯参数θ_s 和 运动参数θ_g

    min_{θ} ½||zscore(y) - zscore(A(θ))||² + λ·TV(R(θ))

其中 A(θ) 是前向测量，R(θ) 是渲染出的视频，
TV是Total Variation正则化（让图像更平滑）。

### 5.2 为什么不能直接梯度下降？

理论上可以（这就是SGD solver做的事）。但问题是：
1. TV的梯度不好算（TV有proximal算子但梯度不光滑）
2. 以后换扩散先验时，扩散模型没有梯度

ADMM的核心思想：把一个难问题拆成两个简单问题。

### 5.3 变量分裂

引入辅助变量z（和视频同形状的张量）：

    min_{θ, z}  ½||zscore(y) - zscore(A(θ))||²  +  λ·TV(z)
    subject to:  z = R(θ)

约束 z = R(θ) 的意思是："z必须等于渲染出的视频"。

### 5.4 增广Lagrangian（Boyd et al. 2011）

    L_ρ(θ, z, u) = f(θ) + g(z) + (ρ/2)||R(θ) - z + u||²

其中：
- f(θ) = 数据保真项
- g(z) = λ·TV(z) = 正则项
- u = 对偶变量（拉格朗日乘子）
- ρ = 惩罚参数（控制约束的松紧度）

### 5.5 ADMM三步（这是核心算法）

**Step 1（θ-step）**：固定z和u，优化θ

    min_θ  f(θ) + (ρ/2)||R(θ) - (z - u)||²

"让渲染出的视频既匹配测量数据，又接近z-u"

用Adam跑100步梯度下降。

代码：`admm.py → theta_step()`
    target_vid = self.z - self.u     ← 正确符号！
    loss = loss_data + 0.5 * ρ * MSE(video, target_vid)

**Step 2（z-step）**：固定θ和u，优化z

    min_z  λ·TV(z) + (ρ/2)||z - (R(θ) + u)||²

这是TV去噪问题，有Chambolle算法可以精确高效求解。

代码：`admm.py → z_step()`
    self.z = prior.proximal(video + self.u, λ/ρ)  ← 正确符号！

**Step 3（u-step）**：更新对偶变量

    u ← u + R(θ) - z

代码：`admm.py → u_step()`
    self.u = self.u + video - self.z

### 5.6 符号为什么重要

Boyd 2011的scaled form增广Lagrangian是：

    L = f(θ) + g(z) + (ρ/2)||R(θ) - z + u||²

展开 ||R(θ) - z + u||²：
- 对θ求最小化 → ||R(θ) - (z - u)||² → target = z - u
- 对z求最小化 → ||z - (R(θ) + u)||² → input = R(θ) + u

如果把符号搞反（V2/V3的bug）：
- target = z + u → TV去掉的高频通过u被加回来
- z-step的效果被完全抵消
- ADMM发散

### 5.7 SGD baseline做什么不同？

SGD直接优化：

    loss = ½ MSE(zscore(y_pred), zscore(y_gt)) + λ · TV(video)

TV直接反向传播通过rendering pipeline。没有z和u。

代码：`sgd.py → step()`

### 5.8 ADMM vs SGD的区别

| | ADMM | SGD |
|--|------|-----|
| TV怎么做 | Chambolle proximal（精确） | 直接梯度反传（近似） |
| 有辅助变量吗 | 有(z, u) | 无 |
| Phase 2扩展 | 只换z-step的proximal | 需要重写 |

---

## 第6章 TV正则化——Chambolle算法

### 6.1 什么是Total Variation

    TV(x) = Σ_{i,j} sqrt((x[i+1,j]-x[i,j])² + (x[i,j+1]-x[i,j])²)

TV衡量图像的"粗糙程度"。自然图像TV低，噪声图像TV高。
最小化TV会让图像变平滑但保留边缘。

### 6.2 Proximal算子

z-step需要求解：

    prox_{τ·TV}(v) = argmin_z  TV(z) + 1/(2τ) ||z - v||²

含义："找一个z，它既TV低（平滑），又不离v太远"。

Chambolle算法通过对偶投影求解，50次迭代即可收敛。

**对应代码**：`tv.py → _chambolle()` 方法。

---

## 第7章 代码文件结构

```
v4/
├── configs/default.yaml     所有超参数配置
├── train.py                 训练入口（ADMM或SGD）
├── gsdiff/
│   ├── scene/
│   │   └── gaussian2d.py    第2章：2D高斯渲染
│   ├── motion/
│   │   └── se2.py           第3章：SE(2)运动变换
│   ├── forward/
│   │   └── spi.py           第4章：SPI前向模型
│   ├── prior/
│   │   └── tv.py            第6章：TV正则
│   ├── solver/
│   │   ├── admm.py          第5章：ADMM三步
│   │   └── sgd.py           第5章：SGD baseline
│   ├── data/
│   │   ├── simulation.py    仿真数据生成
│   │   ├── patterns.py      pattern生成
│   │   └── dgi.py           DGI baseline
│   └── utils.py             工具函数
└── requirements.txt
```

### 7.1 train.py做了什么？

```python
1. 读配置 → cfg = load_config("configs/default.yaml")
2. 生成数据 → data = generate_spi_data(...)
3. DGI baseline → dgi_img = dgi_reconstruct(patterns, measurements)
4. 建模型 → scene + motion + forward_model
5. 选求解器 → ADMMSolver 或 SGDSolver
6. 跑循环 → 每步打印 loss_data / prim_res / velocity
7. 评估 → PSNR + 运动恢复误差
8. 保存 → 图片/GIF/JSON/checkpoint
```

### 7.2 为什么分这么多文件？

因为Phase 2你要做的事情是：
- 把 `tv.py` 换成 `diffusion.py`（DAPS扩散先验）
- 只需要实现 `proximal(x, weight)` 接口
- **其他文件一行都不改**

如果全写在一个脚本里，改TV就会影响到渲染、运动、测量...全乱了。

---

## 第8章 训练过程的完整数据流

一轮ADMM的数据流：

```
θ = {centers, scales, angles, amps, velocity, omega}
  │
  ├── transform_centers(centers, t_grid)        → [T, M, 2]
  ├── transform_covariances(Sigma, t_grid)      → [T, M, 2, 2]
  │
  ├── render each frame                         → video [T, 1, H, W]
  │     └── for each pixel: sum over M Gaussians
  │
  ├── measure(video, patterns, frame_idx)       → y_pred [K]
  │     └── y_k = <P_k, frame[f(k)]>
  │
  ├── loss_data = MSE(zscore(y_pred), zscore(y_gt))
  ├── loss_consist = ρ/2 · MSE(video, z - u)
  │
  ├── loss.backward()    ← PyTorch自动微分
  │     └── 梯度传回到centers, scales, velocity...
  │
  └── optimizer.step()   ← Adam更新参数

然后：
  z ← TV_proximal(video + u)
  u ← u + video - z
```

---

## 第9章 如何debug和检查

### 9.1 判断是否正确的关键指标

| 指标 | 健康范围 | 异常 |
|------|---------|------|
| loss_data初始值 | 0.5 - 1.5 | > 10：z-score有问题 |
| loss_data最终值 | < 0.05 | > 0.5：没收敛 |
| prim_res | 单调下降或先升后降 | 指数增长：ADMM符号错 |
| velocity | 趋向GT | 不动或发散：lr太小/大 |
| TV/px | 稳定在0.3-1.0 | 持续增长：z变量有问题 |

### 9.2 常见问题排查

**问题：loss_data很大（>1000）**
→ 检查z-score是否在forward中正确应用

**问题：prim_res指数增长**
→ 检查ADMM的符号：target=z-u, prox_input=R(θ)+u

**问题：velocity不收敛**
→ 试增大lr_motion或减小lr_scene

**问题：图像全黑**
→ 检查高斯初始化：centers是否在[0,H)×[0,W)范围内
