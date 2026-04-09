# STEP 论文分析与 GSDiff-SPI 结合的第一性原理推导

**论文**：STEP: A Framework for Solving Scientific Video Inverse Problems with Spatiotemporal Diffusion Priors  
**arXiv**：2504.07549v2 (Caltech + OpenAI, 2025)  
**核心关键词**：视频逆问题、时空扩散先验、DAPS、PnP-Diffusion、VAE 潜在空间  

---

## 一、STEP 论文核心框架（逐层拆解）

### 1.1 问题设定

考虑视频逆问题的统一形式：

$$
\mathbf{y} = \mathcal{A}(\mathbf{x}_0) + \mathbf{n}
\tag{STEP-1}
$$

- $\mathbf{x}_0 \in \mathbb{R}^{n_f \times n_h \times n_w}$：待重建的干净视频（$n_f$ 帧，每帧 $n_h \times n_w$）  
- $\mathcal{A}$：正向测量算子（可以是非线性的，例如黑洞闭合量）  
- $\mathbf{n}$：测量噪声  

**目标**：从 $\mathbf{y}$ 恢复 $\mathbf{x}_0$，等价于从后验 $p(\mathbf{x}_0|\mathbf{y}) \propto p(\mathbf{y}|\mathbf{x}_0)\,p(\mathbf{x}_0)$ 采样。

### 1.2 潜在空间公式化

直接在高维视频空间中参数化先验 $p(\mathbf{x}_0)$ 代价高昂。STEP 引入 VAE（变分自编码器）：

$$
\mathbf{z}_0 = \mathcal{E}(\mathbf{x}_0), \quad \mathbf{x}_0 \approx \mathcal{D}(\mathbf{z}_0)
\tag{STEP-2}
$$

假定 $\mathbf{x}_0 \in \mathrm{range}(\mathcal{D})$，则测量方程变为：

$$
\mathbf{y} = \mathcal{A}(\mathcal{D}(\mathbf{z}_0)) + \mathbf{n}
\tag{STEP-3}
$$

后验变为 $p(\mathbf{z}_0|\mathbf{y})$；重建时先在潜在空间采样再解码。

**关键性质**：$\mathcal{D}$ 对每帧独立作用（2D 空间解码器），不在时间维上做耦合，便于与时序扩散模块分离。

### 1.3 时空扩散先验架构（三步训练）

**Step 1 —— 训练图像 LDM 作为空间先验**

VAE 训练目标：

$$
\mathcal{L}_\text{VAE} = \mathbb{E}\left[\|\mathcal{D}(\mathbf{z}_0) - \mathbf{x}_0\|_1\right] + \beta_\text{KL}\,D_\text{KL}(q_\phi(\mathbf{z}_0|\mathbf{x}_0)\|p(\mathbf{z}_0))
\tag{STEP-4}
$$

图像扩散 UNet 的 score-matching 训练目标（EDM 形式）：

$$
\mathcal{L}_\text{IDM} = \mathbb{E}_{{\mathbf{z}_0, \boldsymbol{\epsilon}, t}}\left[\sigma_t^2\, \|s_\theta(\mathbf{z}_t; \sigma_t) - \nabla_{\mathbf{z}_t}\log p_t(\mathbf{z}_t|\mathbf{z}_0)\|^2\right]
\tag{STEP-5}
$$

**Step 2 —— 图像先验 → 时空先验**

在 2D UNet 的每个卷积块旁并联一个 **零初始化** 的 1D 时序卷积模块，输出通过 $\alpha$-混合融合：

$$
\mathbf{f}_\text{out} = (1-\alpha)\,\mathbf{f}_\text{spat} + \alpha\,\mathbf{f}_\text{temp}, \quad \alpha \in \mathbb{R},\ \alpha(0) = 0
\tag{STEP-6}
$$

初始时 $\alpha=0$，保证不破坏预训练图像先验权重，并可以用 ON/OFF 开关让模型同时兼容图像和视频输入。

**Step 3 —— 图像+视频联合微调**

以概率 $p_\text{joint}$ 接收视频，以 $1-p_\text{joint}$ 接收图像（temporal OFF），用 STEP-5 损失联合训练，防止过拟合小规模视频数据集。

### 1.4 DAPS 采样算法（Algorithm 1 精确推导）

**核心定理**（DAPS 原文 Proposition 1）：  
若 $\hat{\mathbf{z}}_0 \sim p(\mathbf{z}_0|\mathbf{z}_{t_i}, \mathbf{y})$，则  
$$
\mathbf{z}_{t_{i-1}} \sim \mathcal{N}(\hat{\mathbf{z}}_0,\ \sigma_{t_{i-1}}^2\,I) \implies \mathbf{z}_{t_{i-1}} \sim p(\mathbf{z}_{t_{i-1}}|\mathbf{y})
\tag{DAPS-prop}
$$

这把"从 $p(\mathbf{z}_0|\mathbf{y})$ 采样"分解为序列问题：每步从 $p(\mathbf{z}_0|\mathbf{z}_{t_i}, \mathbf{y})$ 采样，再加噪到下一级。

**三步循环**（$i = N,\ldots,1$）：

**① PF-ODE 反向求解（pure diffusion denoising）**

$$
d\mathbf{z}_t = -\dot{\sigma}_t \sigma_t\,\nabla_{\mathbf{z}_t}\log p(\mathbf{z}_t;\sigma_t)\,dt
\tag{DAPS-1}
$$

从 $\mathbf{z}_{t_i}$ 积分到 $t=0$，得到纯先验下的 clean estimate $\hat{\mathbf{z}}_0$（不依赖测量）。

**② MCMC 数据一致性修正（HMC）**

为从 $p(\mathbf{z}_0|\mathbf{z}_{t_i}, \mathbf{y}) \propto p(\mathbf{y}|\mathcal{D}(\mathbf{z}_0))\,p(\mathbf{z}_0|\mathbf{z}_{t_i})$ 采样，运行 HMC：

$$
p^+ = (1-\gamma\eta)\,p + \eta\,\nabla_{\mathbf{z}_0}\log p(\mathbf{z}_0|\mathbf{z}_{t_i}) + \sqrt{2\gamma\eta}\,\boldsymbol{\epsilon}
\tag{DAPS-2a}
$$

$$
\mathbf{z}_0^+ = \mathbf{z}_0 + \eta\,p^+
\tag{DAPS-2b}
$$

其中 **数据梯度** 为：

$$
\nabla_{\mathbf{z}_0}\log p(\mathbf{y}|\mathcal{D}(\mathbf{z}_0)) = -\frac{1}{\sigma_n^2}\,\frac{\partial \mathcal{A}(\mathcal{D}(\mathbf{z}_0))^T}{\partial \mathbf{z}_0}\left(\mathcal{A}(\mathcal{D}(\mathbf{z}_0)) - \mathbf{y}\right)
\tag{DAPS-3}
$$

**先验梯度**近似为：

$$
\nabla_{\mathbf{z}_0}\log p(\mathbf{z}_0|\mathbf{z}_{t_i}) \approx \nabla_{\mathbf{z}_0}\log p(\mathbf{z}_{t_i}|\mathbf{z}_0) + s_\theta(\mathbf{z}_0, t_\min)
\tag{DAPS-4}
$$

**③ 进入下一噪声级别**

$$
\mathbf{z}_{t_{i-1}} \sim \mathcal{N}(\hat{\mathbf{z}}_0,\ \sigma_{t_{i-1}}^2\,I)
\tag{DAPS-5}
$$

---

## 二、GSDiff-SPI 核心框架回顾

### 2.1 正向模型

SPI 测量方程（线性）：

$$
y_k = \langle \mathbf{P}_k,\, \mathbf{I}_{f(k)} \rangle = \sum_{i,j} P_k[i,j]\cdot I_{f(k)}[i,j], \quad k = 1,\ldots,K
\tag{SPI-1}
$$

矩阵形式：$\mathbf{y} = \mathbf{A}\,\mathbf{x}_0 + \mathbf{n}$，其中 $\mathbf{x}_0 = \mathrm{vec}(\mathbf{V}) \in \mathbb{R}^{T \times H \times W}$，

$$
(\mathbf{A}\,\mathbf{x}_0)_k = \mathrm{vec}(\mathbf{P}_k)^T\,\mathrm{vec}(\mathbf{x}_0^{(f(k))})
\tag{SPI-2}
$$

**SPI 算子 $\mathbf{A}$ 是线性的**，且 $\mathbf{A}^T \mathbf{v}$ 有解析梯度（随机模式内积的转置等于加权投影之和）。

### 2.2 参数化场景表示

当前用 2D 高斯 Splatting 参数化场景：

$$
\mathbf{x}_0 = R(\boldsymbol{\theta}) \in \mathbb{R}^{T \times 1 \times H \times W}
\tag{SPI-3}
$$

$$
\boldsymbol{\theta} = \{\text{centers},\,\text{log\_scales},\,\text{angles},\,\text{raw\_amps}\} \cup \{\mathbf{v},\,\omega\}
$$

参数量：$500 \times 6 + 3 = 3003$ DOF，远小于全视频维度 $T \times H \times W = 20 \times 64 \times 64 = 81920$。

### 2.3 当前 ADMM 求解器（TV 先验）

增广拉格朗日（Boyd 2011 scaled form）：

$$
\mathcal{L}_\rho(\boldsymbol{\theta}, \mathbf{z}, \mathbf{u}) = f(\boldsymbol{\theta}) + g(\mathbf{z}) + \frac{\rho}{2}\|R(\boldsymbol{\theta}) - \mathbf{z} + \mathbf{u}\|^2
\tag{ADMM-1}
$$

三步迭代：
- $\boldsymbol{\theta}$-step：Adam 梯度下降，目标 $\mathbf{z} - \mathbf{u}$
- $\mathbf{z}$-step：TV 近端算子（Chambolle），$g(\mathbf{z}) = \lambda\,\mathrm{TV}(\mathbf{z})$
- $\mathbf{u}$-step：$\mathbf{u} \leftarrow \mathbf{u} + R(\boldsymbol{\theta}) - \mathbf{z}$

**TV 先验的局限**：TV 仅能编码分片常数结构，无法捕捉视频中复杂的纹理、时间相关性和非刚体运动细节。

---

## 三、第一性原理推导：两者是否可以结合？

### 3.1 结构对应性分析

将 GSDiff-SPI 和 STEP 置于同一贝叶斯框架下：

| 元素 | STEP | GSDiff-SPI |
|------|------|-----------|
| 观测变量 | $\mathbf{y}$ | $\mathbf{y}$ |
| 未知量 | $\mathbf{x}_0 \in \mathbb{R}^{T \times H \times W}$（全视频） | $\boldsymbol{\theta}$（3003 参数）|
| 正向算子 $\mathcal{A}$ | 黑洞/MRI（线性/非线性） | SPI（**线性**，$y_k = \langle P_k, x_0^{f(k)}\rangle$）|
| 先验 $p(\mathbf{x}_0)$ | 视频扩散模型 | TV 正则项（等价于 Laplace 先验）|
| 求解策略 | DAPS 后验采样 | ADMM MAP 优化 |

**关键观察**：SPI 的正向算子 $\mathcal{A}$ 是**线性的**，而 STEP 已被证明对非线性算子（黑洞闭合量）有效。SPI 比 STEP 的应用场景更简单，因此 **理论上完全兼容**。

### 3.2 梯度可行性证明

STEP MCMC 步骤（DAPS-3）需要计算：

$$
\nabla_{\mathbf{x}_0}\log p(\mathbf{y}|\mathbf{x}_0) = -\frac{1}{\sigma_n^2}\,\mathbf{A}^T(\mathbf{A}\mathbf{x}_0 - \mathbf{y})
\tag{COMPAT-1}
$$

对 SPI 算子展开：

$$
\mathbf{A}^T(\mathbf{A}\mathbf{x}_0 - \mathbf{y}) \Bigr|_{\text{frame }t} = \sum_{k:\,f(k)=t} (y_k^\text{pred} - y_k)\,\mathbf{P}_k
\tag{COMPAT-2}
$$

**这就是 SGD 求解器目前反向传播所用的梯度**，只是形式被封装在 PyTorch autograd 中。梯度**已存在且可计算**，无需任何修改。

Z-score 归一化对梯度的影响：当前损失 $f(\boldsymbol{\theta}) = \frac{1}{2}\|\mathrm{zscore}(\hat{\mathbf{y}}) - \mathrm{zscore}(\mathbf{y})\|^2$。DAPS 要求的数据梯度是 $\nabla \log p(\mathbf{y}|\mathbf{x}_0)$，对应 $\nabla_{\mathbf{x}_0} f$（取负号），两者通过链式法则直接对应。

### 3.3 变量分裂的代数等价性证明

ADMM 的核心是变量分裂：把"数据保真"和"先验"解耦到两个子问题。

**命题**：ADMM z-step 的近端算子

$$
\mathbf{z}^* = \arg\min_{\mathbf{z}}\left\{-\log p(\mathbf{z}) + \frac{\rho}{2}\|\mathbf{z} - \hat{\mathbf{v}}\|^2\right\}, \quad \hat{\mathbf{v}} = R(\boldsymbol{\theta}) + \mathbf{u}
\tag{COMPAT-3}
$$

等价于在以 $\hat{\mathbf{v}}$ 为中心、方差 $\sigma_z^2 = 1/\rho$ 的高斯噪声下做**后验去噪**：

$$
\mathbf{z}^* = \mathbb{E}_{\mathbf{x}_0 \sim p_\text{noisy}(\cdot|\hat{\mathbf{v}})}\left[\mathbf{x}_0\right]
\tag{COMPAT-4}
$$

其中 $p_\text{noisy}(\mathbf{x}_0|\hat{\mathbf{v}}) \propto p(\mathbf{x}_0)\,\exp\left(-\frac{\rho}{2}\|\hat{\mathbf{v}} - \mathbf{x}_0\|^2\right)$。

**这正是扩散模型去噪器（MMSE denoiser）的作用域**。  

由 Tweedie 公式，扩散模型的 score function 给出：

$$
\mathbb{E}[\mathbf{x}_0|\mathbf{x}_0 + \sigma_z \boldsymbol{\epsilon} = \hat{\mathbf{v}}] = \hat{\mathbf{v}} + \sigma_z^2\,\nabla_{\hat{\mathbf{v}}}\log p_{\sigma_z}(\hat{\mathbf{v}})
\tag{COMPAT-5}
$$

代入 $\sigma_z = 1/\sqrt{\rho}$，则 **ADMM z-step 可以直接用扩散模型的一步去噪替代 Chambolle TV**，这是代数上严格成立的等价替换（PnP-ADMM 框架的标准结论）。

### 3.4 GS+SE2 参数化与 STEP 先验的相容性

GS+SE2 表示 $R(\boldsymbol{\theta})$ 在视频空间中定义了一个**低维流形** $\mathcal{M}_\text{SE2}$：

$$
\mathcal{M}_\text{SE2} = \{R(\boldsymbol{\theta}) : \boldsymbol{\theta} \in \Theta\} \subset \mathbb{R}^{T \times H \times W}
$$

扩散先验 $p(\mathbf{x}_0)$ 在视频空间中定义了另一个流形（高密度区域 $\mathcal{M}_\text{diff}$）。

**结合的条件**：需要 $\mathcal{M}_\text{SE2} \cap \mathcal{M}_\text{diff} \neq \emptyset$，即真实视频同时满足：
1. 可以被 SE(2) + 高斯 Splatting 建模（运动先验）
2. 落在扩散模型的高密度区（视觉先验）

在 SPI 的实际场景中（如坦克、刚体目标），SE(2) 近似是合理的，而扩散先验的约束可以帮助补全 TV 无法建模的细节纹理，两个约束是**互补而非矛盾**的。

---

## 四、具体结合方案（两种策略）

### 策略 A：ADMM + 扩散 z-step（最小修改，即插即用）

**动机**：保留 GS+SE2 参数化（效率高，仅 3003 DOF）；用视频扩散先验替换 TV，获得更丰富的时空正则。

**修改点**：仅修改 `solver/admm.py` 的 z-step，其余不变。

**完整算法**：

$$
\text{初始化：} \boldsymbol{\theta}^{(0)},\, \mathbf{z}^{(0)},\, \mathbf{u}^{(0)} = \mathbf{0}
$$

Warmup 阶段（$n < n_\text{warmup}$）：与现有相同，运行 Adam 梯度步。

ADMM 阶段（$n \geq n_\text{warmup}$）：

**$\boldsymbol{\theta}$-step**（$n_\text{inner}$ 步 Adam）：

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \alpha\,\nabla_{\boldsymbol{\theta}}\left[f(\boldsymbol{\theta}) + \frac{\rho}{2}\|R(\boldsymbol{\theta}) - (\mathbf{z} - \mathbf{u})\|^2\right]
\tag{ALG-A1}
$$

**$\mathbf{z}$-step**（扩散去噪，噪声级 $\sigma_z = 1/\sqrt{\rho}$）：

$$
\hat{\mathbf{v}} = R(\boldsymbol{\theta}) + \mathbf{u}
\tag{ALG-A2a}
$$

$$
\mathbf{z} = \text{VideoDiffusionDenoiser}(\hat{\mathbf{v}},\, \sigma_z = 1/\sqrt{\rho})
\tag{ALG-A2b}
$$

具体实现：向 $\hat{\mathbf{v}}$ 加噪到对应扩散时间步 $t^* = t(\sigma_z)$，然后从 $t^*$ 运行 $m$ 步反向扩散（DDIM/DPM-Solver）得到 $\mathbf{z}$。

**$\mathbf{u}$-step**（不变）：

$$
\mathbf{u} \leftarrow \mathbf{u} + R(\boldsymbol{\theta}) - \mathbf{z}
\tag{ALG-A3}
$$

**关键参数**：
- $\sigma_z = 1/\sqrt{\rho}$：随 $\rho$ 增大，去噪强度降低，ADMM 收紧约束
- $\rho$ 调度：仍可使用 `rho_growth` 递增（与现有一致）
- 扩散去噪步数 $m$：$m=1$（单步 Tweedie）到 $m=5$（少步 DDIM）均可

**伪代码**（接口修改）：

```python
# prior/diffusion.py (新增文件)
class VideoDiffusionPrior:
    def proximal(self, v_hat: Tensor, noise_level: float) -> Tensor:
        """
        v_hat: [T,1,H,W]  (R(theta) + u)
        noise_level: 1/sqrt(rho)
        Returns z: [T,1,H,W]
        """
        # 1. 确定扩散时间步 t* s.t. sigma(t*) = noise_level
        t_star = self.sigma_inv(noise_level)
        # 2. 加噪
        eps = torch.randn_like(v_hat)
        z_noisy = v_hat + noise_level * eps
        # 3. 用 score function 一步 Tweedie 去噪（最简单）
        with torch.no_grad():
            score = self.score_net(z_noisy, t_star)
        z = z_noisy + noise_level**2 * score
        return z
```

在 `solver/admm.py` 的 z-step 处替换：

```python
# 原来:  z = self.prior.proximal(v_hat, weight=tv_weight/rho)
# 替换:  z = self.prior.proximal(v_hat, noise_level=1/sqrt(rho))
```

### 策略 B：STEP 作为外层后验采样器，GS+SE2+TV 提供初始化

**动机**：放弃 SE(2) 刚体约束（更通用），用完整 DAPS 做后验采样，GS+SE2 仅提供好的初始视频。

**算法流程**：

```
Phase 1 (GSDiff-SPI warm-start):
  使用现有 SGD/ADMM+TV 运行，得到视频估计 x_hat = R(theta_final)

Phase 2 (STEP posterior refinement):
  z_0^(init) = E(x_hat)          # VAE 编码
  运行 DAPS Algorithm 1：
    - score function: 视频扩散 UNet
    - 数据梯度: ∇_{z_0} log p(y | D(z_0))
                = A^T(A·D(z_0) - y) / sigma_n^2    ← SPI 梯度（已有）
  输出: x_0_final = D(z_0_final)
```

**数据梯度（对 SPI 的精确推导）**：

$$
\nabla_{\mathbf{z}_0}\log p(\mathbf{y}|\mathcal{D}(\mathbf{z}_0)) = \frac{\partial \mathcal{D}(\mathbf{z}_0)^T}{\partial \mathbf{z}_0} \cdot \nabla_{\mathbf{x}_0}\log p(\mathbf{y}|\mathbf{x}_0)
\tag{GRAD-1}
$$

其中：

$$
\nabla_{\mathbf{x}_0}\log p(\mathbf{y}|\mathbf{x}_0)\Bigr|_\text{frame } t = -\frac{1}{\sigma_n^2}\sum_{k: f(k)=t}(\langle \mathbf{P}_k, \mathbf{x}_0^{(t)}\rangle - y_k)\,\mathbf{P}_k
\tag{GRAD-2}
$$

（这就是 SPI 反投影，每帧对应 pattern 的加权叠加，与当前 autograd 计算完全一致。）

通过 `torch.autograd.grad` 可以自动计算穿过 $\mathcal{D}$ 的梯度（若 VAE decoder 参数固定，只需一次反向传播）。

---

## 五、技术挑战与应对

### 5.1 扩散先验的训练数据问题

STEP 在黑洞和 MRI 上使用了域内训练数据（648 个黑洞视频 / 3324 个心脏 MRI 序列）。

**SPI 场景的选项**：
1. **域内扩散先验**：在合成 SPI 视频（SE2 运动场景）上训练，数据生成成本低（可无限生成）
2. **通用视频扩散先验**：用 ModelScope/Stable Video Diffusion 等大模型，利用其通用时空先验
3. **类别特定先验**：若 SPI 目标是特定类别（如人、车辆），用对应视频数据微调

STEP 已证明"有限视频数据（几百个）+ 大量图像"可以高效微调（几小时，单 A100）。这对 SPI 场景是可行的。

### 5.2 GS+SE2 表示与全视频先验的张量对齐

当前 `render_video` 输出 `[T,1,H,W]`（单通道）；扩散模型通常在 RGB 或多通道上训练。

**对齐方案**：
- 使用灰度视频扩散先验（1 通道），与 GS 输出直接对齐
- 或将 `[T,1,H,W]` 沿通道复制为 `[T,3,H,W]` 传入 RGB 扩散模型（推理时兼容）

### 5.3 $\rho$ 调度与扩散噪声级的对应

ADMM 中 $\rho$ 从小变大（`rho_growth = 1.05`），对应扩散噪声级 $\sigma_z = 1/\sqrt{\rho}$ 从大变小：

| ADMM 迭代 | $\rho$ | $\sigma_z = 1/\sqrt{\rho}$ | 物理含义 |
|-----------|--------|---------------------------|---------|
| 初期 | 0.1 | 3.16 | 强去噪，z 自由探索先验 |
| 中期 | 0.5 | 1.41 | 中等约束 |
| 后期 | 2.0 | 0.71 | 弱去噪，z 紧跟 $R(\theta)+u$ |

这和 DAPS 的"退火"策略天然对应：$\rho$ 增大等价于逐步降低扩散噪声级，从大范围探索到精细对齐。

### 5.4 计算代价分析

| 方法 | z-step 代价 | 整体代价 |
|------|------------|---------|
| TV (Chambolle) | 50 次迭代，无 GPU 反向传播 | 基准 |
| 扩散（Tweedie 单步） | 1 次 score net 推断（无梯度） | ~10× Chambolle |
| 扩散（5-step DDIM） | 5 次 score net 推断 | ~50× Chambolle |
| DAPS 完整 | M=60 HMC + ODE 求解 | ~1000× Chambolle |

**推荐**：策略 A 用单步 Tweedie 去噪作为 z-step，计算代价可控（每个 ADMM 外迭代比 TV 慢 10 倍左右，但 ADMM 迭代次数少）。

---

## 六、收敛性分析

### 6.1 PnP-ADMM 收敛条件

将 $\mathbf{z}$-step 替换为扩散去噪器 $D_\sigma$（PnP-Diffusion）后，ADMM 的收敛性需满足：

**充分条件**（来自 PnP-ADMM 理论，Chan 2016 / Xu 2020）：

1. $D_\sigma$ 是某个凸函数的近端算子 ← **扩散 MMSE 去噪器在 $\sigma \to 0$ 时满足此条件**（RED 框架）
2. $D_\sigma$ 是非扩张的（Lipschitz 常数 $\leq 1$）← **需要用双重梯度正则化或约束噪声级**

实践中，PnP-Diffusion（Wu et al. 2024）已证明可以收敛，但需要：
- 使用足够小的 $\eta$（DAPS step size）
- 保持 $\rho$ 单调递增

### 6.2 GS+SE2 的额外稳定性

由于 GS+SE2 $\boldsymbol{\theta}$-step 是一个强结构约束（仅 3003 自由度），它能防止 $\mathbf{z}$ 偏离测量一致区域。这使得 PnP-Diffusion 的探索空间被压缩到更小的流形上，**实际收敛速度更快**，不需要完整 DAPS 的 N=25 外迭代。

---

## 七、结论

**答案：是，两者可以结合，且具有第一性原理上的严格对应关系。**

**核心逻辑链**：

```
SPI 测量方程 y = Ax_0 + n （线性，比 STEP 的黑洞问题更简单）
    ↓
贝叶斯后验 p(x_0|y) ∝ p(y|x_0) p(x_0)
    ↓
ADMM 变量分裂：θ-step（数据保真 + GS+SE2）↔ z-step（先验）
    ↓
z-step 的近端算子 = 扩散去噪 MMSE（代数严格等价）
    ↓
STEP 的时空扩散先验恰好提供 p(x_0) 的建模
    ↓
数据梯度 ∇_{x_0} log p(y|x_0) = -A^T(Ax_0 - y)/σ_n²（可直接计算）
    ↓
结合成立
```

**推荐路径**：
1. **短期**：策略 A（ADMM + 单步 Tweedie 扩散 z-step），最小代码修改，在现有 `solver/admm.py` 的 z-step 处插入，`prior/tv.py` 与 `prior/diffusion.py` 并存，由配置切换
2. **长期**：策略 B（STEP 全 DAPS，GS+SE2 提供初始化），用于高质量后验采样和不确定性量化

**相比纯 TV**，扩散先验预期带来：
- 更好的空间纹理（TV 会过平滑）
- 更强的时间一致性（时空 UNet 学习了帧间依赖）
- 多模态解（对高度欠定问题可采样多个解）

---

## 附录：关键公式索引

| 公式标签 | 含义 |
|---------|------|
| STEP-1 | 视频逆问题统一形式 |
| STEP-3 | 潜在空间测量方程 |
| STEP-6 | 时空模块 α-混合 |
| DAPS-1~5 | DAPS 三步循环 |
| SPI-1~2 | SPI 测量算子（线性） |
| COMPAT-1~5 | 兼容性证明核心公式 |
| ALG-A1~A3 | 策略 A 完整算法 |
| GRAD-1~2 | SPI 数据梯度显式形式 |
