# CLAUDE.md — Claude Code 完全指南：从零到精通

> 本文件同时作为你学习Claude Code的教程，
> 以及放在项目根目录供Claude Code读取的项目说明文件。

---

## 第一部分：什么是Claude Code

Claude Code是Anthropic出品的**命令行AI编程助手**。
它不是网页聊天（那是claude.ai），而是在你的终端（Terminal）中运行的工具。

它能做什么：
- 读取并理解你整个项目的代码
- 帮你写代码、修bug、做重构
- 运行命令、创建文件、管理Git
- 直接在终端中对话，不需要复制粘贴代码

和网页版Claude的区别：
- 网页版：你手动复制代码给它看
- Claude Code：它**直接看你的文件系统**，比你自己找文件还快

---

## 第二部分：安装（Windows）

### 2.1 前置要求

1. **Anthropic账号**：必须是 Claude Pro、Max 或 Enterprise 订阅
   （免费版不能用Claude Code）
2. **Git for Windows**：去 https://git-scm.com 下载安装

### 2.2 安装Claude Code

打开 PowerShell（不需要管理员权限），运行：

```powershell
# 方法1：原生安装器（推荐，不需要Node.js）
irm https://claude.ai/install.ps1 | iex

# 方法2：如果你有Node.js
npm install -g @anthropic-ai/claude-code
```

验证安装：
```powershell
claude --version
```

### 2.3 首次登录

```powershell
claude
```

它会打开浏览器让你登录Anthropic账号。登录后自动返回终端。

---

## 第三部分：基本使用

### 3.1 启动会话

```powershell
# 进入你的项目目录
cd D:\Research\v4

# 启动Claude Code
claude
```

Claude Code会扫描当前目录，了解你的项目结构。

### 3.2 基本对话

启动后你就在一个交互式对话中。直接打字就行：

```
> 帮我看看train.py里的ADMM循环有没有问题

> 解释一下gaussian2d.py中render方法的工作原理

> 帮我在configs/default.yaml里加一个新的配置项
```

Claude Code会直接读取你的文件，不需要你复制粘贴。

### 3.3 退出

```
> /exit
```

或者按 Ctrl+C。

### 3.4 恢复上次对话

```powershell
claude -c    # continue上一次的对话
```

---

## 第四部分：核心命令

### 4.1 斜杠命令（在Claude Code会话内使用）

| 命令 | 作用 |
|------|------|
| `/help` | 显示所有可用命令 |
| `/init` | 生成CLAUDE.md项目说明文件 |
| `/clear` | 清空当前对话上下文 |
| `/exit` | 退出Claude Code |
| `/bug` | 报告一个bug给Anthropic |
| `/model` | 切换AI模型 |
| `/compact` | 压缩对话历史以节省上下文 |

### 4.2 文件引用

用 @ 符号引用文件：

```
> 看看 @gsdiff/solver/admm.py 里的z_step方法

> 对比 @gsdiff/solver/admm.py 和 @gsdiff/solver/sgd.py 的区别
```

### 4.3 运行Shell命令

用 ! 前缀运行Shell命令：

```
> !python train.py --solver admm
> !git status
> !pip install torch
```

或者直接让Claude帮你运行：

```
> 帮我运行train.py看看结果
```

（Claude会问你是否允许执行命令，你确认就行）

### 4.4 命令行参数

```powershell
# 直接问一个问题（不进入交互模式）
claude "解释一下这个项目的整体架构"

# 指定模型
claude --model opus       # 最强模型，复杂任务用
claude --model sonnet     # 日常任务用（默认）

# 添加额外目录的访问权限
claude --add-dir ../other_project

# 输出为JSON格式
claude -p "列出所有Python文件" --output-format json
```

---

## 第五部分：CLAUDE.md文件（最重要的概念）

### 5.1 什么是CLAUDE.md

CLAUDE.md是放在项目根目录的特殊文件。
Claude Code每次启动都会读它，用来了解你的项目。
相当于给AI一份"项目说明书"。

### 5.2 放什么内容

```markdown
# Project: GSDiff-SPI

## 项目描述
动态单像素成像的重建算法：2D Gaussian + SE(2)运动 + ADMM/TV

## 技术栈
- Python 3.10+, PyTorch 2.0+
- 数值计算：numpy, scipy

## 项目结构
- gsdiff/scene/     场景表征（2D高斯）
- gsdiff/motion/    运动模型（SE2）
- gsdiff/solver/    求解器（ADMM, SGD）
- gsdiff/data/      数据生成
- train.py          训练入口

## 代码规范
- 所有loss使用mean reduction
- 测量值必须z-score归一化
- ADMM符号遵循Boyd 2011 scaled form

## 常用命令
python train.py --solver admm
python train.py --solver sgd

## 当前Phase
Phase 1: 2DGS + SE(2) + ADMM/TV
下一步: Phase 2 — 替换TV为扩散先验
```

### 5.3 为什么重要

没有CLAUDE.md：Claude每次都要重新"猜"你的项目在做什么。
有了CLAUDE.md：Claude从第一秒就知道项目结构和编码规范。

---

## 第六部分：Git 基础（Claude Code 需要 Git）

### 6.1 什么是Git

Git是版本控制系统——记录你代码的每一次修改。
就像游戏存档：你可以随时回到之前的版本。

### 6.2 初始化Git仓库

```powershell
cd D:\Research\v4
git init
git add .
git commit -m "V4: initial commit with ADMM/SGD solver"
```

### 6.3 基本Git命令

```powershell
git status              # 看有哪些文件被修改了
git add .               # 把所有修改加入暂存区
git commit -m "描述"    # 提交一次"存档"
git log --oneline       # 查看历史
git diff                # 查看当前修改了什么
```

### 6.4 Claude Code + Git 联动

在Claude Code中：

```
> 帮我commit当前的修改，写一个合适的commit message
> 查看最近5次commit的内容
> 帮我revert上一次的修改
```

Claude会调用git命令帮你完成。

---

## 第七部分：GitHub 基础

### 7.1 什么是GitHub

Git在你本地电脑上记录版本。
GitHub是云端的Git仓库托管服务——让别人也能看到你的代码。

### 7.2 创建GitHub账号

1. 去 https://github.com 注册
2. 创建一个新仓库（New Repository）
3. 仓库名填 `gsdiff-spi`
4. 选 Public 或 Private
5. 不要勾选 "Initialize with README"（因为你本地已经有了）

### 7.3 把本地代码推送到GitHub

```powershell
# 把GitHub仓库添加为远程地址
git remote add origin https://github.com/你的用户名/gsdiff-spi.git

# 推送代码
git push -u origin main
```

### 7.4 日常流程

```powershell
# 修改代码后
git add .
git commit -m "fix: correct ADMM sign convention"
git push
```

---

## 第八部分：Claude Code 工作流（日常使用）

### 8.1 开始新的工作会话

```powershell
cd D:\Research\v4
claude
```

### 8.2 典型工作流程

**场景1：修bug**
```
> 我运行 python train.py 时出现了这个错误：[粘贴错误信息]
> 帮我找到原因并修复
```

**场景2：加新功能**
```
> 我想在data/simulation.py中添加一个新的运动类型 "spiral"
> 物体沿螺旋线运动，参数是半径和角速度
> 请帮我实现，并在configs/default.yaml中加上相应配置
```

**场景3：代码审查**
```
> 仔细检查 @gsdiff/solver/admm.py 中的数学是否正确
> 特别注意ADMM的符号约定是否和Boyd 2011一致
```

**场景4：写论文**
```
> 帮我写Method section的草稿，描述我们的2DGS + SE(2) + ADMM框架
> 用学术英语，适合投稿Optics Express
```

**场景5：做实验**
```
> 帮我设计一组消融实验：
> 1. 不同数量的Gaussians（64, 128, 256, 512）
> 2. 不同TV权重
> 生成对应的config文件和运行脚本
```

### 8.3 高效使用技巧

1. **具体而非模糊**：
   - 差："帮我改改代码"
   - 好："在admm.py的theta_step中，把Adam换成LBFGS，保持其他不变"

2. **给上下文**：
   - 差："为什么loss不下降"
   - 好："loss_data从0.5降到0.3就卡住了，velocity一直在[0.1, 0.2]附近不动，GT是[3.0, 2.0]。是不是lr_motion太小？"

3. **分步做**：
   不要一次让Claude做太多事。先完成一步，验证正确后再下一步。

4. **用CLAUDE.md**：
   把你的编码习惯、数学约定、重要决策写进去。

---

## 第九部分：进阶功能

### 9.1 模型切换

```
> /model opus     # 复杂推理用Opus（更慢但更聪明）
> /model sonnet   # 日常编码用Sonnet（更快）
> /model haiku    # 简单任务用Haiku（最快最便宜）
```

### 9.2 管道（Pipe）

把其他命令的输出传给Claude：

```powershell
# 让Claude解释测试失败的原因
python -m pytest tests/ 2>&1 | claude -p "解释这些测试失败的原因"

# 让Claude写release notes
git log --oneline -n 10 | claude -p "根据这些commit写release notes"

# 让Claude分析profiling结果
python -m cProfile train.py 2>&1 | claude -p "这个程序哪里最慢？怎么优化？"
```

### 9.3 非交互模式（适合脚本）

```powershell
# -p 参数：直接给prompt，输出结果后退出
claude -p "列出gsdiff目录下所有.py文件的行数"

# 结合管道使用
cat train.py | claude -p "找出这个文件中所有潜在的bug"
```

### 9.4 MCP（Model Context Protocol）

MCP让Claude Code连接外部服务：

```powershell
# 添加GitHub MCP（让Claude直接操作你的GitHub仓库）
claude mcp add github

# 添加文件系统MCP
claude mcp add filesystem -- npx @modelcontextprotocol/server-filesystem /path/to/project
```

---

## 第十部分：本项目的CLAUDE.md配置

下面这段是给Claude Code读的项目说明。
如果你把这个文件放在项目根目录，Claude Code每次启动都会读它。

---

# GSDiff-SPI Project Context

## What this project does
Dynamic single-pixel imaging reconstruction using:
- 2D Gaussian Splatting for scene representation
- SE(2) rigid-body motion model
- ADMM with TV regularization (Phase 1)
- Future: spatiotemporal diffusion prior (Phase 2)

## Architecture
```
gsdiff/
├── scene/gaussian2d.py    # Canonical 2D Gaussian rendering (differentiable)
├── motion/se2.py          # SE(2) transform of Gaussian params (Theorem 3.1)
├── forward/spi.py         # Physics: scene+motion → video → measurements
├── prior/tv.py            # Chambolle TV proximal operator
├── solver/admm.py         # ADMM outer loop (Boyd 2011 scaled form)
├── solver/sgd.py          # Direct Adam baseline
├── data/simulation.py     # Synthetic data generation (multiple motion types)
├── data/patterns.py       # Bernoulli / Gaussian / S-matrix patterns
├── data/dgi.py            # DGI baseline reconstruction
└── utils.py               # Seed, config, metrics, I/O
```

## Critical math conventions
- ADMM signs follow Boyd et al. 2011 §3.1.1 (SCALED form):
  - θ-step target: z − u (NOT z + u)
  - z-step input: R(θ) + u (NOT R(θ) − u)
  - u-step: u ← u + R(θ) − z
- Measurements are z-scored INDEPENDENTLY (pred and target each normalized separately)
- All losses use mean reduction (NOT sum)
- SE(2) rotation center is ((H-1)/2, (W-1)/2) to match scipy.ndimage.rotate

## How to run
```bash
python train.py --solver admm    # ADMM with TV prior
python train.py --solver sgd     # SGD baseline
```

## Code style
- PyTorch tensors with explicit shape comments
- Config via YAML (configs/default.yaml)
- No magic numbers: all hyperparams in config
- Every module must be replaceable (for Phase 2 extensions)

## Current status
- Phase 1 complete: ADMM/TV working, PSNR ~15-22 dB on 28×28 test images
- SGD baseline working: PSNR ~17-22 dB
- Next: Phase 2 = replace TV with spatiotemporal diffusion prior (STEP/DAPS)
