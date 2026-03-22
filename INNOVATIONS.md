# RPMNet-Surgical 创新点总结

> 本文档面向论文撰写，对 RPMNet-Surgical 相对于原始 RPMNet 的三项核心创新进行系统梳理。  
> 每项创新均从**物理/几何直觉 → 数学形式 → 代码实现 → 论文故事线**四个维度展开。

---

## 背景：手术配准的挑战

手术机器人场景中，探针（probe）沿器官表面滑行，采集一段连续的轨迹点云（Partial Source），目标是将其配准到完整的器官模板（Complete Reference）。这是一个典型的 **Partial-to-Complete（P2C）配准**问题，具有三重特殊性：

| 特性 | 传统 P-P 配准 | 手术 P2C 配准 |
|------|:---:|:---:|
| 重叠率 | 对称，≈70% | **极端非对称**，Source 覆盖 Reference 的 10–50% |
| 源点云类型 | 散点 | **有序轨迹**（1-D 曲线嵌入 3-D） |
| 旋转范围 | ≤45° | **≤180°**，存在强旋转歧义 |

原始 RPMNet 的特征、匹配、损失三个模块均为对称设计，无法很好地处理上述约束。本工作在这三个模块分别引入了针对性改造。

---

## 创新一：T-PPF — 切线点对特征（Tangent Point Pair Features）

### 物理/几何直觉

普通 PPF（Point Pair Features）利用**表面法向**描述点对的几何关系。然而探针轨迹是一条 1-D 曲线，其最显著的几何属性不是法向，而是**曲线切向（tangent）**。旋转越大，传统 PPF 与法向之间的夹角会大幅漂移，而切向对于大角度旋转天然具有不变性——旋转同时作用于所有向量，相对夹角保持不变。

> **直觉**：想象拿着一根弯曲的线段，无论怎样旋转，线段上任意两点之间"线段走向"与"连线方向"的夹角保持不变。这正是 T-PPF 所捕捉的量。

### 数学形式

对源点云中每个点 $p_i$，在半径邻域 $\mathcal{N}(p_i)$ 内估计局部切向 $\mathbf{t}_i$，方法为对邻域位移向量的散布矩阵做**幂迭代**取主特征向量：

$$
C = \frac{1}{K}\sum_{p_j \in \mathcal{N}(p_i)} \mathbf{d}_{ij}\mathbf{d}_{ij}^\top, \qquad
\mathbf{t}_i = \text{PowerIteration}(C)
$$

其中 $\mathbf{d}_{ij} = p_j - p_i$。

T-PPF 定义为切向与邻域几何量之间的三个旋转不变角度：

$$
\text{T-PPF}(p_i, p_j) =
\left[\angle(\mathbf{t}_i,\, \mathbf{d}_{ij}),\;\;
       \angle(\mathbf{t}_i,\, \mathbf{n}_j),\;\;
       \angle(\mathbf{t}_i,\, \mathbf{n}_i)\right]
$$

| 分量 | 符号 | 几何含义 |
|------|------|----------|
| 切向 vs. 邻域方向 | $\angle(\mathbf{t}_i, \mathbf{d}_{ij})$ | 点对沿轨迹的"走向" |
| 切向 vs. 邻域法向 | $\angle(\mathbf{t}_i, \mathbf{n}_j)$ | 轨迹对邻域表面的入射角 |
| 切向 vs. 中心法向 | $\angle(\mathbf{t}_i, \mathbf{n}_i)$ | 中心点曲线倾斜度 |

这 3 个角度与标准 PPF 的 4 个角度拼接，形成 7 维局部描述子（当 `--features ppf dxyz xyz tpf` 时为 13 维输入）。

### 代码位置

```
src/models/pointnet_util.py
  ├── estimate_local_tangent()   # 幂迭代估计切向
  └── sample_and_group_multi()   # compute_tpf=True 时计算 T-PPF

src/models/feature_nets.py
  └── FeatExtractionEarlyFusion  # 将 tpf 与 ppf/dxyz/xyz 早融合
```

启用方式：`--features ppf dxyz xyz tpf`

### 论文故事线

> "探针轨迹是一条嵌入三维空间的 1-D 流形，其最本质的局部几何属性是**曲线切向**而非表面法向。我们提出 T-PPF，在每个轨迹点处通过邻域散布矩阵的主特征向量稳健地估计切向，再以切向为锚点构造三个旋转不变角度，与传统 PPF 早期融合。这使模型在 180° 旋转下仍能区分轨迹'正向'与'反向'，从根本上缓解了大旋转歧义。"

---

## 创新二：P2C-Sinkhorn — 非对称松弛偏置（Asymmetric Slack Bias）

### 物理/几何直觉

标准 Sinkhorn 为行和与列和同等地添加一个"垃圾桶"（dust-bin）行与列，允许任意点被判定为外点。这对对称 P-P 配准是合理的，但在 P2C 场景中存在**根本性的不对称**：

- **源点（探针轨迹）**：每个点**必须**落在器官表面上，不可能是外点；
- **参考点（模板表面）**：绝大多数点**合理地未被探针访问**，对应着高外点率。

若对源列与参考列施加相同的松弛容量，模型会因训练信号不明确而将大量探针点错误地扔进垃圾桶。

> **直觉**：给垃圾桶"加门槛"——源点侧的垃圾桶门口竖一块高墙（负偏置），让每个探针点都尽力找到匹配，而参考侧的垃圾桶门口保持畅通。

### 数学形式

在 Sinkhorn 迭代开始前，对对数亲和矩阵 $\log \mathbf{A}$ 扩展后的源松弛列施加一个可调负偏置 $b < 0$：

$$
\widetilde{A}_{j,K+1} \leftarrow A_{j,K+1} + \exp(b), \qquad j = 1,\ldots,J \quad (\text{源点行})
$$

等价地在对数域：

$$
\log \widetilde{\alpha}_{j,K+1} = \log \alpha_{j,K+1} + b
$$

其中 $b = $ `src_slack_bias` $< 0$（默认 $-2.0$，即将源外点概率缩小约 $e^2 \approx 7.4$ 倍）。参考松弛行 $\log \alpha_{J+1, k}$ 不作修改。

Sinkhorn 随后迭代，自然地将压缩后的源外点质量重新分配到有效匹配位置。

### 代码位置

```
src/models/rpmnet.py
  └── sinkhorn()          # src_slack_bias 参数
      if src_slack_bias != 0.0:
          log_alpha_padded[:, :-1, -1] += src_slack_bias   # 源行对应的外点列

src/arguments.py          # --src_slack_bias（默认 -2.0）
src/models/rpmnet.py
  └── RPMNetSurgical.__init__()  # 读取并赋值 self.src_slack_bias
```

启用方式：`--src_slack_bias -2.0`

### 论文故事线

> "P2C 配准的核心约束是：每个探针点**必须**有对应的表面匹配，而大多数表面点**合法地**没有对应的探针点。原始 Sinkhorn 的对称松弛无法编码这一不对称先验。我们提出 P2C-Sinkhorn，在迭代前对源侧垃圾桶列施加负对数偏置，使探针点向垃圾桶的路径在概率意义上指数级地更难走通。这一单参数修改不引入任何额外可学习权重，却从根本上改变了匹配质量分布，在极端低重叠率（10–30%）下显著提升配准精度。"

---

## 创新三：SP-Loss — 源优先内点损失（Source-Priority Inlier Loss）

### 物理/几何直觉

原始 RPMNet 的内点正则化项对源点和参考点施加**相同权重**，鼓励双方的行/列和都尽量接近 1。这在 P2C 场景中带来训练信号冲突：

- 参考点的低行和（大量未匹配）是**物理正确**的，若强行惩罚会扭曲模型；
- 源点的低列和（探针点未被匹配）是**需要纠正**的，需要强烈惩罚。

将两者用同一权重混合，模型会在二者之间妥协，最终两项约束都不能得到有效执行。

> **直觉**："源点全部要匹配"是一个硬约束，"参考点多数未匹配"是一个软约束。训练信号中也应如此：前者权重高，后者权重低。

### 数学形式

将内点正则化项拆分为源优先项与参考宽松项：

$$
\mathcal{L}_{\text{SP}} = w_{\text{src}} \underbrace{\mathbb{E}\!\left[1 - \sum_k P_{jk}\right]}_{\text{源外点惩罚}}
                        + w_{\text{ref}} \underbrace{\mathbb{E}\!\left[1 - \sum_j P_{jk}\right]}_{\text{参考外点惩罚}}
$$

其中 $w_{\text{src}} \gg w_{\text{ref}}$（推荐 $w_{\text{src}} = 0.1,\; w_{\text{ref}} = 0.001$，比例 100:1）。

原始 RPMNet 等价于 $w_{\text{src}} = w_{\text{ref}} = w_{\text{inliers}}$。

### 代码位置

```
src/train.py  compute_losses()
  # SP-Loss
  wt_src = _args.wt_src_inliers if _args.wt_src_inliers is not None else _args.wt_inliers
  wt_ref = _args.wt_ref_inliers if _args.wt_ref_inliers is not None else _args.wt_inliers
  ref_outliers_strength = (1 - Σ_j P[i,j,k]) * wt_ref   # 参考点外点项
  src_outliers_strength = (1 - Σ_k P[i,j,k]) * wt_src   # 源点外点项

src/arguments.py   # --wt_src_inliers, --wt_ref_inliers（可分别设置）
```

启用方式：`--wt_src_inliers 0.1 --wt_ref_inliers 0.001`

### 论文故事线

> "P2C 场景的内点率天然不对称：探针点应全匹配（近 100%），表面点仅少量匹配（10–50%）。将两者混为一个权重进行正则化会给模型发出矛盾的训练信号。我们提出 SP-Loss，将内点惩罚分解为源优先（高权重）和参考宽松（低权重）两项，在损失函数层面显式编码 P2C 不对称性。SP-Loss 与 P2C-Sinkhorn 协同作用：前者在前向匹配中限制外点分配，后者在反向传播中放大纠正信号。"

---

## 三项创新的协同关系

```
输入: 探针轨迹(Source) + 器官模板(Reference)
           │
  ┌────────▼────────┐
  │  特征提取层      │  ← 创新一 T-PPF
  │  xyz + dxyz     │    切向锚定的旋转不变特征
  │  + ppf + tpf    │    缓解 180° 旋转歧义
  └────────┬────────┘
           │ 亲和矩阵
  ┌────────▼────────┐
  │  匹配层          │  ← 创新二 P2C-Sinkhorn
  │  非对称 Sinkhorn │    源侧垃圾桶加负偏置
  │  src_slack_bias  │    强制探针点全部找到匹配
  └────────┬────────┘
           │ 置换矩阵
  ┌────────▼────────┐
  │  损失函数        │  ← 创新三 SP-Loss
  │  MAE(注册误差)   │    源外点高权重惩罚
  │  + SP-Loss       │    参考外点低权重惩罚
  └─────────────────┘
```

三项创新分别从**特征空间**、**匹配约束**、**训练信号**三个维度，一致地编码"每个探针点必须匹配，大多数表面点合法未匹配"这一物理约束，形成相互增强的整体设计。

---

## 推荐命令行（完整复现）

```bash
python train.py \
    --dataset_type surgical \
    --method rpmnet_surgical \
    --features ppf dxyz xyz tpf \   # T-PPF
    --src_slack_bias -2.0 \          # P2C-Sinkhorn
    --wt_src_inliers 0.1 \           # SP-Loss 源权重
    --wt_ref_inliers 0.001 \         # SP-Loss 参考权重
    --angle_range 180 \
    --coverage_ratio 0.5
```

---

## 与现有方法的对比定位

| 方法 | 特征类型 | Sinkhorn 是否非对称 | Loss 是否区分源/参考 | P2C 支持 |
|------|:--------:|:-------------------:|:--------------------:|:--------:|
| RPMNet (原始) | PPF + dxyz + xyz | ✗（对称） | ✗（统一权重） | ✗ |
| **RPMNet-Surgical（本工作）** | **+T-PPF** | **✓（P2C-Sinkhorn）** | **✓（SP-Loss）** | **✓** |
| OverlapPredNet | 点特征 + 重叠预测 | ✗ | 部分 | 部分 |
| RoReg | 旋转等变特征 | ✗ | ✗ | ✗ |

---

*文档生成于代码库版本 `RPMNet_surgical/src/models/rpmnet.py`，对应分支 `copilot/rpmnet-architecture-innovation`。*
