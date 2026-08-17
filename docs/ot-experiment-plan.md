# RCSL 最优传输实验方案

## 1. 文档状态

- 状态：待实现
- 基线分支：`feature/mnn-mining`和`baseline`
- 实验分支：`feature/ot-experiment`
- 首要数据集：Flickr30K (`f30k_precomp`)
- 首要目标：验证最优传输（Optimal Transport, OT）能否提高半配对图文检索中的伪配对质量，并最终提高检索指标。

## 2. 背景与问题

当前 RCSL 在前 `MineEpoch` 个 epoch 使用已知配对样本训练图像、文本编码器，之后进入 mining 阶段：

1. `data.py` 通过 `shuffle_inx` 打乱未配对部分的图像索引，保留合成实验中的真实对应关系用于评估。
2. `train.py::UpdateMemoryBank` 对全局图像、文本嵌入计算余弦相似度，并分别选择 image-to-text 和 text-to-image 的 top-1 最近邻。
3. 只有互为最近邻（Mutual Nearest Neighbor, MNN）的候选被标记为接受；其他候选仍通过 `rejected_weight_floor` 获得较低训练权重。
4. `data.py::__getitem__` 从 memory bank 中取一个伪图像或伪文本，`model.py::robust_mining_loss` 将构造后的对角线视为正样本。

在当前数据构造中，`paired_length` 按去重后的图像数量截断，而不是按 caption 数量截断；同一图像组的全部 `im_div` 条 caption 共享一次 `shuffle_inx` 置换。因此首轮 OT 可以把未配对部分建模为完整的一图多文节点组，但实现必须通过断言验证该条件，不能仅依赖 Flickr30K 的数据惯例。

该策略的主要限制是 top-1 与 MNN 都是局部、近似一对一的选择。Flickr30K/COCO 的数据结构通常是一张图像对应五条描述，MNN 最多只能让一条描述与该图像形成互相 top-1，可能造成高精度但低覆盖率的伪配对。OT 可以在全局容量约束下产生一对多的软运输计划，因此适合作为 memory-bank mining 的替代方案。

## 3. 研究问题与假设

### RQ1：OT 是否能改善伪配对本身？

**H1：** 在相同 encoder checkpoint 和候选集合下，OT 相比当前 MNN 能提高伪配对的 precision-coverage 曲线，或将更多运输质量分配给真实图文对应关系。

**否证条件：** 在固定 checkpoint 的 mining-only 评估中，OT 相对 MNN/NN 的 precision-coverage AUC 和 GT assigned mass 均未提高至少 1 个百分点，或在 50% coverage 处 precision 更低。

### RQ2：更好的运输计划是否会转化为检索收益？

**H2：** 使用 OT 伪配对继续训练后，验证集与测试集 rSum、I2T R@1、T2I R@1 相比 MNN 基线有稳定提升。

**否证条件：** 在至少三个随机种子上，OT 相对 MNN 的平均 rSum 改变量不大于 0，或者 I2T/T2I 任一方向的平均 R@1 下降至少 1.0 点。

### RQ3：收益是否来自 OT，而不是更多候选或不同权重？

**H3：** 在候选数量、训练步数、encoder 初始化和置信度加权方式一致时，OT 仍优于 top-k NN 和 MNN + 相似度加权。

**否证条件：** 简单 top-k NN 或相似度加权达到相同或更好的效果，说明 OT 的全局约束没有提供额外价值。

## 4. OT 形式化

设未配对图像和文本的归一化嵌入分别为 $x_i$ 和 $y_j$，相似度与运输成本为：

$$
S_{ij}=x_i^\top y_j, \qquad C_{ij}=1-S_{ij}.
$$

平衡熵正则 OT 求解：

$$
\min_{P\ge 0}\langle P,C\rangle
+\varepsilon\sum_{i,j}P_{ij}(\log P_{ij}-1),
$$

满足：

$$
P\mathbf{1}=a,\qquad P^\top\mathbf{1}=b.
$$

若有 $N$ 张图像、每张图像五条描述，则使用均匀边缘分布 $a_i=1/N$、$b_j=1/(5N)$，可以表达一张图像向五条文本分配质量。

OT 图中的图像节点必须是去重后的图像，文本节点是原始 caption。未配对池按图像组构造：某张图像进入未配对池时，它的全部 `im_div` 条 caption 一起进入。运行前断言 caption 数量、图像数量与 `dataset.im_div` 一致；若数据集不满足固定一图多文结构，则从实际 group size 构造边缘分布，不能继续假设 $5N$。

全量实验优先采用非平衡 OT（Unbalanced OT, UOT）或带 dustbin 的部分 OT。原因是 top-k 稀疏候选可能漏掉真实匹配；强制满足严格边缘约束会把质量分给错误候选。UOT 通过边缘 KL 惩罚允许部分质量不被匹配：

$$
\min_{P\ge0}\langle P,C\rangle+\varepsilon\operatorname{KL}(P\|a\otimes b)
+\rho\operatorname{KL}(P\mathbf{1}\|a)
+\rho\operatorname{KL}(P^\top\mathbf{1}\|b).
$$

首版运输计划在 memory-bank 更新阶段用 `torch.no_grad()` 计算，不让梯度穿过 Sinkhorn。这样可以先隔离伪配对质量的影响，再决定是否研究可微 OT loss。

## 5. 实验变量

### 5.1 对照方法

| ID | 方法 | 目的 |
| --- | --- | --- |
| B0 | 当前 MNN + `rejected_weight_floor` | 主基线 |
| B1 | 双向 top-1 NN，无 MNN 过滤 | 检验 MNN 过滤本身的贡献 |
| B2 | top-k NN + 局部 `softmax(S/epsilon)` 权重，源置信度固定为 1 | 与 O3 使用相同候选边、回写数、温度和 loss，只用局部相似度分配边权重 |
| O1 | 稠密平衡 Sinkhorn（数据子集） | 验证 OT 假设与实现正确性 |
| O2 | 稀疏候选 + UOT + top-1 回写 | 与现有 memory-bank schema 最接近的 OT 版本 |
| O3 | 稀疏候选 + UOT 条件权重 + top-k，源置信度固定为 1 | 相对 B2 只改变边权重的全局分配方式，隔离 OT 约束 |
| O4 | O3 + OT 源置信度 | 检验 UOT 拒绝质量和集中度的额外价值，作为完整方法 |

### 5.2 固定条件

- 所有 mining 方法从同一个 `MineEpoch - 1` checkpoint 开始。
- 同一比较组使用相同的数据打乱、batch 顺序、训练 epoch、学习率和 memory-bank 更新周期。
- 已知配对样本继续使用现有监督损失，不参与未配对 OT 的边缘质量竞争。
- 训练集中的合成 GT mining 指标只用于检验 H1 和故障诊断，不参与 OT 超参数选择。超参数根据预注册默认值和验证集检索指标选择；测试集只用于最终确认。
- 实验路径中关闭当前 `train.py` 每个 epoch 的测试集评估，只在方法和超参数冻结后对最终 checkpoint 运行一次测试，避免隐式测试集选择。
- 每次运行记录 git commit、完整配置、随机种子和起始 checkpoint 标识。
- B2 与 O3 使用完全相同的候选边、`pseudo_topk`、`epsilon` 和 soft-positive loss，源置信度都固定为 1。B2 使用 $\operatorname{softmax}(S/\varepsilon)$，O3 使用相同 $\varepsilon$ 的 UOT 条件分布 $q$；截取 `pseudo_topk` 后均重新归一化为和 1。两个方向的 loss 分别按有效源样本数归一化，确保两者唯一设计差异是局部与全局边权分配。O4 再单独加入 OT 源置信度。

### 5.3 首轮搜索空间

| 参数 | 候选值 | 说明 |
| --- | --- | --- |
| `ot_candidate_k` | 16, 32, 64 | 每个节点进入稀疏 OT 图的候选数量 |
| `ot_epsilon` | 0.03, 0.05, 0.10 | 熵正则强度，成本范围约为 $[0,2]$ |
| `ot_rho` | 0.5, 1.0, 2.0 | UOT 边缘约束强度 |
| `ot_pseudo_topk` | 1, 5 | 每个样本回写的伪配对数量 |
| `ot_confidence` | row_mass, concentration, mass×concentration | 伪配对权重定义 |
| `memory_update_interval` | 5 | 首轮保持当前默认值，不与 OT 参数联合搜索 |

不进行完整笛卡尔积。先固定 `candidate_k=32, epsilon=0.05, rho=1.0, pseudo_topk=1` 完成端到端验证，再对单个参数做局部搜索。

这里 `candidate_k` 表示求解 OT 前每个节点保留的稀疏候选边数量，`pseudo_topk` 表示求解后实际回写 memory bank 并参与训练的正样本数量，两者不得混用。

### 5.4 固定运行协议

- 首轮沿用 `train.sh` 的 `batch_size=256`、`tau=0.03`、`MineEpoch=25`、`MaxEpoch=40` 和 `memory_update_interval=5`。
- shared checkpoint 固定为 learning 阶段最后一个 epoch（epoch 24）的 checkpoint，而不是各方法独立选择的 best checkpoint。
- mining 阶段最终模型按验证集 rSum 选择；并列时选择更早 epoch。测试集只评估这个冻结后的 checkpoint。
- Flickr30K 使用仓库当前 `fold5=False` 的完整 test split 评估方式；如增加 1K-fold 结果，必须单独标注，不能与主结果混报。
- 首版求解器使用 log-domain FP32、`max_iter=200`、`tol=1e-3`、相似度 block size 1024；连续 5 次迭代的目标值相对变化和 log-scaling 最大变化都低于 `tol` 时停止。UOT 的边缘偏差只作诊断，不以逼近平衡边缘作为收敛条件。任何变更都写入运行配置。
- KL 使用广义定义 $\operatorname{KL}(p\|q)=\sum_i[p_i\log((p_i+10^{-12})/(q_i+10^{-12}))-p_i+q_i]$。
- 每次运行记录 PyTorch/CUDA 版本、GPU 型号、可用显存和实际 wall-clock time。
- 正式实现只依赖现有 PyTorch。若阶段 A 安装参考求解器，必须在项目虚拟环境中安装，不写入运行时依赖。

## 6. 实施设计

### 6.1 代码边界

建议新增独立模块 `ot_mining.py`，避免将求解器继续堆入 `train.py`。模块接口应只接收嵌入、未配对 mask 和配置，返回可序列化的 memory bank 数据。

计划修改点：

- `opts.py`：增加 `--mining_method` 及 OT 参数；默认值保持 `mnn`，保证旧实验可复现。
- `train.py`：抽取全局嵌入计算；根据 `mining_method` 调用 MNN 或 OT miner；记录耗时、显存和 mining 指标。
- `data.py`：使 memory bank 支持 top-k 索引和权重；top-1 模式保持兼容。
- `model.py`：top-k 实验中按 OT 质量计算 soft positive loss；首个 O2 实验复用现有 `robust_mining_loss`。
- `train.sh`：增加独立的 OT 实验命名和参数，避免覆盖 MNN 结果。

实现时同时消除新路径中的两个硬编码假设：嵌入维度从 `model.opt.embed_size` 读取，图文数量关系从 `dataset.im_div` 读取，不在 OT 代码中写死 `1024` 或 `5`。

### 6.2 候选图构建

完整 Flickr30K 训练集的图文成本矩阵可能包含数十亿元素，不能常驻显存。全量方案采用：

1. 以 block matrix multiplication 计算相似度。
2. 分别保留 I2T top-k 和 T2I top-k。
3. 取双向候选边的并集，避免单方向截断。
4. 首版从图中排除已知配对节点；anchor 方案只作为后续消融，避免首轮存在两种语义不同的实现。
5. 在稀疏候选图上运行 log-domain UOT；必要时按连通分量求解。

每个 checkpoint 的 blockwise top-k 结果只计算一次并冻结：B0/B1 从中读取双向 top-1，B2/O2/O3/O4 从同一个双向并集候选图读取边，避免候选检索差异混入方法比较。

若稀疏 UOT 的目标值或 log-scaling 长期不能收敛，优先检查候选图连通性、增加 dustbin、调整 $\rho$，或在小连通分量回退到稠密 UOT，而不是通过增大 `candidate_k` 无限制增加显存。

### 6.3 置信度与 memory bank

UOT 的“总共愿意匹配多少质量”和“已匹配质量集中在哪些边”必须分开。对源节点 $i$ 定义相对行质量 $m_i$、条件分布 $q_{ij}$ 和归一化集中度 $c_i$：

$$
m_i=\operatorname{clip}\left(\frac{\sum_jP_{ij}}{a_i+10^{-12}},0,1\right),
\qquad q_{ij}=\frac{P_{ij}}{\sum_jP_{ij}+10^{-12}},
$$

$$
c_i=1+\frac{\sum_jq_{ij}\log(q_{ij}+10^{-12})}{\log d_i},
$$

其中 $d_i$ 是双向候选并集后节点 $i$ 的实际支持集大小；当 $d_i\le1$ 时定义 $c_i=1$。T2I 方向对列对称计算。

首轮比较三种权重：

- `row_mass`：$m_i$，保留 UOT 拒绝匹配的信号。
- `concentration`：$c_i$，只反映条件分布的集中程度。
- `mass×concentration`：$m_ic_i$，同时要求足够质量和较低熵；作为首选默认值。

一条边的最终权重为源节点置信度乘以 $q_{ij}$。截取 `pseudo_topk` 后，在保留边上重新归一化 $q$ 再乘源置信度。因此总运输质量极小但只有一条候选边的节点不会因“分布集中”而自动得到高置信度。

O4 的源节点置信度裁剪到 $[w_{min},1]$，边权重再由归一化的 $q_{ij}$ 分配，因此一个源节点的边权重总和等于它的源置信度。阶段 C 必须比较 `w_min=0` 与当前 `rejected_weight_floor=0.5`；0.25 仅作为后续局部消融。B2/O3 的源置信度固定为 1，不受该参数影响。

## 7. 分阶段实验

### 阶段 A：实现与小规模正确性

- 从固定 checkpoint 抽取 512 或 1,000 张未配对图像及其全部描述。
- 在 CPU/GPU 上构建完整成本矩阵，运行稠密平衡 Sinkhorn。
- 对同一小矩阵运行稠密 UOT，并与稀疏候选覆盖全部边时的 UOT 结果对齐。
- 检查非负性、有限值、目标值、行列边缘残差和迭代收敛。
- 增加缺失真实边、孤立节点、非连通候选图和极端相似度的测试。
- 用合成 GT 统计 `GT transport mass`，并可视化少量运输矩阵。
- 对小矩阵用高精度参考实现或线性规划结果做数值抽查；正式代码不因此增加运行时依赖。

通过标准：无 NaN/Inf；平衡 OT 边缘最大绝对误差小于 $10^{-3}$；稠密与全边稀疏 UOT 的相对目标值误差小于 $10^{-3}$、归一化运输计划 L1 误差小于 $10^{-2}$；固定输入重复运行结果一致；打乱文本后运输计划按相同置换变化；缺边和非连通输入能返回有限结果或明确错误。

### 阶段 B：固定 checkpoint 的 mining-only 比较

- 数据：Flickr30K，`paired_length=1000`，seed=42。
- 编码器固定，不继续训练。
- O1 只在与阶段 A 相同的数据子集上作为数值参考；全量数据比较 B0、B1、B2、O2、O3，以及 `w_min=0/0.5` 的 O4。
- 绘制 precision-coverage、GT assigned mass-coverage 和置信度校准曲线。

门控条件：O2、O3 或 O4 至少一个方法相对 $\max(\mathrm{B0},\mathrm{B2})$ 的 precision-coverage AUC 或 GT assigned mass 提高至少 1 个百分点，并且在 50% coverage 处的 precision 不降低，才进入阶段 C。O2 失败但 O3 通过时，结论应限定为“一对多软计划有效”；只有 O4 通过时，结论应限定为“OT 源置信度有效，而 OT 条件边权尚无证据”。若三者均未通过，停止大规模训练，先分析候选召回率、UOT 收敛和置信度校准。

### 阶段 C：单设置端到端验证

- 数据：Flickr30K，`paired_length=1000`，seed=42。
- 从完全相同的 learning-stage checkpoint 分叉运行 B0、B2、O2、O3、O4。O4 分别使用 `w_min=0` 和 0.5；这是隔离 OT confidence 与检验拒绝能力的必做消融，不因资源限制省略。
- 保持 mining epoch 数与 memory-bank 刷新次数一致。
- 先使用固定默认 OT 参数，不在测试集上调参。

门控条件：O2/O3/O4 至少一个方法的验证集 rSum 高于 B0，且 I2T/T2I R@1 没有任一方向超过 1.0 点的退化，才进入多种子实验。

### 阶段 D：主要结果与鲁棒性

- `paired_length`：500、1,000、5,000。
- 随机种子：42、3407、2026。
- 方法：B0、B2、阶段 C 最佳 OT 方法。
- 每个 seed 与 paired setting 共享对应的 learning-stage checkpoint。
- 报告平均值、标准差和 OT-MNN 的配对差值。
- 三种子阶段不作统计显著性声明，只报告均值、标准差和每个 seed 的配对差值，不只报告最佳 seed。
- 若结论依赖小幅收益或种子间方向不一致，扩展到至少五个 seed，再以 seed 级 OT-MNN 配对差值为重采样单位报告 bootstrap 95% 置信区间；该区间仍标为探索性，不替代逐 seed 结果。

资源允许后，再在 COCO 上用冻结的 Flickr30K 最优超参数确认可迁移性，不重新针对测试集调参。

## 8. 指标与记录

### 8.1 最终检索指标

- I2T：R@1、R@5、R@10、MedR、MeanR。
- T2I：R@1、R@5、R@10、MedR、MeanR。
- 总指标：rSum。

### 8.2 Mining 指标

- I2T/T2I top-1 GT hit rate。
- `Precision(c)`：按源置信度排序取 top coverage $c$ 的源节点，其全部回写边中真实边所占的加权比例。
- `Coverage(c)`：进入 top coverage 集合的源节点数除以全部待匹配源节点数；I2T/T2I 分别计算。
- `Recall(c)`：top coverage 集合中选中的真实边数除以全部真实边数；I2T 分母为每张图像的全部真实 caption 边，T2I 分母为每条 caption 的唯一真实图像边。
- precision-coverage AUC：在共同 coverage 网格 $[0.1,0.9]$（步长 0.05）计算。置信度并列时以整个并列组的正确率在边界处按比例计数，不用源 ID 任意截断，也不凭线性插值制造方法不可达到的 operating point；同时报告 25%、50%、75% coverage 的 precision。
- MNN precision 与 MNN coverage。
- `GT assigned mass`：所有方法先把每个源节点回写边的条件权重归一化为和 1；B0/B1 的单条硬边质量为 1，B2 使用局部 softmax，O2/O3/O4 使用截断后重归一化的 $q$。I2T 统计落在该图全部真实 caption 上的质量，T2I 统计落在唯一真实图像上的质量，再按源节点平均。该指标可跨方法比较。
- `raw GT transport mass`：直接使用未条件归一化的 UOT 质量，仅作 O2/O3/O4 内部诊断，不用于跨方法门控。
- 候选召回率：真实对应是否进入 top-k 候选图。
- 运输分布平均熵及分位数。
- 置信度分桶后的实际正确率，用于判断权重是否校准。

### 8.3 效率与稳定性指标

- memory-bank 更新 wall-clock time。
- 单次更新峰值 GPU 显存与 CPU 内存。
- Sinkhorn 迭代次数、停止残差、未收敛次数。
- NaN/Inf 数量和空候选节点数量。
- 完整训练 wall-clock time。

### 8.4 W&B 组织建议

- project：沿用 `RCSL`。
- group：`f30k-pl{paired_length}-seed{seed}-ot-study`。
- run name：`{method}-k{k}-eps{epsilon}-rho{rho}-ptop{pseudo_topk}`。
- tags：`baseline/mnn`、`control/topk`、`ot/balanced`、`ot/unbalanced`。
- 每次 memory-bank 更新使用同一 step 记录 mining、效率和收敛指标。

## 9. 成功标准与结论边界

OT 被认为值得保留，需要同时满足：

1. **伪配对有效性：** 在主要设置上，precision-coverage AUC 或 GT assigned mass 相对 $\max(\mathrm{B0},\mathrm{B2})$ 提高至少 1 个百分点，且 50% coverage 的 precision 不降低。
2. **检索收益：** 多种子平均 rSum 提升至少 1.0，且 I2T/T2I R@1 不出现稳定退化。
3. **稳定性：** 三个 seed 中至少两个取得正向 rSum 改善，结果不依赖单个异常 run。
4. **成本可接受：** 完整训练时间不超过 MNN 的 1.25 倍；memory-bank 更新峰值资源不超过目标机器容量。

若只改善 mining GT 指标而不改善检索，应得出“OT 更好地恢复了合成配对，但当前训练目标不能利用软计划”的结论，并优先检查 soft positive loss，而不能直接宣称 OT 提升了跨模态检索。

若 B2 与 OT 表现相当，应优先选择实现更简单的 B2，并将结论限定为“top-k 软伪配对有效，未观察到全局 OT 约束的额外收益”。

若平均 $\Delta\mathrm{rSum}\le0$，视为不支持 H2；若 $0<\Delta\mathrm{rSum}<1.0$，视为存在弱正向证据但未达到工程保留阈值；只有达到上述四项标准时，才将 OT 作为默认候选方法。研究假设证据强度与工程采用决策分开报告。

## 10. 风险与诊断顺序

| 风险 | 可观察信号 | 优先诊断/缓解 |
| --- | --- | --- |
| 候选图漏掉真匹配 | candidate recall 低 | 增大 k、取双向并集、改进 warm-up encoder |
| 平衡 OT 强迫错误匹配 | 边缘满足但 GT mass 低 | 使用 UOT、dustbin 或 partial OT |
| $\varepsilon$ 太小 | 迭代不稳定、熵接近 0 | log-domain 实现并适当增大 $\varepsilon$ |
| $\varepsilon$ 太大 | 分配近似均匀 | 减小 $\varepsilon$，检查成本尺度 |
| 伪标签自举错误 | 后续更新 GT hit 持续下降 | 降低刷新频率、冻结 encoder、提高置信度门槛 |
| 一对多监督被 top-1 丢失 | O2 无收益、O3 有收益 | 保留 top-k 计划并改用 soft positive loss |
| 结果来自额外候选而非 OT | B2 与 O3 持平 | 不宣称 OT 贡献，保留简单方法 |
| 运行不可比较 | checkpoint/seed 不一致 | 强制记录起点 checkpoint 哈希和数据排列 |

诊断顺序固定为：候选召回率 → Sinkhorn 收敛与边缘残差 → GT mass/精度覆盖率 → 置信度校准 → 训练损失 → 检索指标。这样可以区分“候选不存在”“求解错误”“权重错误”和“训练目标无法利用运输计划”。

## 11. 预期产物

- `ot_mining.py`：独立、可测试的 OT/UOT miner。
- 单元测试：边缘约束、置换等变性、数值稳定性、top-k 回写和空候选处理。
- mining-only 评估脚本或模式。
- W&B run group 及导出的汇总表。
- precision-coverage、GT mass、检索指标与资源开销图表。
- 一份包含正结果、负结果和失败配置的实验记录。

## 12. 最小执行清单

- [ ] 固化 B0 的 checkpoint、配置和 mining 指标。
- [ ] 抽取 MNN 与 OT 共用的 embedding/candidate 计算代码。
- [ ] 实现并测试稠密 log-domain Sinkhorn。
- [ ] 完成阶段 A 数值正确性验证。
- [ ] 实现 top-k 候选图与 UOT。
- [ ] 完成阶段 B mining-only 对照。
- [ ] 通过门控后完成阶段 C 单设置训练。
- [ ] 固定超参数后完成阶段 D 多种子实验。
- [ ] 汇总均值、方差、逐 seed 配对差值、失败案例和资源开销；扩展到至少五个 seed 后再给置信区间。
- [ ] 根据成功标准决定保留 OT、仅保留 top-k 软匹配，或否定该方向。
