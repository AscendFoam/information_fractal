# 快速验证实验计划：风险排查与可行性判定

## 〇、本文档的定位

Research_Roadmap 给出了完整的11周NeurIPS路线。但在全面投入之前，需要用**最小成本**快速回答一个根本问题：

> **这条路走不走得通？**

本文档设计一组**可在3-5天内完成**的快速验证实验（Quick Validation Experiments, QVE），针对NeurIPS方案中的每一个风险项，设计最小规模的探针实验（Probe Experiment），根据结果判定：继续（GO）、调整（PIVOT）、或放弃（KILL）。

---

## 一、风险项清单与优先级排序

从 Research_Roadmap 第四节提取风险项，按"验证紧迫度"重新排序：

| 编号 | 风险项 | 致命性 | 能否快速验证 | 优先级 |
|:---|:---|:---|:---|:---|
| **R1** | 四类文本的维度阶梯不存在（$D_2$无法分离四类） | 致命 | 能 | **P0** |
| **R2** | 幻觉与有效文本的$D_2$差异不显著 | 高 | 能 | **P0** |
| **R3** | Hidden States提取/维度计算的技术流程跑不通 | 致命(工程) | 能 | **P0** |
| **R4** | 短序列下$D_2$估计不稳定/不收敛 | 高 | 能 | **P1** |
| **R5** | 李雅普诺夫指数对四类文本无区分力 | 中 | 能 | **P1** |
| **R6** | 跨层ID曲线不呈现三阶段模式 | 中 | 能 | **P1** |
| **R7** | 不同模型上结果不一致 | 中 | 部分能 | **P2** |
| **R8** | LZ压缩率与$D_2$高度相关导致信息增量不足 | 中 | 能 | **P2** |

下面逐项设计验证实验。

---

## 二、P0级验证实验（必须先做，决定GO/KILL）

### QVE-0：工程流程端到端打通

**要排除的风险**：R3——技术流程本身跑不通

**实验内容**：
1. 加载一个小模型（Qwen-2.5-1.5B 或 LLaMA-3.2-1B，显存需求小）
2. 输入一段文本（~128 tokens），用 PyTorch forward hooks 提取每一层的 hidden states
3. 对提取到的 hidden states 矩阵（shape: `[n_tokens, hidden_dim]`），调用 `scikit-dimension` 计算 TwoNN 内蕴维数
4. 确认整个流程无报错、输出合理数值

**判定标准**：
- GO：成功提取 hidden states 并计算出维度数值（合理范围1-100）
- KILL：不适用——这是纯工程问题，跑不通就调试到跑通

**预计耗时**：2-4小时（含环境安装）

**最简实现思路**：
```
步骤:
1. pip install transformers scikit-dimension torch
2. 加载模型, 注册 forward hook 到每一层
3. 输入一段文本, 收集所有层的 hidden states
4. 对每层的 [n_tokens, hidden_dim] 矩阵调用 skdim.id.TwoNN().fit()
5. 打印每层的 ID 值
```

---

### QVE-1：四类文本维度阶梯快速探针（核心实验）

**要排除的风险**：R1（四类不可分）+ R2（幻觉与有效不可分）

这是整个项目的**生死实验**。如果四类文本在维度指标上无法区分，整个NeurIPS方案的核心假说就不成立。

**实验设计**：

#### 数据：每类仅需10-20个样本

不需要500个样本。快速验证只需看到信号是否存在。

```
样本构造方案（以 Qwen-2.5-7B 或 LLaMA-3-8B 为例）:

A类 (重复/死循环) - 10条:
  方法: Temperature=0.01, 无 repetition_penalty
  Prompt: 简单续写类 ("Once upon a time", "The answer is")
  预期: 模型陷入重复循环

B类 (流畅幻觉) - 10条:
  方法: Temperature=0.7 (正常), 提问模型不知道的虚假事实
  Prompt: "请详细介绍2025年诺贝尔物理学奖得主张三的贡献"
         "Describe the capital city of the country Atlantica"
         或使用 TruthfulQA 中模型倾向于出错的题目
  预期: 流畅但事实错误的文本

C类 (有效推理) - 10条:
  方法: Temperature=0.7, 提问模型擅长的推理问题
  Prompt: GSM8K 中模型能正确解答的数学题 (用CoT)
         或要求解释一个已知概念
  预期: 逻辑正确的推理链

D类 (乱码) - 10条:
  方法: Temperature=5.0+, 无 top-p/top-k 约束
  Prompt: 任意
  预期: 语义崩坏的乱码
```

#### 度量：对每条输出计算三个指标

```
对每条生成文本(取最后一层或中间层的hidden states轨迹):

1. TwoNN 内蕴维数 (LID)
   - 用 skdim.id.TwoNN()
   - 输入: [n_tokens, hidden_dim] 矩阵

2. 关联维数 D2
   - 用 skdim.id.CorrInt() 或自己实现 Grassberger-Procaccia
   - 注意: 对 40 个样本, N^2 计算量不大

3. LZ压缩率 (作为参考基线)
   - 对原始文本做 zlib.compress, 计算 压缩后长度/原始长度
```

#### 分析：画箱线图，目视判断

对三个指标分别画四类的箱线图（boxplot），核心关注：

1. **四类之间是否存在可见的分离？**（R1）
2. **B类（幻觉）是否低于C类（有效）？**（R2）
3. **各类的方差（箱体宽度）相对于类间距离是否可接受？**

#### 判定标准

```
GO条件 (任一满足即可继续):
  ✓ 四类的中位数呈现清晰的阶梯: D(A) < D(B) < D(C) < D(D)
  ✓ 至少三类可分（例如 重复 vs 有效 vs 乱码 可分，幻觉待定）
  ✓ 某一层或某一指标上四类显著分离

PIVOT条件 (信号弱但有方向):
  △ 仅在特定层（如中间层）可见分离 → 调整为逐层分析策略
  △ D2不行但LID可以 → 切换主指标
  △ 幻觉和有效无法分离，但其他三类可分 → 缩小claim范围，
    论文聚焦"三类分离"而非"四类分离"
  △ 绝对值不可分但变化率（层间差分）可分 → 改为层间动力学分析

KILL条件:
  ✗ 所有指标、所有层、四类完全混在一起，无任何分离趋势
  ✗ D2/LID计算结果随机波动极大，看不到任何模式
```

**预计耗时**：1-2天（含数据生成 + 维度计算 + 绘图分析）

---

### QVE-1b：距离度量的选择验证

**要排除的风险**：高维空间中欧氏距离退化（distance concentration），导致维度估计失效

**实验内容**：
- 对 QVE-1 的同一批数据，分别用三种距离度量计算维度：
  1. 欧氏距离（L2）
  2. 余弦距离（1 - cosine similarity）
  3. 对 hidden states 先做 PCA 降到50维，再用欧氏距离

**判定标准**：
- 如果三种度量给出一致趋势 → 结果稳健，GO
- 如果仅某种度量有效 → 锁定该度量，GO with note
- 如果全部无效 → 距离退化问题严重，需要更深的降维策略

**预计耗时**：与QVE-1同步，额外增加约2小时

---

## 三、P1级验证实验（确认技术可行性与辅助假说）

### QVE-2：$D_2$估计对序列长度的敏感性

**要排除的风险**：R4——短序列下维度估计不收敛

**实验内容**：
选择一条较长的有效文本生成（512+ tokens），分别截取前N个tokens的hidden states（N=32, 64, 128, 256, 512），计算$D_2$和LID，画出 N vs. D 曲线。

**判定标准**：
- GO：从某个N开始（如128），D值趋于稳定（波动<10%）
- PIVOT：需要256+才稳定 → 后续实验要求更长的生成序列
- KILL：到512还在剧烈波动 → $D_2$在此场景下不可靠，需换用其他指标

**预计耗时**：2-3小时

---

### QVE-3：跨层ID曲线的三阶段模式

**要排除的风险**：R6——三阶段演化不存在

**实验内容**：
对QVE-0中已提取的各层hidden states，画出 Layer vs. ID 曲线。使用3种不同类型的输入：
- 一段自然语言文本
- 一段CoT推理文本
- 一段重复文本

**判定标准**：
- GO：至少自然语言和CoT文本展现出明显的"先升后降"趋势
- PIVOT：曲线形态不是标准的三阶段但不同输入类型的曲线有差异 → 转为分析"曲线形态差异"
- INFO：纯信息收集，不构成KILL条件

**预计耗时**：1-2小时（数据在QVE-0中已有）

---

### QVE-4：李雅普诺夫指数的快速探针

**要排除的风险**：R5——$\lambda$对四类无区分力

**实验内容**：
1. 选C类（有效）和A类（重复）各3条样本
2. 对每条样本的prompt embedding加微小高斯噪声（$\epsilon = 10^{-4} \sim 10^{-3}$级别）
3. 重新推理，提取hidden states，计算轨迹分离度
4. 估算$\lambda$

**注意**：这个实验的技术实现比QVE-1复杂（需要两次forward pass + 轨迹对比），但样本量很少。

**判定标准**：
- GO：有效文本的$\lambda > 0$（微弱正值），重复文本的$\lambda < 0$或$\approx 0$
- PIVOT：$\lambda$数值不稳定 → 可能需要更精细的扰动设计，暂缓此指标
- INFO：即使$\lambda$不可用，$D_2$+LID已足够支撑论文

**预计耗时**：半天到1天

---

## 四、P2级验证实验（增强信心但非必须）

### QVE-5：LZ压缩率与$D_2$的冗余性检查

**要排除的风险**：R8——如果压缩率和$D_2$高度正相关，审稿人会质疑$D_2$的独立价值

**实验内容**：
- 用QVE-1的数据，计算每条样本的LZ压缩率
- 画$D_2$ vs. LZ压缩率的散点图
- 计算Pearson/Spearman相关系数

**判定标准**：
- 理想情况：相关性中等（0.3-0.7），说明$D_2$提供了压缩率之外的独立信息
- 可接受：高相关但$D_2$在幻觉检测上优于LZ压缩率
- 需调整：几乎完全相关 → 论文叙事需要从"$D_2$是新指标"转为"$D_2$提供了理论解释框架"

**预计耗时**：1小时（数据已有）

---

### QVE-6：第二个模型的快速交叉验证

**要排除的风险**：R7——发现仅在某个模型上成立

**实验内容**：
- 用QVE-1中表现最好的那一组参数/层/指标
- 在第二个模型上重复（例如QVE-1用LLaMA则此处用Qwen，反之亦然）
- 仅需A/C/D三类各5条即可

**判定标准**：
- GO：第二个模型上同样可见分离趋势
- PIVOT：量值不同但趋势一致 → 需要归一化处理
- 需关注：方向完全不同 → 发现可能是模型特异性的，需进一步排查

**预计耗时**：半天

---

## 五、一般信息序列的快速探针（加分项）

### QVE-7：CGR映射的视觉直觉验证

**要排除的风险**：不是排除风险，而是**快速获取视觉直觉**——如果成功，论文的视觉冲击力将极大提升

**实验内容**：
准备四种二进制序列，用CGR（混沌游戏表示法）映射到二维正方形中：

```
序列类型:
1. 周期序列:  010101... 或 001001001... (有理数)
2. 伪随机序列: Python random.getrandbits() 生成
3. 自然语言文本的UTF-8二进制编码 (取一段Wikipedia文本)
4. LLM有效输出的UTF-8二进制编码

CGR算法:
  初始点 P = (0.5, 0.5)
  对序列中每一位 b:
    if b == 0: P = (P + (0,0)) / 2  (左下角)
    if b == 1: P = (P + (1,1)) / 2  (右上角)
  # 对于二进制只有两个顶点, 图像是一维CGR
  # 更好的做法: 将二进制序列按2-bit分组得到4字母表{00,01,10,11}
  # 然后用四个角作为顶点做标准CGR
```

**预期**：
- 周期序列 → 少数几个聚集点
- 伪随机序列 → 均匀填满正方形
- 自然语言/LLM有效输出 → 呈现非均匀但有结构的分形图案

**判定标准**：
- 视觉上四种序列的CGR图像是否呈现明显不同的图案？
- 如果是 → 这将成为论文中极有冲击力的Figure
- 如果差异不大 → 暂时搁置CGR方向，聚焦hidden states分析

**预计耗时**：3-4小时（CGR算法非常简单，核心是可视化调参）

---

## 六、验证实验的执行顺序与决策树

```
Day 1 (上午): QVE-0 工程流程打通
  │
  ├─ 失败 → 调试环境，解决工程问题（不影响科学判断）
  │
  └─ 成功 ↓

Day 1 (下午) + Day 2: QVE-1 四类维度阶梯探针 + QVE-1b 距离度量
  │
  ├─ KILL信号 → 停止。考虑完全换方向或重新设计指标体系
  │
  ├─ PIVOT信号 → 根据具体情况调整:
  │   ├─ 仅特定层可分 → QVE-3 确认跨层模式
  │   ├─ D2不行LID可以 → 锁定LID为主指标
  │   ├─ 幻觉不可分 → 缩小claim为"三类分离"
  │   └─ 绝对值不行但变化率行 → 调整为层间动力学
  │
  └─ GO信号 ↓

Day 3: QVE-2 (序列长度敏感性) + QVE-3 (跨层曲线)
  │  两者独立，可并行
  │
  ├─ QVE-2 发现需要很长序列 → 调整后续实验的生成长度
  └─ QVE-3 获取跨层模式信息 → 确定最佳分析层

Day 4: QVE-4 (李雅普诺夫指数) + QVE-5 (LZ冗余性)
  │
  ├─ QVE-4 成功 → 论文多一个强指标
  ├─ QVE-4 失败 → 论文不依赖λ，只用D2+LID也足够
  └─ QVE-5 高冗余 → 调整论文叙事，强调理论框架而非指标新颖性

Day 5: QVE-6 (跨模型) + QVE-7 (CGR可视化)
  │
  ├─ QVE-6 一致 → 极大增强信心，确认普遍性
  └─ QVE-7 视觉惊艳 → 论文增加一般信息序列的section
```

**整体决策树**：

```
              QVE-1 结果
              /    |     \
           KILL  PIVOT   GO
            |      |      |
        放弃该    调整    全速推进
        方向     策略
                  |
            QVE-2/3/4
            进一步确认
            调整后的方案
```

---

## 七、各风险项的预判与备选方案

### R1：四类维度阶梯不存在

**预判**：成立概率约60-70%。理由如下：
- 支持证据：Du & Tanaka-Ishii (2025) 已经在关联维数上观察到了退化文本和正常文本的差异
- 反对因素：他们的工作是基于文本token序列而非hidden states，hidden states空间的几何性质可能不同

**如果不成立，备选方案**：

```
Plan B1: 切换分析对象
  不分析 hidden states, 改为分析 token embedding 序列
  (更接近 Du & Tanaka-Ishii 的方法)

Plan B2: 切换指标
  放弃 D2, 改用:
  - 持续同调 (Persistent Homology) 的 Betti numbers
  - 持续图 (Persistence Diagram) 的统计特征
  - 轨迹的递归图 (Recurrence Plot) + 递归量化分析 (RQA)

Plan B3: 切换粒度
  不分析 token 级别, 改为:
  - 句子级: 每句话一个向量，分析句子序列的轨迹
  - 段落级: 更粗粒度的语义轨迹

Plan B4: 缩小范围
  放弃"四类分离"的宏大claim, 转为:
  "分形维数可以区分高质量和低质量文本"(二分类)
  这个更容易成立, 也仍然有论文价值
```

### R2：幻觉与有效文本的$D_2$差异不显著

**预判**：这是最大的不确定性。成立概率约50%。

**关键思考**：幻觉文本在语法和流畅度上与有效文本非常相似，hidden states的全局几何特征可能确实无法区分它们。但如果Du & Tanaka-Ishii的发现可复现，则有希望。

**如果不成立，备选方案**：

```
Plan B-R2-1: 跨层变化率而非绝对值
  假设幻觉文本在中间层开始"维度坍缩",
  而有效文本维持高维到更深的层。
  指标: d(ID)/d(Layer) 在中间层的梯度差异

Plan B-R2-2: 拓扑特征
  D2衡量"维度"，但拓扑特征衡量"连通结构"
  即使两者维度相同，拓扑空洞(holes)的数量和分布可能不同
  → 引入 Persistent Homology

Plan B-R2-3: 缩小claim
  承认幻觉不可区分，论文改为:
  "分形维数区分三类: 重复/正常/乱码"
  这仍然是有价值的发现，只是没那么惊艳

Plan B-R2-4: 组合指标
  D2 + LID + λ + LZ压缩率 的组合特征
  虽然单指标不够，但特征组合可能足以区分
  → 训练一个轻量分类器
```

### R4：短序列估计不稳定

**预判**：大概率存在此问题（概率70%以上）。维度估计是统计方法，短序列天然方差大。

**预判备选方案**：
```
方案1: 提高生成长度到 512-1024 tokens
方案2: 滑动窗口 + 取中位数（对 1024 tokens 的序列，
       用窗口大小 256、步长 64 做多次估计取中位数）
方案3: 使用对短序列更鲁棒的估计方法（如 MLE 而非 TwoNN）
```

### R5：李雅普诺夫指数无区分力

**预判**：技术实现有难度，可能不稳定。但这不是致命风险——论文可以不依赖$\lambda$。

**如果不成立**：从论文中去掉$\lambda$相关内容，仅保留$D_2$+LID作为核心指标。论文叙事从"三个互补指标"调整为"维度指标族"。

---

## 八、快速验证的成功判定标准汇总

完成全部QVE后，根据结果对NeurIPS方案做最终判定：

### 全速推进（Full GO）

以下条件**全部满足**：
- [ ] QVE-1：至少三类文本的$D_2$或LID显著可分（p<0.05或目视明显）
- [ ] QVE-2：序列长度256+时维度估计趋于稳定
- [ ] QVE-3：跨层ID曲线展示出可解释的模式

满足时的行动：**按原定Research_Roadmap执行**

### 有条件推进（Conditional GO）

以下条件**至少一个满足**：
- [ ] QVE-1中只有两类可分（如重复 vs 有效），但效果量大
- [ ] 某一个指标可以，另一个不行
- [ ] 某些层可以，另一些不行

满足时的行动：
1. 缩小论文claim范围
2. 根据PIVOT方向调整实验设计
3. 重新评估NeurIPS时间线是否还来得及

### 方向性调整（Major Pivot）

QVE-1中：
- [ ] Hidden states空间的$D_2$/LID完全混乱
- [ ] 但token embedding空间（而非hidden states）展示出了维度差异

满足时的行动：
1. 切换分析对象从hidden states到token embeddings
2. 重新设计计算流程
3. NeurIPS时间线可能紧张，评估是否改投ICLR 2027

### 停止（KILL）

以下条件**全部满足**：
- [ ] QVE-1中所有指标、所有层、所有距离度量下四类完全无法区分
- [ ] QVE-1b的降维方案也无效
- [ ] 换到token embedding空间也无效

满足时的行动：
1. 放弃"维度指标区分输出质量"这一核心假说
2. 转向其他方向，如：
   - 纯理论方向（一般信息序列的分形映射，不涉及LLM实验）
   - 其他可解释性方向

---

## 九、每个实验的具体代码蓝图

以下给出各QVE的伪代码级实现方案，便于立刻动手。

### QVE-0 代码蓝图

```python
# qve0_pipeline_test.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import skdim

# 1. 加载模型
model_name = "Qwen/Qwen2.5-1.5B"  # 小模型, 快速验证
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# 2. 注册 hook 提取 hidden states
hidden_states_all_layers = {}
def make_hook(layer_idx):
    def hook_fn(module, input, output):
        # output 通常是 tuple, 第一个元素是 hidden state
        if isinstance(output, tuple):
            hidden_states_all_layers[layer_idx] = output[0].detach().cpu()
        else:
            hidden_states_all_layers[layer_idx] = output.detach().cpu()
    return hook_fn

for i, layer in enumerate(model.model.layers):
    layer.register_forward_hook(make_hook(i))

# 3. 推理
text = "The theory of fractal geometry provides a mathematical framework..."
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# 4. 计算每层 ID
for layer_idx, hs in sorted(hidden_states_all_layers.items()):
    # hs shape: [batch=1, n_tokens, hidden_dim]
    data = hs.squeeze(0).float().numpy()  # [n_tokens, hidden_dim]
    if data.shape[0] >= 10:  # TwoNN 需要足够多的点
        estimator = skdim.id.TwoNN()
        estimator.fit(data)
        print(f"Layer {layer_idx}: ID = {estimator.dimension_:.2f}")
```

### QVE-1 代码蓝图

```python
# qve1_dimension_staircase.py
import numpy as np
import matplotlib.pyplot as plt

# --- 数据生成 (分四组) ---
def generate_samples(model, tokenizer, prompts, gen_kwargs, n_samples=10):
    """对一组 prompt 生成文本并提取最后一层 hidden states"""
    all_hidden = []
    for prompt in prompts[:n_samples]:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=256, output_hidden_states=True,
                return_dict_in_generate=True, **gen_kwargs
            )
        # 提取生成阶段的 hidden states (最后一层)
        # 注意: generate with output_hidden_states 的返回格式
        # 需要根据具体实现拼接每一步的 hidden state
        hs = extract_generation_hidden_states(outputs)  # 自行实现
        all_hidden.append(hs)
    return all_hidden

# 四组参数
configs = {
    "A_repetition": {"temperature": 0.01, "do_sample": True},
    "B_hallucination": {"temperature": 0.7, "do_sample": True},
    "C_effective": {"temperature": 0.7, "do_sample": True},
    "D_gibberish": {"temperature": 5.0, "do_sample": True, "top_k": 0, "top_p": 1.0},
}

# --- 维度计算 ---
def compute_dimensions(hidden_states_list):
    """对一组 hidden states 计算 TwoNN 和 CorrDim"""
    results = []
    for hs in hidden_states_list:
        data = hs.float().numpy()
        if data.shape[0] < 20:
            continue
        # TwoNN
        twonn = skdim.id.TwoNN()
        twonn.fit(data)
        # 可选: CorrInt
        # corrint = skdim.id.CorrInt()
        # corrint.fit(data)
        results.append({"twonn": twonn.dimension_})
    return results

# --- 可视化 ---
def plot_staircase(all_results):
    """四类箱线图"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    labels = list(all_results.keys())
    data = [
        [r["twonn"] for r in all_results[label]]
        for label in labels
    ]
    ax.boxplot(data, labels=labels)
    ax.set_ylabel("Intrinsic Dimension (TwoNN)")
    ax.set_title("QVE-1: Dimension Staircase Test")
    plt.savefig("qve1_staircase.png", dpi=150)
    plt.show()
```

### QVE-4 (李雅普诺夫) 代码蓝图

```python
# qve4_lyapunov.py

def estimate_lyapunov(model, tokenizer, prompt, epsilon=1e-4, n_steps=50):
    """通过 embedding 微扰估计准李雅普诺夫指数"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 原始 embedding
    with torch.no_grad():
        embed_orig = model.model.embed_tokens(inputs.input_ids)  # [1, seq, dim]

    # 加扰动
    noise = torch.randn_like(embed_orig) * epsilon
    embed_perturbed = embed_orig + noise

    # 分别通过网络（需要用 inputs_embeds 而非 input_ids）
    with torch.no_grad():
        out_orig = model(inputs_embeds=embed_orig, output_hidden_states=True)
        out_pert = model(inputs_embeds=embed_perturbed, output_hidden_states=True)

    # 计算每层的分离度
    separations = []
    for layer_idx in range(len(out_orig.hidden_states)):
        h_orig = out_orig.hidden_states[layer_idx]  # [1, seq, dim]
        h_pert = out_pert.hidden_states[layer_idx]
        delta = (h_orig - h_pert).norm(dim=-1).mean().item()
        separations.append(delta)

    # 拟合指数增长率
    log_seps = np.log(np.array(separations) + 1e-12)
    layers = np.arange(len(separations))
    # 线性拟合 log(delta) vs layer → 斜率即为 λ
    slope, intercept = np.polyfit(layers, log_seps, 1)
    return slope  # 近似李雅普诺夫指数
```

### QVE-7 (CGR) 代码蓝图

```python
# qve7_cgr_visualization.py

def text_to_binary(text):
    """文本 → UTF-8 二进制序列"""
    return ''.join(format(byte, '08b') for byte in text.encode('utf-8'))

def binary_to_4letter(binary_str):
    """二进制 → 4字母表 {00, 01, 10, 11}"""
    symbols = []
    for i in range(0, len(binary_str) - 1, 2):
        symbols.append(binary_str[i:i+2])
    return symbols

def cgr_2d(symbols, alphabet={"00": (0,0), "01": (1,0), "10": (0,1), "11": (1,1)}):
    """标准 CGR: 四角映射"""
    x, y = 0.5, 0.5
    points = []
    for s in symbols:
        cx, cy = alphabet[s]
        x = (x + cx) / 2
        y = (y + cy) / 2
        points.append((x, y))
    return np.array(points)

# 生成四种序列的CGR
sequences = {
    "periodic": "01" * 10000,                             # 有理数
    "pseudo_random": bin(random.getrandbits(20000))[2:],  # 伪随机
    "natural_text": text_to_binary(wikipedia_text),       # 自然语言
    "llm_output": text_to_binary(llm_effective_output),   # LLM有效输出
}

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for ax, (name, seq) in zip(axes.flat, sequences.items()):
    symbols = binary_to_4letter(seq)
    points = cgr_2d(symbols)
    ax.scatter(points[:, 0], points[:, 1], s=0.1, alpha=0.3)
    ax.set_title(name)
    ax.set_aspect('equal')
plt.savefig("qve7_cgr_comparison.png", dpi=200)
```

---

## 十、验证完成后的决策矩阵

| QVE-1结果 | QVE-2结果 | QVE-4结果 | 综合判定 | 下一步行动 |
|:---|:---|:---|:---|:---|
| 四类显著分离 | 128+稳定 | λ可区分 | **最佳情况** | 全速按Roadmap执行 |
| 四类显著分离 | 128+稳定 | λ不可用 | **良好** | 去掉λ，论文仅用D2+LID |
| 三类可分(幻觉混) | 128+稳定 | 任意 | **可行** | 缩小claim为三类；幻觉分析用跨层梯度或拓扑特征 |
| 仅特定层可分 | 需512+ | 任意 | **需调整** | 改为"跨层动力学分析"视角，强调层间变化而非绝对维度 |
| 两类可分(正常vs异常) | 任意 | 任意 | **勉强可行** | 论文降级为"二分类检测"，简化理论框架 |
| 完全不可分 | 不稳定 | 不可用 | **KILL** | 放弃此方向，或转向纯理论/CGR可视化方向 |

---

## 十一、总结

快速验证的核心逻辑是：**用3-5天的探针实验，避免11周的沉没成本。**

最关键的一步是 **QVE-1**（四类维度阶梯探针）。这个实验只需要约40条生成样本和几小时的计算，就能回答"这条路是否值得走"这个根本问题。如果QVE-1给出GO信号，后续的QVE-2到QVE-7都是在此基础上的信心增强和技术调优。

建议的即刻行动：

1. 今天：搭建环境，完成QVE-0（工程打通）
2. 明天：执行QVE-1（生死实验）
3. 后天：根据QVE-1结果决定是GO还是PIVOT，再依次执行P1/P2级验证
