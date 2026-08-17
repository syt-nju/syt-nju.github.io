# OPD深度解析：从数学推导到DeepSeek V4、SWIFT与verl实践

> Source: https://zhuanlan.zhihu.com/p/2033212181823608430
> Author: Zhihu on Obsidian
> Collected: 2026-08-17
> Published: Unknown
> License: Unknown
> Completeness: Complete
> Retrieval note: Direct zhuanlan fetch returned HTTP 403/503. Zhihu CLI is installed but this environment has no keychain and no ZHIHU_ACCESS_SECRET, so open-platform search was not used. Full article body retrieved via https://r.jina.ai/https://zhuanlan.zhihu.com/p/2033212181823608430. Zhihu in-article zhida search URLs were stripped as navigation noise; wording otherwise preserved.

> 这篇文章要回答的问题：OPD 到底在优化什么？训练 prefix 从哪里来，以及在每个 prefix 上你比较的是一个 sampled token、一个 top-k 局部分布，还是整个 full-vocab 分布？sampled-token、top-k、full-vocab、off-policy、on-policy 分别差在哪里？MiniLLM、GKD、DeepSeek V4、SWIFT、verl 这些实现又分别对应哪种数学目标？

* * *

### 0. 先给结论

**0.1 OPD 的一句话定义**

**On-Policy Distillation（OPD，同策略蒸馏）** 指：

> 学生模型先用自己的当前策略生成回答，再让教师模型在这些学生自己生成的轨迹上提供监督信号，学生据此更新。

形式上：

y \sim \pi_\theta(\cdot|x)\\

第 t 个位置的 prefix 是：

c_t=(x,y_{<t})\\

teacher 在这个 c_t 上提供：

*  sampled token 的 logprob；或
*  teacher top-k 分布；或
*  full-vocab logits。

然后 student 更新。

**0.2 OPD 与普通 SFT/KD 的根本区别**

普通 SFT/KD 在数据前缀上训练：

c_t=(x,y^{data}_{<t})\\

或者 teacher-generated 前缀：

c_t=(x,y^{teacher}_{<t})\\

这叫 off-policy，因为训练 prefix 不是学生自己会走到的状态。

OPD 在学生 prefix 上训练：

c_t=(x,y^{student}_{<t}),\quad y^{student}\sim\pi_\theta(\cdot|x)\\

这能缓解 exposure bias：学生推理时会犯自己的错误，OPD 让训练也发生在这些错误前缀上。

**0.3 OPD 的三个核心维度**

| 维度 | 选项 | 关键问题 |
| --- | --- | --- |
| prefix 来源 | dataset / teacher / student | 是 off-policy 还是 on-policy？ |
| teacher 信号粒度 | sampled-token / top-k / full-vocab | 每个 prefix 上看一个 token、一小撮 token，还是整个词表？ |
| 优化方式 | direct loss / policy gradient advantage | 是直接反传 KL，还是把 KL 当 reward/advantage？ |

**0.4 本文最终判断**

*  **MiniLLM**：最接近 sequence-level reverse KL 的 policy-gradient OPD，但需要很多稳定化技巧。
*  **GKD / SWIFT**：更像“在 student/teacher/dataset prefix 上做 distribution matching”，可以 full-vocab，也可以 teacher top-k。
*  **Thinking Machines / verl sampled-token 模式**：只比较 sampled token 的 teacher-student logprob gap，把它当 reward/advantage。
*  **Revisiting OPD 的 top-K local support matching**：用 teacher top-k 支持集上的局部分布 KL 替代 sampled-token。
*  **DeepSeek V4**：多教师、学生生成轨迹上的 **full-vocabulary reverse KL logit distillation**，不是 sampled-token，也不是 top-k。
*  **SWIFT**：`rlhf_type=gkd`，`lmbda=1,beta=1,gkd_logits_topk=None` 最接近单教师 full-vocab reverse-KL OPD；设 `gkd_logits_topk` 后是 teacher top-k 重归一化版本。
*  **verl 主仓**：同时支持 sampled-token KL estimator（k1/k2/k3）和 top-k forward KL；可 direct loss 或 policy-gradient。
*  **verl-recipe/gkd**：大规模 Megatron top-k OPD，teacher server 返回 top-k logprobs/indices，支持 KL/RKL/KL_RKL/JSD。

* * *

### 1. 基础符号：先把问题写清楚

设：

*  x：prompt；
*  y=(y_1,\dots,y_T)：模型生成的回答；
*  c_t=(x,y_{<t})：第 t 步的上下文/prefix；
*  \pi_\theta：学生模型；
*  q 或 \pi_T：教师模型；
*  \mathcal V：词表。

自回归分解：

\pi_\theta(y|x)=\prod_{t=1}^{T}\pi_\theta(y_t|x,y_{<t}) =\prod_{t=1}^{T}\pi_\theta(y_t|c_t)\\

取 log：

\log\pi_\theta(y|x)=\sum_{t=1}^{T}\log\pi_\theta(y_t|c_t)\\

teacher 同理：

\log q(y|x)=\sum_{t=1}^{T}\log q(y_t|c_t)\\

* * *

### 2. Off-policy KD：最早、最常见的蒸馏

普通蒸馏/SFT 通常是：给定训练数据：

(x,y^{data})\\

或者 teacher 先生成：

y^{teacher}\sim q(\cdot|x)\\

然后 student 在这些固定序列上做 next-token prediction：

\mathcal L_{SFT} = -\sum_t \log \pi_\theta(y_t^{data}|x,y_{<t}^{data})\\

如果加入 teacher logits，则是 soft-label KD：

\mathcal L_{KD,t} = D(P_T(\cdot|x,y_{<t}^{data})\|P_S(\cdot|x,y_{<t}^{data}))\\

例如最常见的 teacher-to-student soft CE：

\mathcal L_{KD,t} = -\sum_{v\in\mathcal V}P_T(v|c_t)\log P_S(v|c_t)\\

它等价于 forward KL：

D_{KL}(P_T\|P_S) =\sum_v P_T(v)\log\frac{P_T(v)}{P_S(v)}\\

因为 teacher entropy：

\sum_v P_T(v)\log P_T(v)\\

对 student 参数是常数。

**2.1 这种方法的问题：exposure bias**

训练时 prefix 是数据/teacher 的：

c_t=(x,y^{data}_{<t})\\

推理时 prefix 是 student 自己生成的：

c_t=(x,y^{student}_{<t})\\

一旦 student 前面走偏，后面的 prefix 就进入训练时没见过的区域，错误会连锁放大。这就是 exposure bias。

OPD 就是为了解决这个分布不一致问题。

* * *

### 3. OPD 的基础目标：sequence-level reverse KL

最自然的 OPD 目标是让 student 的整条回答分布靠近 teacher：

J(\theta) =\mathbb E_{x\sim D}\left[D_{KL}(\pi_\theta(\cdot|x)\|q(\cdot|x))\right]\\

对单个 x：

J_x(\theta) =\sum_y \pi_\theta(y|x)\log\frac{\pi_\theta(y|x)}{q(y|x)}\\

也可以写成期望：

J_x(\theta) =\mathbb E_{y\sim\pi_\theta(\cdot|x)} \left[ \log\pi_\theta(y|x)-\log q(y|x) \right]\\

这个 KL 方向是 reverse KL：

D_{KL}(student\|teacher)\\

它是 mode-seeking：学生倾向于集中到 teacher 支持的主要模式上，而不是覆盖 teacher 的所有可能模式。

* * *

### 4. Sequence-level OPD 的梯度推导

目标：

J(\theta)=\sum_y \pi_\theta(y)\left[\log\pi_\theta(y)-\log q(y)\right]\\

求导：

\nabla J =\sum_y \nabla\pi_\theta(y)\left[\log\pi_\theta(y)-\log q(y)\right] +\sum_y \pi_\theta(y)\nabla\log\pi_\theta(y)\\

利用：

\nabla\pi_\theta(y)=\pi_\theta(y)\nabla\log\pi_\theta(y)\\

得到：

\nabla J =\mathbb E_{y\sim\pi_\theta} \left[ (\log\pi_\theta(y)-\log q(y))\nabla\log\pi_\theta(y) \right] +\mathbb E_{y\sim\pi_\theta}[\nabla\log\pi_\theta(y)]\\

第二项为 0：

\mathbb E_{y\sim\pi_\theta}[\nabla\log\pi_\theta(y)] =\sum_y \nabla\pi_\theta(y) =\nabla 1 =0\\

所以：

\nabla J =\mathbb E_{y\sim\pi_\theta} \left[ (\log\pi_\theta(y)-\log q(y))\nabla\log\pi_\theta(y) \right]\\

这就是 policy-gradient 形式。

* * *

### 5. 从 sequence-level 到 token-level：return-to-go 与方差问题

定义每个 token 的 log-ratio：

r_t=\log\frac{\pi_\theta(y_t|c_t)}{q(y_t|c_t)}\\

定义 score gradient：

g_t=\nabla_\theta\log\pi_\theta(y_t|c_t)\\

因为：

\log\frac{\pi_\theta(y|x)}{q(y|x)} =\sum_{t'=1}^{T}r_{t'}\\

\nabla\log\pi_\theta(y|x)=\sum_{t=1}^{T}g_t\\

所以直接 estimator 是：

\hat g_{seq} =\sum_{t=1}^{T}\left(\sum_{t'=1}^{T}r_{t'}\right)g_t\\

但第 t 步 action 不能影响过去 token，所以过去项 t'<t 对 g_t 的期望贡献为 0。于是可写成 causal return-to-go：

\hat g_{seq} =\sum_{t=1}^{T}\left(\sum_{t'=t}^{T}r_{t'}\right)g_t\\

**5.1 token-level OPD 的近似**

工程里经常只保留当前 token：

\hat g_{tok} =\sum_{t=1}^{T}r_t g_t\\

这就是 sampled-token OPD / token-level OPD 的基本形式。

它相对 sequence-level 目标是有偏的，因为丢掉了未来项：

\sum_{t'=t+1}^{T}r_{t'}\\

但它大幅降低方差。

**5.2 Revisiting OPD 的 bias-variance 结论**

《Revisiting On-Policy Distillation》给出的结论是：在 reward 和 score-function 项有界时，最坏情况下：

\mathrm{Var}(\hat g_{tok})=O(T^2)\\

而 sequence-level estimator：

\mathrm{Var}(\hat g_{seq})=O(T^4)\\

这解释了为什么长 reasoning / agent 任务里，严格 sequence-level OPD 很容易炸。

可以用折扣形式连接二者：

\hat g_\gamma =\sum_{t=1}^{T}\left(\sum_{t'=t}^{T}\gamma^{t'-t}r_{t'}\right)g_t\\

*  \gamma=0：token-level OPD；
*  \gamma=1：sequence-level OPD；
*  中间值：折中。

实验上，\gamma 越大，未来奖励耦合越强，梯度方差越高。

* * *

### 6. sampled-token、top-k、full-vocab：到底差在哪里？

这是 OPD 最容易混的地方。

**6.1 sampled-token OPD：只看 student 实际采出来的 token**

student rollout：

y\sim\pi_\theta(\cdot|x)\\

第 t 步 sample 出：

y_t\sim\pi_\theta(\cdot|c_t)\\

sampled-token OPD 只比较这个 token：

A_t=\log q(y_t|c_t)-\log\pi_\theta(y_t|c_t)\\

如果 teacher 比 student 更支持这个 token：

q(y_t|c_t)>\pi_\theta(y_t|c_t)\\

则 A_t>0，提高该 token 概率。

如果 teacher 不支持，而 student 自己很支持，则 A_t<0，压低该 token 概率。

这就是 RL-style OPD：

\nabla J\approx A_t\nabla\log\pi_\theta(y_t|c_t)\\

它不需要 teacher top-k，也不需要 full logits，只需要：

q(y_t|c_t)\\

也就是 teacher 对 sampled token 的 logprob。

**6.2 top-k OPD：在 teacher top-k 支持集上做局部分布比较**

teacher 在 prefix c_t 上返回 top-k：

S_t=TopK_q(c_t)\\

然后比较：

q(v|c_t),\quad \pi_\theta(v|c_t),\quad v\in S_t\\

可以做 forward KL：

\sum_{v\in S_t}q(v|c_t)\left[\log q(v|c_t)-\log\pi_\theta(v|c_t)\right]\\

也可以做 support-set reverse KL。若重归一化：

\hat q(v|c_t)=\frac{q(v|c_t)}{\sum_{u\in S_t}q(u|c_t)}\\

\hat\pi(v|c_t)=\frac{\pi(v|c_t)}{\sum_{u\in S_t}\pi(u|c_t)}\\

则 reverse KL：

D_{KL}(\hat\pi\|\hat q) =\sum_{v\in S_t}\hat\pi(v|c_t)\log\frac{\hat\pi(v|c_t)}{\hat q(v|c_t)}\\

这比 sampled-token 稳定，因为每个 prefix 不只看一个 token，而看 teacher 支持的一组候选。

**6.3 full-vocab OPD：对整个词表分布做 KL**

full-vocab 在每个 prefix 上比较：

D(P_S(\cdot|c_t)\|P_T(\cdot|c_t))\\

即：

\sum_{v\in\mathcal V}P_S(v|c_t)\log\frac{P_S(v|c_t)}{P_T(v|c_t)}\\

这是信息最完整的版本，但显存和计算代价最大。

**6.4 三者对比**

| 形式 | teacher 返回 | 每个 prefix 的信息量 | 成本 | 稳定性 |
| --- | --- | --- | --- | --- |
| sampled-token | sampled token logprob | 低 | 低 | 差一些 |
| top-k | top-k ids + logprobs | 中 | 中 | 较好 |
| full-vocab | full logits | 高 | 高 | 最好但昂贵 |

一句话：

> **sampled-token 是“teacher 评价 student 这一步实际说出的 token”；top-k 是“teacher 告诉 student 它最看好的 K 个候选”；full-vocab 是“teacher 把完整分布都给 student”。**

* * *

### 7. KL 方向：forward KL、reverse KL、JSD

**7.1 forward KL：mode-covering**

D_{KL}(P_T\|P_S)=\sum_v P_T(v)\log\frac{P_T(v)}{P_S(v)}\\

期望在 teacher 分布上取。只要 teacher 认为某 token 有概率，student 就不能给太低。

特点：

*  覆盖 teacher 多种模式；
*  保多样性；
*  但学生容量不足时可能学得“平均”。

**7.2 reverse KL：mode-seeking**

D_{KL}(P_S\|P_T)=\sum_v P_S(v)\log\frac{P_S(v)}{P_T(v)}\\

期望在 student 分布上取。student 把概率放到 teacher 不支持的地方会被重罚。

特点：

*  收缩到 teacher 高概率模式；
*  适合推理、代码、工具调用这类要走一条可靠路径的任务；
*  但可能导致多样性坍塌。

**7.3 JSD：两者之间的折中**

广义 JSD：

D_{JSD(\beta)}(P_T,P_S) =\beta KL(P_T\|M)+(1-\beta)KL(P_S\|M)\\

其中：

M=\beta P_T+(1-\beta)P_S\\

在 SWIFT 的实现里：

*  `beta=0`：forward KL；
*  `beta=1`：reverse KL；
*  `beta=0.5`：接近对称 JSD。

* * *

### 8. MiniLLM：早期 OPD 的严格 reverse-KL 路线

MiniLLM 的核心观点是：传统 forward KL 对生成式 LLM 不理想，因为 teacher 分布有很多模式，student 容量小，forward KL 会逼 student 覆盖 teacher 的低概率区域，导致生成质量下降。

MiniLLM 使用 reverse KL：

\min_\theta D_{KL}(\pi_\theta\|q)\\

用 policy gradient 优化，并加入稳定化技巧：

*  single-step decomposition；
*  teacher-mixed sampling；
*  length normalization。

MiniLLM 的价值在于，它把 LLM 蒸馏和 reverse-KL policy-gradient 目标联系起来，是理解 OPD 理论的起点。

但工程上，sequence-level / policy-gradient 形式高方差，需要很多 trick。

* * *

### 9. sampled-token OPD 的失败模式：Revisiting OPD 的三类问题

《Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes》集中分析了 sampled-token OPD 为什么会不稳。

**9.1 失败模式一：单 token 信号高度失衡**

sampled-token reward：

\log q(y_t|c_t)-\log\pi(y_t|c_t)\\

实践中，大多数 sampled token 可能都是负 reward。训练会过度依赖少数正 reward token，容易被填充词、标点、犹豫词这类局部 token 带偏。

**9.2 失败模式二：teacher 在 student OOD prefix 上不可靠**

OPD 假设 teacher 在 student prefix 上能给可靠信号。但如果 student 已经走进重复、自我重置、错误推理轨迹，teacher 对局部下一个 token 的概率未必代表整条轨迹的质量。

长序列越往后，teacher-student logprob gap 的方差越大，信号越不稳定。

**9.3 失败模式三：tokenizer / special token mismatch**

如果 teacher 和 student tokenizer 不同，同一语义会切成不同 token。sampled-token 比较的是单个 token，因此会把 tokenizer 差异误判成语义差异。

例如 `<think>` 在两个 tokenizer 下切法不同，teacher 可能给 student 的某个子 token 极低概率，但语义其实没错。

* * *

### 10. Revisiting OPD 的修复：teacher top-K local support matching

核心想法：别只看 sampled token，而是在 teacher top-K 支持集上做局部分布匹配。

定义：

S_t=TopK_q(c_t)\\

支持集内重归一化：

\hat\pi(v|c_t)=\frac{\pi(v|c_t)}{\sum_{u\in S_t}\pi(u|c_t)}\\

\hat q(v|c_t)=\frac{q(v|c_t)}{\sum_{u\in S_t}q(u|c_t)}\\

局部 reverse KL：

\mathcal L_{topK}(c_t) =\sum_{v\in S_t}\hat\pi(v|c_t)\log\frac{\hat\pi(v|c_t)}{\hat q(v|c_t)}\\

配套技巧：

1.  **支持集内重归一化**：不做会崩。
2.  **top-p rollout**：让 student 轨迹别偏太远。
3.  **special-token masking**：减轻 tokenizer/special token mismatch。

实验上，在数学和 ALFWorld 多任务里，这种 top-K local support matching 比 sampled-token OPD 更稳定，梯度 norm 更小，length clipping 更少，teacher-student gap 更好。

* * *

### 11. Rethinking OPD：OPD 成功不只看 teacher 分数

《Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe》提出两个很重要的条件。

**11.1 条件一：thinking-pattern consistency**

OPD 成功不只看 teacher benchmark 分数，而看 teacher 和 student 的思考模式是否一致。

关键指标是 top-k token overlap：

Overlap(c_t)=\frac{|TopK_S(c_t)\cap TopK_T(c_t)|}{K}\\

成功 OPD 中：

*  overlap ratio 从约 72% 升到 91% 以上；
*  overlap token 承载 97%-99% 的概率质量；
*  entropy gap 逐步缩小。

失败 OPD 中：

*  overlap 一开始低，后面上不去；
*  entropy mismatch 长期存在。

**11.2 条件二：teacher 要有新知识，而不只是更大**

如果 teacher 只是同 pipeline 的更大模型，未必带来有效新能力。真正有效的 teacher 通常经过额外 post-training / RL，具备 student 没学过的新策略。

反向蒸馏实验也说明：student 会被拉回 teacher 的思维模式，而不只是追随 teacher 的 benchmark 高低。

**11.3 实践 recipe**

**off-policy cold start**

先用 teacher 轨迹 SFT，让 student 靠近 teacher 的思考模板，再进入 OPD。

这提升初始 overlap ratio，减少 entropy gap，提高最终性能上限。

**teacher-aligned prompt selection**

prompt 模板要对齐 teacher 的训练格式。即便只是答案格式从普通文本改成 `\boxed{}`，也可能明显影响 OPD。

但只用 teacher-aligned prompt 会降低 student entropy，应混入 OOD prompt 保持泛化。

**长序列要小心**

过长 response 会导致 teacher 信号从后缀开始劣化，甚至向前传播。论文里较优区间大概是 3K-7K，10K-15K 可能平台化或下降。

* * *

### 12. k1/k2/k3：sampled-token 下的三种 KL 估计器

k1、k2、k3 都是对 reverse KL 的单样本估计器，常见于 Schulman 的 KL approximation 讨论。三者差异只在偏差、方差和非负性。

**12.1 reverse KL 的期望形式**

D_{KL}(P_S\|P_T) =\sum_{v\in\mathcal V}P_S(v|c_t)\log\frac{P_S(v|c_t)}{P_T(v|c_t)} =\mathbb E_{y\sim P_S}\left[\log\frac{P_S(y|c_t)}{P_T(y|c_t)}\right]\\

**12.2 k1：最直接的估计器**

k_1=\log\frac{P_S(y_t|c_t)}{P_T(y_t|c_t)},\quad y_t\sim P_S(\cdot|c_t)\\

对 KL 无偏；单样本可正可负，方差较高。作为 policy-gradient 的 advantage 使用时最自然。

**12.3 k2：平方近似**

k_2=\frac12\left(\log\frac{P_S(y_t|c_t)}{P_T(y_t|c_t)}\right)^2\\

始终非负、方差更小；对真实 KL 有偏，是其局部二阶近似。梯度在分布接近时更贴近真实 KL 的梯度。

**12.4 k3：无偏且非负**

k_3=\frac{P_T(y_t|c_t)}{P_S(y_t|c_t)}-\log\frac{P_T(y_t|c_t)}{P_S(y_t|c_t)}-1\\

利用恒等式\mathbb E_{y\sim P_S}[P_T(y)/P_S(y)]=1 即可得到 \mathbb E[k_3]=D_{KL}(P_S\|P_T)，同时 r-\log r-1\ge0。所以 k3 保留无偏性，同时非负、方差低于 k1。

**12.5 三者对比**

| estimator | 表达式 | 无偏 | 非负 | 方差 |
| --- | --- | --- | --- | --- |
| k1 | | 是 | 否 | 高 |
| k2 | | 否 | 是 | 低 |
| k3 | | 是 | 是 | 低 |

**12.6 verl 里的 k1/k2/k3 与 `k3+`**

verl 代码 `verl/trainer/ppo/core_algos.py`：

```
if kl_penalty in ("kl", "k1"):
  return logprob - ref_logprob

if kl_penalty in ("mse", "k2"):
  return 0.5 * (logprob - ref_logprob).square()

if kl_penalty in ("low_var_kl", "k3"):
  kl = ref_logprob - logprob
  ratio = torch.exp(kl)
  kld = ratio - kl - 1
```

verl 注释提醒：k1/k3 的数值期望等于 KL，但直接反传它们时，梯度期望不一定等于真实 KL 的梯度；k2 的梯度在局部更合适。所以 `k3+` 用 straight-through：forward 数值走 k3，backward 梯度走 k2。

```
backward_score = 0.5 * (logprob - ref_logprob).square()
return backward_score - backward_score.detach() + forward_score.detach()
```

**12.7 实践选择**

| 你能拿到的 teacher 信号 | 目标 |
| --- | --- |
| full-vocab logits | 对整个词表显式求和，写 full-vocab KL/JSD |
| teacher top-k | 在 top-k 上求和（可做重归一化），写 top-k KL/JSD |
| 只有 sampled token 的 logprob | 选择 k1/k2/k3 作为单样本估计器：PG 优先 k1，direct loss 优先 k3 或 k3+ |

* * *

### 13. DeepSeek V4 的 OPD：full-vocab multi-teacher reverse KL

DeepSeek V4 技术报告里明确说：

*  先训练多个 domain-specific expert；
*  再用 multi-teacher OPD 合并到最终模型；
*  student 在自己的 generated trajectories 上学习 teacher output distributions；
*  目标是多教师 reverse KL：

\mathcal L_{OPD}(\theta) =\sum_{i=1}^{N}w_i\cdot D_{KL}(\pi_\theta\|\pi_{E_i})\\

其中 \pi_{E_i} 是第 i 个 expert teacher。

报告还明确对比了 prior works：很多工作把 full-vocab KL 简化成 token-level KL estimate，再用 teacher-student logprob gap 当 advantage；DeepSeek V4 认为这样资源省，但梯度方差高、训练不稳定。

所以它采用：

> **full-vocabulary logit distillation。**

也就是说，DeepSeek V4 是：

y\sim\pi_\theta(\cdot|x)\\

c_t=(x,y_{<t})\\

\mathcal L_t=\sum_i w_i D_{KL}(P_S(\cdot|c_t)\|P_{E_i}(\cdot|c_t))\\

它不是 sampled-token，也不是 top-k。

**13.1 为什么它能做 full-vocab？**

full-vocab logits 很大，多个 teacher 更贵。DeepSeek V4 的工程方案是：

1.  teacher forward 时不显式缓存完整 logits；
2.  只缓存 teacher last-layer hidden states；
3.  训练时通过 teacher prediction head 动态重构 full logits；
4.  对 teacher 样本按 teacher index 排序，减少 head 加载；
5.  用专门 kernel 算 exact KL。

这就是它能做 full-vocab OPD 的关键。

* * *

### 14. SWIFT 的实现细节

调研代码：

```
ms-swift/swift/rlhf_trainers/gkd_trainer.py
ms-swift/swift/arguments/rlhf_args.py
ms-swift/docs/source/Instruction/GKD.md
ms-swift/examples/train/on_policy_distillation.sh
```

**14.1 SWIFT 的三种 prefix 来源**

SWIFT 的 `GKDTrainer.training_step()`：

```
if random() <= lmbda:
  data_source = STUDENT
  y = student.generate(x)
elif seq_kd:
  data_source = TEACHER
  y = teacher.generate(x)
else:
  data_source = DATASET
  y = y_data
```

所以：

| 参数 | prefix 来源 |
| --- | --- |
| lmbda=1 | student rollout，纯 OPD |
| lmbda=0, seq_kd=true | teacher generated，sequential KD |
| lmbda=0, seq_kd=false | dataset，off-policy KD |

**14.2 SWIFT 的 loss**

代码核心是 `generalized_jsd_loss()`。

如果 full-vocab：

```
teacher_output.full_logits
student_logits
```

如果 top-k：

```
teacher_logits, topk_idx = torch.topk(teacher_logits, k=topk)
student_logits = torch.gather(student_logits, dim=-1, index=topk_idx)
```

然后按 `beta` 算：

*  `beta=0`：forward KL；
*  `beta=1`：reverse KL；
*  中间：generalized JSD。

**14.3 SWIFT top-k 是重归一化的**

因为 gather 后又做 `log_softmax`，所以是在 top-k 子集上重归一化。这与 Revisiting OPD 的 support-set KL 接近。

**14.4 SWIFT 关键参数**

| 参数 | 作用 |
| --- | --- |
| --rlhf_type gkd | 开启 GKD |
| --teacher_model | 本地 teacher |
| --teacher_model_server | 外部 teacher API |
| --lmbda | student rollout 概率 |
| --seq_kd | 非 student 时是否 teacher generate |
| --beta | KL/JSD 方向 |
| --gkd_logits_topk | None=full-vocab；整数=teacher top-k |
| --temperature | rollout 与 loss 温度 |
| --sft_alpha | 非 student 数据混入 SFT loss |
| --max_completion_length | 最大生成长度 |
| --use_vllm | student rollout 加速 |
| --teacher_deepspeed | teacher ZeRO 配置 |
| --offload_teacher_model | teacher CPU offload |

**14.5 SWIFT 如何跑不同目标**

**full-vocab reverse KL OPD**

```
--rlhf_type gkd \
--lmbda 1 \
--beta 1 \
--gkd_logits_topk None
```

数学：

\sum_tD_{KL}(P_S(\cdot|c_t)\|P_T(\cdot|c_t)),\quad c_t=(x,y^{student}_{<t})\\

**teacher top-k reverse KL OPD**

```
--rlhf_type gkd \
--lmbda 1 \
--beta 1 \
--gkd_logits_topk 64
```

数学：

\sum_tD_{KL}(\hat P_S^{TopK_T}\|\hat P_T^{TopK_T})\\

* * *

### 15. verl 主仓实现细节

调研代码：

```
verl/verl/workers/config/distillation.py
verl/verl/trainer/config/distillation/distillation.yaml
verl/verl/trainer/distillation/losses.py
verl/verl/trainer/distillation/fsdp/losses.py
verl/verl/trainer/distillation/megatron/losses.py
verl/verl/trainer/ppo/core_algos.py
```

**15.1 verl 支持两类 distillation**

**A. `forward_kl_topk`**

teacher 返回：

```
teacher_logprobs: [B, L, K]
teacher_ids: [B, L, K]
```

student logits 是 full vocab：

`student_logits: [B, L, V]`

代码：

```
student_log_probs = F.log_softmax(student_logits, dim=-1)
student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
distillation_losses = KL(teacher_topk_log_probs || student_topk_log_probs)
```

数学：

\sum_{v\in TopK_T}P_T(v|c_t)\left[\log P_T(v|c_t)-\log P_S(v|c_t)\right]\\

这是 teacher top-k truncated forward KL。

注意：verl 主仓这个版本不是重归一化 support-set KL。它会记录 `teacher_mass`、`student_mass`，并对负值 clamp。

**B. `k1/k2/k3` sampled-token estimator**

teacher 和 student 都只看 sampled token 的 logprob：

\log P_S(y_t|c_t),\quad \log P_T(y_t|c_t)\\

然后用 k1/k2/k3 算 sampled-token KL estimator。

**15.2 `use_policy_gradient` 是核心开关**

如果：

`use_policy_gradient: true`

代码会把负 distillation loss 当 advantage：

`advantages = -distillation_losses.detach()`

这就是 TML-style OPD。

如果：

`use_policy_gradient: false`

直接把 distillation loss 聚合反传，类似 GKD direct loss。

**15.3 verl 主仓配置建议**

**TML sampled-token OPD**

```
distillation.enabled: true
distillation.distillation_loss.loss_mode: k1
distillation.distillation_loss.use_policy_gradient: true
distillation.distillation_loss.use_task_rewards: false
```

**sampled-token low-var direct loss**

```
distillation.enabled: true
distillation.distillation_loss.loss_mode: k3
distillation.distillation_loss.use_policy_gradient: false
```

**top-k distribution matching**

```
distillation.enabled: true
distillation.distillation_loss.loss_mode: forward_kl_topk
distillation.distillation_loss.topk: 64
distillation.distillation_loss.use_policy_gradient: false
```

* * *

### 16. verl-recipe/gkd：大规模 Megatron top-k OPD

调研代码：

```
verl-recipe/gkd/megatron/README.md
verl-recipe/gkd/megatron/megatron_distill_losses.py
verl-recipe/gkd/megatron/megatron_workers.py
verl-recipe/gkd/megatron/config/on_policy_distill_trainer.yaml
verl-recipe/gkd/megatron/teacher/worker.py
```

**16.1 数据流**

1.  student rollout worker 用 vLLM 生成 response；
2.  teacher server 返回 top-k logprobs 和 indices；
3.  Megatron actor forward；
4.  pipeline 最后一 stage 用 `logits_processor` 计算 KL；
5.  反传更新 student。

teacher worker 参数：

`--n-logprobs 256`

说明它是 teacher top-k，不是 full-vocab。

**16.2 支持四类 loss**

`megatron_distill_losses.py` 支持：

```
kl
rkl
kl_rkl
jsd
```

**`kl`**

KL(P_{T,topk}\|P_{S,full})\\

teacher top-k truncated forward KL。

**`rkl`**

top-k 内重归一化 reverse KL：

KL(\hat P_{S,topk}\|\hat P_{T,topk})\\

这和 Revisiting OPD 的 local support reverse KL 很接近。

**`kl_rkl`**

(1-r)KL+rRKL\\

参数是 `rkl_ratio`。

**`jsd`**

JSD beta，并且对 student top-k 外的 mass 做 analytic rest term，不是简单丢掉。

**16.3 recipe 参数**

| 参数 | 作用 |
| --- | --- |
| teacher --n-logprobs | teacher top-k 数 |
| teacher --tp-size | teacher tensor parallel |
| actor_rollout_ref.actor.distill_loss.name | kl/rkl/kl_rkl/jsd |
| distill_loss.rkl_ratio | KL_RKL 混合比例 |
| distill_loss.beta | JSD beta |
| rollout temperature | student rollout 温度 |
| rollout top_p | student rollout top-p |
| rollout top_k | student rollout top-k |
| trainer.scheduler | one_step_off / two_step_off |
| use_dynamic_bsz | 动态 batch |
| max_token_len | 动态 batch token 上限 |

* * *

### 17. SWIFT vs verl：怎么选？

| 目标 | 推荐框架 | 配置 |
| --- | --- | --- |
| 最像 DeepSeek V4 单教师 full-vocab reverse KL | SWIFT | lmbda=1,beta=1,gkd_logits_topk=None |
| 显存有限的 top-k reverse KL | SWIFT | lmbda=1,beta=1,gkd_logits_topk=64 |
| TML sampled-token OPD | verl 主仓 | loss_mode=k1,use_policy_gradient=true |
| sampled-token 低方差 direct loss | verl 主仓 | loss_mode=k3,use_policy_gradient=false |
| top-k forward KL | verl 主仓 | loss_mode=forward_kl_topk,topk=64 |
| 大规模 Megatron top-k RKL | verl-recipe/gkd | distill_loss.name=rkl |
| top-k KL/RKL/JSD 对比实验 | verl-recipe/gkd | kl/rkl/kl_rkl/jsd |

* * *

### 18. OPD 实际应用的局限

综合 Revisiting OPD、Rethinking OPD 和工程实现，OPD 的局限主要有这些。

**18.1 teacher 不可靠时，dense signal 反而有害**

OPD 的优势是 dense token-level signal。但如果 teacher 在 student prefix 上不可靠，dense signal 只是更密集地提供错误监督。

尤其是长序列、agent、多轮工具调用，student prefix 很容易偏离 teacher 熟悉分布。

**18.2 sampled-token 信息量不足**

sampled-token 每个 prefix 只看一个 token。它可能知道当前采样 token 不好，但不知道 teacher 真正想要哪个 token。

这就是 sampled-token 方差和不稳定性的来源。

**18.3 top-k 有截断偏差**

top-k 更稳定，但它忽略 teacher top-k 外的 token。即使重归一化，也改变了原始 full-vocab KL 目标。

**18.4 full-vocab 太贵**

DeepSeek V4 能做 full-vocab，是因为它有专门工程：hidden-state caching、prediction head 重构、teacher scheduling、specialized kernel。一般团队很难直接复现。

**18.5 thinking pattern mismatch**

如果 teacher 和 student 的思维模式不一致，OPD 不一定有效。更强 teacher 未必更好，关键是 high-prob token overlap 和 entropy gap。

**18.6 OPD 不能替代 RL**

OPD 擅长把 teacher 已经发现的能力迁移给 student。它不擅长发现 teacher 不会的新策略。

更合理 pipeline 是：

1.  用 RL / verifier 训练 expert；
2.  用 OPD 把 expert 能力蒸回 general model；
3.  再迭代。

* * *

### 19. 实践建议：如果要自己跑

**19.1 最小实验矩阵**

我建议至少跑：

| 实验 | 目的 |
| --- | --- |
| SFT baseline | 看 off-policy hard label 基线 |
| SWIFT top-k RKL | 看稳定 top-k OPD |
| SWIFT full-vocab RKL | 如果资源够，看 full-vocab 上限 |
| verl k1 PG | 看 TML sampled-token OPD |
| verl k3 direct | 看 sampled-token low-var direct |
| verl top-k forward KL | 看 top-k distribution matching |

**19.2 监控指标**

不要只看 loss。要监控：

*  task accuracy / pass@1 / pass@k；
*  response length；
*  length clipping；
*  entropy；
*  teacher-student logprob gap；
*  top-k overlap；
*  gradient norm；
*  特殊 token 比例；
*  OOD prompts 上的退化。

**19.3 超参建议**

| 参数 | 建议 |
| --- | --- |
| rollout top-p | 0.8-0.95 起步 |
| rollout temperature | 0.7-1.0 |
| top-k | 32/64/128 做 ablation |
| response length | 先别太长，3K-7K 区间更稳 |
| beta | reverse KL 用 1；JSD 用 0.5； |
| cold start | teacher/student 差距大时必须做 |
| special token | tokenizer 不同必须 mask/对齐 |

* * *

### 20. 最后总结

OPD 不是单一算法，而是一族方法。真正决定它行为的是：

1.  prefix 是谁生成的？
2.  teacher 给 sampled token、top-k 还是 full-vocab？
3.  KL 方向是 forward、reverse 还是 JSD？
4.  loss 是 direct backprop 还是 policy-gradient advantage？
5.  teacher 和 student 的 thinking pattern 是否一致？

如果你把这些问题拆开，很多争论就清楚了：

*  sampled-token OPD 省，但信号稀疏、方差高；
*  top-k OPD 是性价比最高的折中；
*  full-vocab OPD 最完整，但工程极贵；
*  reverse KL 适合推理/代码这类 mode-seeking 任务；
*  forward KL 更保守、更覆盖；
*  JSD 是折中；
*  OPD 需要 teacher 有新能力，也需要 teacher/student 思维模式一致；
*  OPD 更像 post-training glue，不是 RL 的替代品。

如果要用一句话概括这篇文章：

> **OPD 的核心价值，是让模型在自己会遇到的状态上，接受来自更强模型的密集监督；而 OPD 的核心风险，是这个密集监督可能在错误 prefix、错误 token 粒度、错误 KL 方向下变成密集噪声。**

* * *

### 参考资料

1.  Gu et al., _MiniLLM: On-Policy Distillation of Large Language Models_, arXiv:2306.08543.
2.  Agarwal et al., _On-policy Distillation of Language Models: Learning from Self-Generated Mistakes_, ICLR 2024 / arXiv:2306.13649.
3.  Fu et al., _Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes_, arXiv:2603.25562.
4.  _Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe_, arXiv:2604.13016.
5.  _Entropy-Aware On-Policy Distillation of Language Models_, arXiv:2603.07079.
6.  Thinking Machines Lab, _On-Policy Distillation_.
7.  DeepSeek, _DeepSeek V4 Technical Report_.
8.  ModelScope ms-swift: `swift/rlhf_trainers/gkd_trainer.py`, `docs/source/Instruction/GKD.md`.
9.  verl: `verl/trainer/distillation/losses.py`, `verl/workers/config/distillation.py`, `verl/trainer/ppo/core_algos.py`.
10.  verl-recipe: `gkd/megatron/megatron_distill_losses.py`, `gkd/megatron/README.md`.

> 本文由 [Zhihu on Obsidian](https://zhuanlan.zhihu.com/p/1901622331102696374) 创作并发布
