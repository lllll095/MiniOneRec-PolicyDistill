# Interview Notes

This file is intentionally written in Chinese for interview preparation.

## 一、30 秒版本

我做的是一个生成式推荐项目，基于 MiniOneRec。先完整复现了 0.5B 模型的 SFT 和 GRPO，HR@10 从 12.84% 提升到 13.41%。

之后我分析训练过程，发现 GRPO 的推荐 reward 比较稀疏，而官方 1.5B 模型明显更强，所以我尝试把强模型在 SID 合法动作空间上的完整 policy distribution 蒸馏给 0.5B student。

最终用 constrained policy distillation 把 0.5B 的 HR@10 做到 14.41%，NDCG@10 做到 10.02%，HR@50 做到 24.09%，明显超过原始 GRPO，并且 HR@50 已经接近 1.5B teacher 的 24.47%。

---

## 二、90 秒版本

这个项目是在 MiniOneRec 上做的生成式推荐项目。

第一步我先严格复现它的 0.5B SFT 和 GRPO。SFT 的 HR@10 是 12.84%，做完官方风格的 GRPO 后提高到 13.41%。

然后我去分析为什么 GRPO 的提升有限。MiniOneRec 的推荐 reward 是 sequence-level 的，而且 exact target hit 比较稀疏。在我们的诊断里，大量 completion group 里没有真实 item，因此能提供的 exact recommendation signal 有限。

与此同时，官方 1.5B 模型 HR@10 能到 15.77%，明显强于 0.5B。因此我考虑能不能不只给 student 一个稀疏的 reward，而是直接把强 teacher 的 recommendation policy 蒸馏下来。

MiniOneRec 的一个特点是它不是在整个自然语言词表上自由生成，而是通过 SID trie 限制每一步合法 token。所以我把 teacher 和 student 的 logits 都限制到当前 legal SID action set，然后在这个小的动作空间上做 exact reverse KL。

我实现了两个版本。一个是在 ground-truth SID prefix 上做 Gold-KD，另一个是在 student 自己生成的 prefix 上做 on-policy distillation。完整训练后，Gold-KD 的 HR@10 是 14.25%，OPD 是 14.41%。两者都明显超过 GRPO，因此我最后的结论不是“OPD 本身贡献了全部提升”，而是主要收益来自 strong teacher 的 dense policy transfer，on-policy state 只提供了小幅额外收益。

我还做了一个负实验：在 OPD 后面继续做 GRPO，NDCG@10 和 HR@50 反而下降，所以没有继续堆模块。我把这个结果解释为 sparse RL 可能会部分破坏已经蒸馏得到的 dense teacher policy。

---

## 三、5 分钟完整讲法

### 1. 原始问题

MiniOneRec 把推荐转成 hierarchical Semantic ID 的自回归生成问题。

每个 item 对应一个三层 SID：

```text
[SID-1] -> [SID-2] -> [SID-3]
```

生成时使用 catalogue trie，只允许生成有效 item 对应的 SID 路径。

我先复现 0.5B SFT：

```text
HR@10   = 12.84%
NDCG@10 = 8.63%
HR@50   = 21.71%
```

再复现 GRPO：

```text
HR@10   = 13.41%
NDCG@10 = 9.24%
HR@50   = 22.35%
```

### 2. 我发现的问题

我没有直接开始改 reward，而是先分析训练过程。

GRPO 每个 prompt 生成 16 个 completion，但是 exact recommendation reward 很稀疏。很多 group 一个真实 target 都没有。

因此 supervision 更像：

```text
generate many complete SID sequences
        ↓
few target hits
        ↓
sparse sequence-level reward
```

这意味着大量中间 SID 决策没有 dense supervision。

### 3. 为什么想到 teacher distillation

官方 1.5B checkpoint 的结果明显更强：

```text
HR@10   = 15.77%
NDCG@10 = 11.75%
HR@50   = 24.47%
```

所以问题变成：

> 能不能直接把 1.5B 的推荐 policy 迁移给 0.5B？

但进一步诊断后我发现，teacher 并不是简单地在每一个 SID token 上 Top-1 更准。

例如 gold-prefix 下：

```text
L1 Top-1:
student 27.8%
teacher 26.2%
```

teacher 最终推荐却明显更好。

所以我不想做 hard pseudo-label，而是保留 teacher 对多个合法 candidate 的相对 preference。

### 4. 方法

当前 state 为 `s` 时，SID trie 给出合法动作集合 `A(s)`。

我只在这个集合上归一化 teacher/student policy，然后优化：

```text
KL(student || teacher)
```

优势是：

- supervision 是 dense 的；
- 只关注 recommendation-relevant actions；
- legal action 数远小于完整 vocabulary；
- 可以做 exact full-distribution KL。

### 5. KD 和 OPD

我实现了两个版本。

Gold-KD：

```text
state = ground-truth SID prefix
```

OPD：

```text
state = student-generated SID prefix
```

其他设置全部保持一致。

最终：

```text
Gold-KD:
HR@10   = 14.25%
NDCG@10 = 9.98%
HR@50   = 24.02%

OPD:
HR@10   = 14.41%
NDCG@10 = 10.02%
HR@50   = 24.09%
```

所以两者都明显超过 GRPO。

### 6. 我最后怎么解释结果

我不会说“OPD 显著优于 KD”，因为差异很小。

更准确的结论是：

> 主要提升来自 strong teacher 的 constrained policy distribution；on-policy state 可能有小幅额外帮助，但不是主要来源。

### 7. 负实验

我又测试：

```text
SFT -> OPD -> GRPO
```

结果：

```text
HR@10   14.41% -> 14.32%
NDCG@10 10.02% -> 9.68%
HR@50   24.09% -> 23.49%
```

所以我没有继续堆叠 GRPO。

这个结果说明：dense distillation 和 sparse RL 并不一定简单可加，后续 RL 可能会改变已经学到的 teacher ranking structure。

---

## 四、必须会回答的高频问题

### Q1. 为什么不用普通 cross-entropy 蒸馏？

因为我们希望保留 teacher 对合法候选之间的相对 preference，而不是只学习一个 hard target。

实验也说明 teacher 的优势不是简单的 greedy token Top-1 accuracy。

因此完整 policy distribution 比 argmax pseudo-label 更符合问题。

### Q2. 为什么不在整个 vocabulary 上做 KL？

因为 MiniOneRec 的推荐决策已经被 SID trie 限制。

当前 state 真正有意义的动作只有 `A(s)` 中的合法 SID token。

完整 vocabulary 里绝大多数 token 对推荐决策没有意义。

而且合法动作通常只有几十个，因此可以精确计算 full legal-action KL。

### Q3. 为什么用 reverse KL？

当前项目把 reverse KL 作为 policy distillation baseline。

它会更偏向 teacher 的高概率 mode，因此我们同时观察了 student entropy。

我们不声称 reverse KL 一定是最优选择，只是它构成了一个干净、明确的 baseline。

### Q4. KD 和 OPD 的区别是什么？

唯一核心区别是 state distribution。

Gold-KD：

```text
s_t = (x, gold prefix)
```

OPD：

```text
s_t = (x, student prefix)
```

loss、teacher、student、数据和优化设置保持一致。

### Q5. 为什么 OPD 没有明显碾压 KD？

SID recommendation 和开放式 reasoning 不完全一样。

如果第一层 SID 走错，就进入了另一个 catalogue subtree，后面通常不能再恢复到真实 item。

因此“在 student 错误 state 上继续指导”在这里没有 reasoning 任务里那么大的价值。

实验也支持这一点：KD 和 OPD 都有效，差异很小。

### Q6. 为什么 OPD 后继续 GRPO 反而下降？

我们观察到 NDCG@10 和 HR@50 都下降。

一个合理解释是 OPD 提供 dense teacher policy，而 GRPO 的 sequence reward 相对稀疏。

后续 RL 更新可能提升一部分局部 reward，但同时改变更广泛的 candidate ranking structure。

这是实验解释，不是理论证明。

### Q7. 这个项目真正的创新/改进是什么？

对面试项目来说，我不会把它包装成论文式“全新算法”。

真正的工程改进是：

1. 复现 MiniOneRec；
2. 从训练日志诊断 sparse reward bottleneck；
3. 利用 SID trie 构造 legal-action policy distillation；
4. 系统比较 Gold-KD 和 OPD；
5. 用负实验验证 OPD 和 GRPO 不能简单堆叠。

### Q8. 为什么 HR@50 已经接近 teacher，但是 NDCG@10 还有差距？

这说明 student 已经很好地学到了 candidate coverage，但是更精细的 top-ranking 仍然没有完全复制 teacher。

也就是说：

```text
candidate recall transfer: strong
fine-grained ranking transfer: incomplete
```

这是当前方法的一个重要 limitation。

---

## 五、简历建议

### 项目名称

生成式推荐系统 MiniOneRec 复现与策略蒸馏优化

### 三条版本

- 基于 MiniOneRec 完成 0.5B 生成式推荐模型 SFT 与 GRPO 全流程复现，覆盖 Semantic ID、层级自回归生成、Trie 约束 Beam Search 与 HR/NDCG 评测；在 Amazon Industrial & Scientific 上复现 HR@10 12.84% -> 13.41%。
- 针对 GRPO 序列级 reward 稀疏问题，引入 1.5B 强模型进行 constrained SID policy distillation，在 Trie 合法动作空间上计算 exact reverse-KL，并实现 Gold-prefix KD 与 student-state on-policy distillation。
- 完整训练后 0.5B student 达到 HR@10 14.41%、NDCG@10 10.02%、HR@50 24.09%，明显优于 GRPO 的 13.41% / 9.24% / 22.35%；通过 KD/OPD 及 OPD->GRPO 消融分析定位主要收益来源和方法边界。

---

## 六、不要说过头的话

不要说：

- “OPD 显著优于 KD”
- “我们首次把 OPD 用于推荐”
- “证明了 GRPO 会破坏 teacher policy”
- “这个方法普遍优于 GRPO”
- “teacher 每一层都更准确”

可以说：

- “OPD 在当前单 seed 下略优于 Gold-KD”
- “主要收益来自 dense teacher policy transfer”
- “OPD->GRPO 的实验结果支持 partial degradation 的解释”
- “当前实验集中在一个 Amazon category”
- “这是一个面试导向的工程研究项目，不声称新论文方法”
