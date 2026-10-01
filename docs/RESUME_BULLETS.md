# Resume Bullets

## 中文版

**生成式推荐系统 MiniOneRec 复现与策略蒸馏优化**

- 基于 MiniOneRec 完成 0.5B 生成式推荐模型 SFT 与 GRPO 全流程复现，覆盖 Semantic ID、层级自回归生成、Trie 约束 Beam Search 与 HR/NDCG 评测；在 Amazon Industrial & Scientific 上复现 HR@10 12.84% -> 13.41%。
- 针对 GRPO 序列级 reward 稀疏问题，引入 1.5B 强模型进行 constrained SID policy distillation，在 Trie 合法动作空间上计算 exact reverse-KL，并实现 Gold-prefix KD 与 student-state on-policy distillation。
- 完整训练后 0.5B student 达到 **HR@10 14.41%、NDCG@10 10.02%、HR@50 24.09%**，明显优于原始 GRPO 的 13.41% / 9.24% / 22.35%；通过 KD/OPD 及 OPD->GRPO 消融分析定位主要收益来源和方法边界。

## English Version

**MiniOneRec Reproduction & Policy Distillation for Generative Recommendation**

- Reproduced the 0.5B MiniOneRec SFT/GRPO pipeline, including hierarchical Semantic ID generation, trie-constrained beam search, and HR/NDCG evaluation on Amazon Industrial & Scientific.
- Diagnosed sparse sequence-level GRPO supervision and implemented constrained policy distillation from a stronger 1.5B teacher, computing exact reverse-KL over the legal SID action space with both gold-prefix and on-policy state distributions.
- Improved the 0.5B model to **HR@10 14.41%, NDCG@10 10.02%, HR@50 24.09%**, outperforming the reproduced GRPO baseline; used KD/OPD and OPD->GRPO ablations to identify the main source of gains and failure modes.
