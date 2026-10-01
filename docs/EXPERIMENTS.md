# Experiments

This document records the main experimental path from MiniOneRec reproduction to constrained policy distillation.

## 1. 0.5B SFT Reproduction

Base model:

```text
Qwen2.5-0.5B-Instruct
```

Main settings:

```text
maximum epochs       = 10
learning rate        = 3e-4
effective batch size = 1024
seed                 = 42
```

Training stopped after epoch 5.5, with the best validation loss around epoch 4.

Full-test beam-50 results:

```text
HR@10    = 0.12839179
NDCG@10  = 0.08626247
HR@50    = 0.21707478
```

This checkpoint is used as the common initialization for the GRPO, Gold-KD, and OPD experiments.

---

## 2. GRPO Reproduction

We reproduced the MiniOneRec recommendation-oriented GRPO configuration.

Key settings:

```text
num_generations          = 16
reward_type              = ranking
beam_search              = True
temperature              = 1.0
learning_rate            = 1e-5
beta                     = 1e-3
sync_ref_model           = True
dynamic_sampling         = False
effective batch size     = 1024
epochs                   = 2
```

Full-test beam-50 results:

```text
HR@10    = 0.13412751
NDCG@10  = 0.09239487
HR@50    = 0.22347231
```

GRPO improves over SFT, but the exact recommendation reward remains sparse.

---

## 3. GRPO Reward Diagnostics

Across the formal GRPO run:

```text
mean rule reward      ≈ 0.0189
mean rank reward      ≈ -0.0167
mean reward std       ≈ 0.0805
```

With 16 generated candidates per completion group, roughly 30% of groups contained an exact target hit in the diagnostic run.

This motivated using denser policy supervision from a stronger teacher.

---

## 4. Teacher Evaluation

The official 1.5B MiniOneRec checkpoint achieves:

```text
HR@10    = 0.15773219
NDCG@10  = 0.11748187
HR@50    = 0.24465034
```

The teacher therefore provides a substantial performance gap relative to the reproduced 0.5B SFT model.

---

## 5. Teacher / Student Policy Diagnostics

Under gold SID prefixes, the teacher is not uniformly better as a next-token classifier.

```text
L1 Top-1:
student = 0.278
teacher = 0.262

L2 Top-1:
student = 0.224
teacher = 0.202

L3 Top-1:
student = 0.654
teacher = 0.644
```

Teacher entropy is much lower at the early SID layers:

```text
L1 entropy:
student = 2.8465
teacher = 1.2379

L2 entropy:
student = 2.7778
teacher = 2.0923
```

The teacher also has stronger final beam-search HR/NDCG.

The working interpretation is therefore that the teacher's advantage is better characterized by constrained sequence-level ranking than by greedy next-token accuracy.

---

## 6. Policy Distillation

At state `s`, the catalogue trie defines the valid next SID tokens `A(s)`.

Student and teacher distributions are normalized only over this legal action set.

The training objective is exact legal-space reverse KL:

```text
KL(student || teacher)
```

Two state distributions are compared.

### Gold-Prefix KD

Training states follow the ground-truth SID prefix.

### OPD

Training states follow student-generated SID prefixes.

---

## 7. 200-Step Pilot

The initial 200-step experiments showed that both methods were promising:

```text
Gold-KD HR@10 = 0.13501
OPD     HR@10 = 0.13655
```

This motivated approximately one full epoch of distillation.

---

## 8. Full-Epoch Distillation

Training mixture:

```text
SidDataset             = 36,259
RLTitle2SidDataset     = 6,516
RLSeqTitle2SidDataset  = 10,000
total                  = 52,775
```

With batch size 32:

```text
approximately one epoch = 1,649 optimizer steps
```

### Gold-KD

```text
HR@10    = 0.14251048
NDCG@10  = 0.09980349
HR@50    = 0.24023825
```

### OPD

```text
HR@10    = 0.14405471
NDCG@10  = 0.10021888
HR@50    = 0.24090007
```

Both substantially outperform SFT and GRPO.

The difference between Gold-KD and OPD is comparatively small.

---

## 9. OPD Followed by GRPO

We tested whether GRPO could further refine the distilled policy:

```text
SFT -> OPD -> GRPO
```

Results:

```text
HR@10    = 0.14317229
NDCG@10  = 0.09680339
HR@50    = 0.23494375
```

Compared with OPD alone:

```text
HR@10    0.14405 -> 0.14317
NDCG@10  0.10022 -> 0.09680
HR@50    0.24090 -> 0.23494
```

Sequential GRPO therefore does not provide an additional improvement and partially reduces the gains from dense teacher policy distillation.

---

## 10. Final Comparison

| Method | HR@10 | NDCG@10 | HR@50 |
|---|---:|---:|---:|
| 0.5B SFT | 0.1284 | 0.0863 | 0.2171 |
| 0.5B SFT -> GRPO | 0.1341 | 0.0924 | 0.2235 |
| 0.5B SFT -> Gold-KD | 0.1425 | 0.0998 | 0.2402 |
| 0.5B SFT -> OPD | 0.1441 | 0.1002 | 0.2409 |
| 0.5B OPD -> GRPO | 0.1432 | 0.0968 | 0.2349 |
| 1.5B Teacher | 0.1577 | 0.1175 | 0.2447 |

---

## 11. Final Takeaways

1. Dense constrained policy distillation is substantially more effective than the reproduced GRPO baseline for the 0.5B student.
2. Gold-prefix KD and OPD perform similarly, suggesting that most of the gain comes from teacher policy transfer rather than on-policy states alone.
3. Applying sparse GRPO after distillation can partially degrade the transferred policy.
4. The strongest current 0.5B result is the OPD model with HR@10 = 0.1441, NDCG@10 = 0.1002, and HR@50 = 0.2409.
