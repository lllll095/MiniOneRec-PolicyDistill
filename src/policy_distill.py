"""
Constrained policy distillation for hierarchical generative recommendation.

state_mode=gold:
    distill teacher policy on ground-truth SID prefixes.

state_mode=on_policy:
    distill teacher policy on student-sampled SID prefixes.

Objective:
    exact reverse KL over the legal SID action set defined by the catalogue trie.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import bitsandbytes as bnb
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from .data import SidDataset, RLTitle2SidDataset, RLSeqTitle2SidDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / max(len(xs), 1))


def entropy_from_logp(logp: torch.Tensor) -> torch.Tensor:
    p = logp.exp()
    return -(p * logp).sum()


def build_catalogue(index_file: str, tokenizer):
    with open(index_file, "r", encoding="utf-8") as f:
        index = json.load(f)

    trie = {}
    sid_to_ids = {}
    sid_tokens_all = set()
    lengths = []

    for _, sid_tokens in index.items():
        sid_string = "".join(sid_tokens)
        ids = [tokenizer.convert_tokens_to_ids(tok) for tok in sid_tokens]

        if any(x is None or x == tokenizer.unk_token_id for x in ids):
            raise RuntimeError(f"Invalid SID tokenization: {sid_tokens} -> {ids}")

        sid_to_ids[sid_string] = ids
        sid_tokens_all.update(sid_tokens)
        lengths.append(len(ids))

        node = trie
        for tid in ids:
            node = node.setdefault(tid, {})

    unique_lengths = sorted(set(lengths))
    if len(unique_lengths) != 1:
        raise RuntimeError(
            f"Variable SID depths detected: {unique_lengths}; fixed-depth SIDs are required."
        )

    depth = unique_lengths[0]

    print("Catalogue SID lengths :", unique_lengths)
    print("Unique SID strings    :", len(sid_to_ids))
    print("SID depth             :", depth)
    print("Root legal actions    :", len(trie))

    return trie, sid_to_ids, depth, sorted(sid_tokens_all)


def verify_sid_token_compatibility(student_tokenizer, teacher_tokenizer, sid_tokens):
    mismatches = []
    for tok in sid_tokens:
        s_id = student_tokenizer.convert_tokens_to_ids(tok)
        t_id = teacher_tokenizer.convert_tokens_to_ids(tok)
        if s_id != t_id:
            mismatches.append((tok, s_id, t_id))

    if mismatches:
        raise RuntimeError(
            "Teacher/student SID token IDs are incompatible. "
            f"First mismatches: {mismatches[:10]}"
        )

    print("SID token compatibility: OK")
    print("SID token count        :", len(sid_tokens))


def get_legal_actions(trie, prefix):
    node = trie
    for tid in prefix:
        if tid not in node:
            raise RuntimeError(f"Illegal SID prefix encountered: {prefix}")
        node = node[tid]
    return list(node.keys())


def build_training_records(train_file, item_file, index_file, category, seqtitle_sample, seed):
    print("\nBuilding MiniOneRec RL prompt mixture...")

    ds_rec = SidDataset(
        train_file,
        category=category,
        sample=-1,
        seed=seed,
    )
    ds_title = RLTitle2SidDataset(
        item_file=item_file,
        index_file=index_file,
        category=category,
        sample=-1,
        seed=seed,
    )
    ds_seqtitle = RLSeqTitle2SidDataset(
        train_file,
        category=category,
        sample=seqtitle_sample,
        seed=seed,
    )

    rec = [x for x in ds_rec.get_inputs_list() if x is not None]
    title = [x for x in ds_title.get_inputs_list() if x is not None]
    seqtitle = [x for x in ds_seqtitle.get_inputs_list() if x is not None]

    print("SidDataset            :", len(rec))
    print("RLTitle2SidDataset    :", len(title))
    print("RLSeqTitle2SidDataset :", len(seqtitle))

    records = rec + title + seqtitle
    rng = random.Random(seed)
    rng.shuffle(records)

    print("Total distill prompts :", len(records))
    return records


def detect_logits_to_keep(model) -> Optional[str]:
    sig = inspect.signature(model.forward)
    if "logits_to_keep" in sig.parameters:
        return "logits_to_keep"
    if "num_logits_to_keep" in sig.parameters:
        return "num_logits_to_keep"
    return None


def get_last_logits(model, input_ids, attention_mask, logits_keep_arg):
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
    }
    if logits_keep_arg is not None:
        kwargs[logits_keep_arg] = 1
    return model(**kwargs).logits[:, -1, :]


def parse_args():
    p = argparse.ArgumentParser(description="Constrained SID policy distillation")

    p.add_argument("--student", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--train_file", required=True)
    p.add_argument("--index_file", required=True)
    p.add_argument("--item_file", required=True)

    p.add_argument("--category", default="industrial and scientific items")
    p.add_argument("--seqtitle_sample", type=int, default=10000)

    p.add_argument(
        "--state_mode",
        choices=["gold", "on_policy"],
        default="on_policy",
    )

    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_steps", type=int, default=1649)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--rollout_temperature", type=float, default=1.0)
    p.add_argument("--distill_temperature", type=float, default=1.0)
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--student_device", default="cuda:0")
    p.add_argument("--teacher_device", default="cuda:1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_final", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    if args.rollout_temperature <= 0 or args.distill_temperature <= 0:
        raise ValueError("Temperatures must be > 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the reported training setup.")

    set_seed(args.seed)

    student_device = torch.device(args.student_device)
    teacher_device = torch.device(args.teacher_device)

    print("=" * 80)
    print("CONSTRAINED POLICY DISTILLATION")
    print("=" * 80)
    print("Student              :", args.student)
    print("Teacher              :", args.teacher)
    print("State mode           :", args.state_mode)
    print("Objective            : exact legal-space reverse KL")
    print("Max steps            :", args.max_steps)
    print("Batch size           :", args.batch_size)
    print("Learning rate        :", args.learning_rate)
    print("=" * 80)

    student_tokenizer = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)

    student_tokenizer.padding_side = "left"
    teacher_tokenizer.padding_side = "left"

    if student_tokenizer.pad_token_id is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
    if teacher_tokenizer.pad_token_id is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    trie, sid_to_ids, depth, sid_tokens = build_catalogue(
        args.index_file,
        student_tokenizer,
    )
    verify_sid_token_compatibility(
        student_tokenizer,
        teacher_tokenizer,
        sid_tokens,
    )

    records = build_training_records(
        args.train_file,
        args.item_file,
        args.index_file,
        args.category,
        args.seqtitle_sample,
        args.seed,
    )

    if len(records) < args.batch_size:
        raise RuntimeError("Training set smaller than batch size.")

    print("\nLoading student...")
    student = AutoModelForCausalLM.from_pretrained(
        args.student,
        torch_dtype=torch.bfloat16,
    ).to(student_device)
    student.config.use_cache = False
    student.train()

    print("Loading frozen teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        torch_dtype=torch.bfloat16,
    ).to(teacher_device)
    teacher.config.use_cache = False
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    s_logits_arg = detect_logits_to_keep(student)
    t_logits_arg = detect_logits_to_keep(teacher)

    print("Student logits slicing:", s_logits_arg)
    print("Teacher logits slicing:", t_logits_arg)

    optimizer = bnb.optim.PagedAdamW32bit(
        student.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(args.max_steps * 0.03)),
        num_training_steps=args.max_steps,
    )

    out_dir = Path(args.output_dir)
    if args.save_final:
        out_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(
        f"policy_distill_{args.state_mode}_{args.max_steps}steps.jsonl"
    )
    if log_path.exists():
        log_path.unlink()

    rng = random.Random(args.seed)
    ptr = 0
    start_time = time.time()

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(
        range(1, args.max_steps + 1),
        total=args.max_steps,
        desc="Training",
        dynamic_ncols=True,
        mininterval=1.0,
        leave=True,
    )

    for step in pbar:
        if ptr + args.batch_size > len(records):
            rng.shuffle(records)
            ptr = 0

        batch = records[ptr : ptr + args.batch_size]
        ptr += args.batch_size

        prompts = [x["prompt"] for x in batch]
        gold_ids = [
            sid_to_ids.get(x["completion"].strip(" \n\""))
            for x in batch
        ]

        if args.state_mode == "gold" and any(g is None for g in gold_ids):
            raise RuntimeError("A gold SID target was not found in the catalogue.")

        enc = student_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
            max_length=args.max_prompt_length,
            add_special_tokens=False,
        )

        ids_cpu = enc["input_ids"]
        mask_cpu = enc["attention_mask"]

        prefixes = [[] for _ in batch]
        step_stats = defaultdict(list)

        for level in range(depth):
            ids_s = ids_cpu.to(student_device, non_blocking=True)
            mask_s = mask_cpu.to(student_device, non_blocking=True)
            ids_t = ids_cpu.to(teacher_device, non_blocking=True)
            mask_t = mask_cpu.to(teacher_device, non_blocking=True)

            s_last = get_last_logits(student, ids_s, mask_s, s_logits_arg)

            with torch.inference_mode():
                t_last = get_last_logits(teacher, ids_t, mask_t, t_logits_arg)

            kl_losses = []
            next_tokens = []

            for i in range(len(batch)):
                legal = get_legal_actions(trie, prefixes[i])
                if not legal:
                    raise RuntimeError(
                        f"No legal actions at level {level + 1}, prefix={prefixes[i]}"
                    )

                legal_s = torch.tensor(legal, dtype=torch.long, device=student_device)
                legal_t = torch.tensor(legal, dtype=torch.long, device=teacher_device)

                raw_s = s_last[i].index_select(0, legal_s).float()

                log_ps = F.log_softmax(
                    raw_s / args.distill_temperature,
                    dim=-1,
                )
                log_pt = F.log_softmax(
                    t_last[i].index_select(0, legal_t).float().to(student_device)
                    / args.distill_temperature,
                    dim=-1,
                )

                ps = log_ps.exp()
                pt = log_pt.exp()

                reverse_kl = (ps * (log_ps - log_pt)).sum()
                forward_kl = (pt * (log_pt - log_ps)).sum()

                kl_losses.append(reverse_kl)

                tag = f"L{level + 1}"
                step_stats[f"{tag}/reverse_kl"].append(reverse_kl.detach().item())
                step_stats[f"{tag}/forward_kl"].append(forward_kl.detach().item())
                step_stats[f"{tag}/student_entropy"].append(
                    entropy_from_logp(log_ps).detach().item()
                )
                step_stats[f"{tag}/teacher_entropy"].append(
                    entropy_from_logp(log_pt).detach().item()
                )
                step_stats[f"{tag}/legal_actions"].append(float(len(legal)))

                student_top1 = legal[int(torch.argmax(log_ps).item())]
                teacher_top1 = legal[int(torch.argmax(log_pt).item())]
                step_stats[f"{tag}/top1_agreement"].append(
                    float(student_top1 == teacher_top1)
                )

                if args.state_mode == "on_policy":
                    rollout_probs = F.softmax(
                        raw_s.detach() / args.rollout_temperature,
                        dim=-1,
                    )
                    sampled_pos = torch.multinomial(
                        rollout_probs,
                        num_samples=1,
                    ).item()
                    next_id = legal[sampled_pos]
                else:
                    gold = gold_ids[i]
                    expected_prefix = gold[:level]
                    if prefixes[i] != expected_prefix:
                        raise RuntimeError(
                            f"Gold-prefix mismatch at level={level + 1}: "
                            f"prefix={prefixes[i]}, expected={expected_prefix}"
                        )
                    next_id = gold[level]

                next_tokens.append(next_id)

                gold = gold_ids[i]
                if gold is not None and level < len(gold):
                    on_gold_state = prefixes[i] == gold[:level]
                    step_stats[f"{tag}/gold_state_rate"].append(float(on_gold_state))
                    if on_gold_state:
                        step_stats[f"{tag}/next_is_gold"].append(
                            float(next_id == gold[level])
                        )

            level_loss = torch.stack(kl_losses).mean()
            (level_loss / float(depth)).backward()

            for i, next_id in enumerate(next_tokens):
                prefixes[i].append(next_id)

            next_cpu = torch.tensor(
                next_tokens,
                dtype=ids_cpu.dtype,
            ).unsqueeze(1)

            ids_cpu = torch.cat([ids_cpu, next_cpu], dim=1)
            mask_cpu = torch.cat(
                [
                    mask_cpu,
                    torch.ones(
                        (mask_cpu.size(0), 1),
                        dtype=mask_cpu.dtype,
                    ),
                ],
                dim=1,
            )

        grad_norm = torch.nn.utils.clip_grad_norm_(
            student.parameters(),
            max_norm=1.0,
        )

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        exact = [
            float(prefix == gold)
            for prefix, gold in zip(prefixes, gold_ids)
            if gold is not None
        ]

        result = {
            "step": step,
            "state_mode": args.state_mode,
            "lr": scheduler.get_last_lr()[0],
            "grad_norm_pre_clip": float(grad_norm),
            "trajectory_exact_rate": mean(exact),
        }

        all_reverse = []
        for level in range(depth):
            tag = f"L{level + 1}"
            for metric in [
                "reverse_kl",
                "forward_kl",
                "student_entropy",
                "teacher_entropy",
                "legal_actions",
                "top1_agreement",
                "gold_state_rate",
                "next_is_gold",
            ]:
                key = f"{tag}/{metric}"
                if key in step_stats:
                    result[key] = mean(step_stats[key])

            if f"{tag}/reverse_kl" in result:
                all_reverse.append(result[f"{tag}/reverse_kl"])

        result["loss"] = mean(all_reverse)

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

        pbar.set_postfix(
            {
                "loss": f"{result['loss']:.3f}",
                "L1KL": f"{result.get('L1/reverse_kl', float('nan')):.2f}",
                "L2KL": f"{result.get('L2/reverse_kl', float('nan')):.2f}",
                "Hs1": f"{result.get('L1/student_entropy', float('nan')):.2f}",
                "exact": f"{result['trajectory_exact_rate']:.3f}",
            }
        )

        if step == 1 or step % args.log_every == 0 or step == args.max_steps:
            elapsed = time.time() - start_time
            print(
                f"step={step:4d} "
                f"mode={args.state_mode} "
                f"loss={result['loss']:.5f} "
                f"grad={result['grad_norm_pre_clip']:.4f} "
                f"exact={result['trajectory_exact_rate']:.4f} "
                f"time={elapsed:.1f}s"
            )

    print("\nTRAINING FINISHED")
    print("State mode :", args.state_mode)
    print("Total time :", time.time() - start_time)
    print("Log        :", log_path)

    if args.save_final:
        final_dir = out_dir / "final_checkpoint"
        final_dir.mkdir(parents=True, exist_ok=True)

        student.save_pretrained(final_dir, safe_serialization=True)
        student_tokenizer.save_pretrained(final_dir)

        print("Saved:", final_dir)


if __name__ == "__main__":
    main()
