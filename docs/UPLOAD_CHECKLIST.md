# Final GitHub Upload Checklist

From the repository root:

## 1. Pull required Apache-2.0 upstream files into the cleaned repository

```bash
bash scripts/finalize_repo.sh ../MiniOneRec
```

## 2. Confirm the repository contains no checkpoints/data

```bash
python scripts/check_repo.py
du -ah . | sort -h | tail -30
```

Do not commit model weights, Amazon data, Hugging Face caches, WandB runs, or raw logs.

## 3. Inspect the final result summary

```bash
python diagnostics/summarize_results.py
```

## 4. Initialize Git

```bash
git init
git add .
git status
git commit -m "Initial release: constrained policy distillation for MiniOneRec"
git branch -M main
```

## 5. Push

Create an empty GitHub repository, then:

```bash
git remote add origin git@github.com:<YOUR_USERNAME>/MiniOneRec-PolicyDistill.git
git push -u origin main
```
