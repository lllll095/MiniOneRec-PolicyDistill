# Results

All reported recommendation metrics are evaluated on the full Amazon **Industrial & Scientific** test split with **4,533 examples** using constrained beam search with beam size 50.

## Main Comparison

| Method | HR@10 | NDCG@10 | HR@50 |
|---|---:|---:|---:|
| 0.5B SFT | 0.1284 | 0.0863 | 0.2171 |
| 0.5B SFT -> GRPO | 0.1341 | 0.0924 | 0.2235 |
| 0.5B SFT -> Gold-KD | 0.1425 | 0.0998 | 0.2402 |
| **0.5B SFT -> OPD** | **0.1441** | **0.1002** | **0.2409** |
| 0.5B OPD -> GRPO | 0.1432 | 0.0968 | 0.2349 |
| 1.5B Teacher | 0.1577 | 0.1175 | 0.2447 |

See `main_results.csv` for HR/NDCG at K = 1, 3, 5, 10, 20, 50.

See `teacher_student_diagnostics.csv` for the teacher/student legal-action diagnostics.
