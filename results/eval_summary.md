# Evaluation results

Score: mean ROUGE-L F1 against the reference answer (0 to 1). 'close' = share of answers with ROUGE-L >= 0.5.

| System | held-out (n=6) | reworded (seen) (n=16) | reworded (unseen) (n=4) |
| --- | --- | --- | --- |
| Retrieval baseline (no model) | 0.13 (0% close) | 0.31 (25% close) | 0.14 (0% close) |
| GPT-2 base (not fine-tuned) | 0.15 (0% close) | 0.09 (0% close) | 0.13 (0% close) |
| GPT-2 fine-tuned | 0.18 (0% close) | 0.21 (6% close) | 0.26 (0% close) |
