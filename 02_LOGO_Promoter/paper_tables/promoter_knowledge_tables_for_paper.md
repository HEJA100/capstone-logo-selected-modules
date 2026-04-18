# Promoter knowledge ablation tables for paper

## Table X. Comparison of promoter prediction performance under different knowledge settings

| Task | Model | F1 |
|---|---|---:|
| BOTH | Sequence-only | 0.9128 |
| BOTH | Original knowledge-enabled LOGO | 0.9547 |
| BOTH | Structural knowledge | **0.9515 ± 0.0030** |
| BOTH | Regulatory knowledge | 0.8918 ± 0.0117 |
| BOTH | Shuffled knowledge | 0.8869 ± 0.0141 |
| TATA_BOX | Sequence-only | 0.8930 |
| TATA_BOX | Original knowledge-enabled LOGO | 0.9387 |
| TATA_BOX | Structural knowledge | 0.9314 ± 0.0092 |
| TATA_BOX | Regulatory knowledge | **0.9336 ± 0.0084** |
| TATA_BOX | Shuffled knowledge | 0.9221 ± 0.0127 |
| NO_TATA_BOX | Sequence-only | 0.9090 |
| NO_TATA_BOX | Original knowledge-enabled LOGO | 0.9529 |
| NO_TATA_BOX | Structural knowledge | **0.9532 ± 0.0024** |
| NO_TATA_BOX | Regulatory knowledge | 0.8885 ± 0.0124 |
| NO_TATA_BOX | Shuffled knowledge | 0.8899 ± 0.0077 |

**Note.** Sequence-only and original knowledge-enabled LOGO are reported as summary values from the main comparison table. Structural, regulatory, and shuffled knowledge results are reported as mean ± SD across 10 folds.

---

## Table Y. F1 improvement over sequence-only and shuffled controls

| Task | Structural vs Sequence-only | Structural vs Shuffled | Regulatory vs Shuffled |
|---|---:|---:|---:|
| BOTH | +0.0387 | +0.0646 | +0.0050 |
| TATA_BOX | +0.0384 | +0.0093 | +0.0115 |
| NO_TATA_BOX | +0.0442 | +0.0633 | -0.0013 |

**Interpretation.** Structural knowledge consistently outperformed shuffled knowledge across all three promoter prediction tasks and retained most of the performance gain of the original knowledge-enabled LOGO. This supports the claim that promoter prediction benefits from biologically organized knowledge injection rather than from arbitrary annotation signals alone. Regulatory knowledge showed weaker or less stable gains, especially for BOTH and NO_TATA_BOX.
