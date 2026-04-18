# 03 EPI robustness note

Across the currently confirmed standard-evaluation cell types (nCD4, tB, tCD4, tCD8),
the mean AUPRC was 0.585 ± 0.046.

For the same set of cell types under balanced evaluation,
the mean AUPRC increased to 0.833 ± 0.024,
representing an absolute gain of 0.249.

Across all six cell types in balanced evaluation,
the mean AUPRC was 0.824 ± 0.027.

Interpretation:
- The standard protocol yields moderate but reproducible performance.
- Balanced evaluation produces consistently higher AUPRC across cell types.
- This suggests that class imbalance / decision threshold strongly affects observed performance in 03_LOGO_EPI.
- Therefore, 03 should be written as an auxiliary result line supporting transferability and cross-cell-type applicability, rather than as the strongest performance result in the thesis.
