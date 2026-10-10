| configuration | non-overlapping BPC | stride-32 BPC | difference vs baseline, non-overlapping | difference vs baseline, stride 32 |
|---|---:|---:|---:|---:|
| baseline (L4 h4 T128) | 1.9985 ± 0.0009 | 1.9626 ± 0.0019 | - | - |
| 2 layers | 2.0621 ± 0.0026 | 2.0307 ± 0.0034 | +0.0636 ± 0.0032 | +0.0681 ± 0.0051 |
| 6 layers | 2.0063 ± 0.0029 | 1.9695 ± 0.0010 | +0.0078 ± 0.0022 | +0.0069 ± 0.0026 |
| 2 heads | 1.9928 ± 0.0017 | 1.9568 ± 0.0027 | -0.0057 ± 0.0010 | -0.0058 ± 0.0030 |
| 8 heads | 2.0095 ± 0.0010 | 1.9744 ± 0.0028 | +0.0110 ± 0.0019 | +0.0118 ± 0.0010 |
| context 64 | 2.0291 ± 0.0019 | 1.9598 ± 0.0019 | +0.0305 ± 0.0015 | -0.0028 ± 0.0028 |
| context 256 | 1.9830 ± 0.0031 | 1.9573 ± 0.0051 | -0.0156 ± 0.0023 | -0.0053 ± 0.0038 |

Validation split, best.pt of each run (selected under the non-overlapping protocol). Mean ± sample std over seeds; differences are paired with the baseline at the same seed. Stride 32: every scored character after the first window sees at least T - 31 characters of context.
