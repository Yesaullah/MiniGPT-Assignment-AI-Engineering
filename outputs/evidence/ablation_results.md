| configuration | params | seeds | best val BPC (mean ± std) | difference vs baseline (paired) | ms/step | train tokens/s | peak MB | train min |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| baseline (L4 h4 T128) | 3,192,832 | 0,1,2 | 1.9985 ± 0.0009 | - | 74.9 | 109,456 | 794 | 6.8 |
| 2 layers | 1,613,312 | 0,1,2 | 2.0621 ± 0.0026 | +0.0636 ± 0.0032 | 38.3 | 213,634 | 439 | 3.5 |
| 6 layers | 4,772,352 | 0,1,2 | 2.0063 ± 0.0029 | +0.0078 ± 0.0022 | 112.7 | 72,719 | 1148 | 10.2 |
| 2 heads | 3,192,832 | 0,1,2 | 1.9928 ± 0.0017 | -0.0057 ± 0.0010 | 72.0 | 113,701 | 722 | 6.6 |
| 8 heads | 3,192,832 | 0,1,2 | 2.0095 ± 0.0010 | +0.0110 ± 0.0019 | 83.7 | 97,929 | 938 | 7.6 |
| context 64 | 3,192,832 | 0,1,2 | 2.0291 ± 0.0019 | +0.0305 ± 0.0015 | 70.8 | 115,765 | 722 | 6.5 |
| context 256 | 3,192,832 | 0,1,2 | 1.9830 ± 0.0031 | -0.0156 ± 0.0023 | 86.1 | 95,139 | 938 | 7.7 |

Per seed:

| configuration | seed | best val loss | best val BPC | best step |
|---|---:|---:|---:|---:|
| baseline (L4 h4 T128) | 0 | 1.3853 | 1.9985 | 5000 |
| baseline (L4 h4 T128) | 1 | 1.3846 | 1.9976 | 5000 |
| baseline (L4 h4 T128) | 2 | 1.3859 | 1.9994 | 5000 |
| 2 layers | 0 | 1.4277 | 2.0597 | 5000 |
| 2 layers | 1 | 1.4312 | 2.0648 | 5000 |
| 2 layers | 2 | 1.4292 | 2.0619 | 5000 |
| 6 layers | 0 | 1.3894 | 2.0045 | 4250 |
| 6 layers | 1 | 1.3896 | 2.0048 | 3500 |
| 6 layers | 2 | 1.3930 | 2.0096 | 3750 |
| 2 heads | 0 | 1.3807 | 1.9919 | 5000 |
| 2 heads | 1 | 1.3806 | 1.9918 | 5000 |
| 2 heads | 2 | 1.3827 | 1.9948 | 5000 |
| 8 heads | 0 | 1.3931 | 2.0098 | 5000 |
| 8 heads | 1 | 1.3935 | 2.0103 | 4500 |
| 8 heads | 2 | 1.3921 | 2.0084 | 5000 |
| context 64 | 0 | 1.4076 | 2.0308 | 5000 |
| context 64 | 1 | 1.4051 | 2.0271 | 4750 |
| context 64 | 2 | 1.4066 | 2.0293 | 4500 |
| context 256 | 0 | 1.3755 | 1.9845 | 4750 |
| context 256 | 1 | 1.3720 | 1.9794 | 4750 |
| context 256 | 2 | 1.3759 | 1.9850 | 4750 |

Seeds 0, 1 and 2 per configuration (fixed in advance, D26); std is the sample standard deviation over seeds. The paired difference compares each variant with the baseline at the same seed. Three seeds give a rough, not precise, estimate of seed-to-seed variation. LR 1e-3 and dropout 0.1 are shared by all runs. ms/step and tokens/s count training steps only; train min is wall clock including evaluation; peak MB is the largest GPU peak over the seeds.
