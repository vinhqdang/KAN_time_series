| scenario | method | AUROC mean+-sd | AUPRC mean+-sd |
|---|---|---|---|
| I1 | SPADE-levels | 0.708 +- 0.076 | 0.483 +- 0.118 |
| I1 | SPADE-diff | 1.000 +- 0.000 | 1.000 +- 0.000 |
| I1 | ECM-SPADE | 1.000 +- 0.000 | 1.000 +- 0.000 |
| I1 | ECM-SPADE(no-placebo) | 1.000 +- 0.000 | 1.000 +- 0.000 |
| I1 | VARLasso-levels | 0.990 +- 0.026 | 0.988 +- 0.024 |
| I1 | PCMCI-levels | 0.972 +- 0.032 | 0.949 +- 0.043 |
| I1 | VARLasso-diff | 0.991 +- 0.027 | 0.991 +- 0.027 |
| I1 | PCMCI-diff | 0.982 +- 0.054 | 0.987 +- 0.039 |
| I1C | SPADE-levels | 0.689 +- 0.072 | 0.601 +- 0.089 |
| I1C | SPADE-diff | 0.964 +- 0.022 | 0.946 +- 0.038 |
| I1C | ECM-SPADE | 0.982 +- 0.023 | 0.968 +- 0.045 |
| I1C | ECM-SPADE(no-placebo) | 0.977 +- 0.026 | 0.962 +- 0.050 |
| I1C | VARLasso-levels | 0.989 +- 0.026 | 0.986 +- 0.032 |
| I1C | PCMCI-levels | 0.949 +- 0.032 | 0.914 +- 0.029 |
| I1C | VARLasso-diff | 0.988 +- 0.026 | 0.988 +- 0.022 |
| I1C | PCMCI-diff | 0.989 +- 0.021 | 0.983 +- 0.025 |
| ST | SPADE-levels | 1.000 +- 0.000 | 1.000 +- 0.000 |
| ST | SPADE-diff | 0.988 +- 0.030 | 0.990 +- 0.020 |
| ST | ECM-SPADE | 1.000 +- 0.000 | 1.000 +- 0.000 |
| ST | ECM-SPADE(no-placebo) | 1.000 +- 0.000 | 1.000 +- 0.000 |
| ST | VARLasso-levels | 0.992 +- 0.024 | 0.991 +- 0.027 |
| ST | PCMCI-levels | 0.988 +- 0.021 | 0.986 +- 0.021 |
| ST | VARLasso-diff | 0.988 +- 0.022 | 0.986 +- 0.020 |
| ST | PCMCI-diff | 0.989 +- 0.023 | 0.989 +- 0.022 |

Paired AUPRC differences (mean +- sd over 10 seeds; mean/(sd/sqrt10) in brackets)
I1: SPADE-diff - SPADE-levels: +0.517 +- 0.118 [t=13.1]
I1: ECM-SPADE - SPADE-diff: +0.000 +- 0.000 [t=0.0]
I1: ECM-SPADE - ECM-SPADE(no-placebo): +0.000 +- 0.000 [t=0.0]
I1: ECM-SPADE - PCMCI-diff: +0.013 +- 0.039 [t=1.0]
I1: ECM-SPADE - VARLasso-diff: +0.009 +- 0.027 [t=1.0]
I1C: SPADE-diff - SPADE-levels: +0.345 +- 0.108 [t=9.6]
I1C: ECM-SPADE - SPADE-diff: +0.022 +- 0.026 [t=2.6]
I1C: ECM-SPADE - ECM-SPADE(no-placebo): +0.006 +- 0.007 [t=2.4]
I1C: ECM-SPADE - PCMCI-diff: -0.014 +- 0.053 [t=-0.8]
I1C: ECM-SPADE - VARLasso-diff: -0.020 +- 0.049 [t=-1.2]
ST: SPADE-diff - SPADE-levels: -0.010 +- 0.020 [t=-1.5]
ST: ECM-SPADE - SPADE-diff: +0.010 +- 0.020 [t=1.5]
ST: ECM-SPADE - ECM-SPADE(no-placebo): +0.000 +- 0.000 [t=0.0]
ST: ECM-SPADE - PCMCI-diff: +0.011 +- 0.022 [t=1.5]
ST: ECM-SPADE - VARLasso-diff: +0.014 +- 0.020 [t=2.1]