# Stage 0 results (test seeds 5005..5029)

Test subgraphs: 100 (25 per size and region). Primary analysis: standardised data, lag order p=1, label rule tau_e < Delta/2. Bootstrap: 2000 resamples of subgraphs, percentile 95% CI. Orientation accuracy is on true contemporaneous edges given the true skeleton (chance 0.5); VAR and correlation are 0.5 by definition there (coefficient-based orientation in the secondary table).

## Primary table

| res | method | directed AUROC [95% CI] | AUPRC [95% CI] | orientation acc, contemporaneous [95% CI] | SE | edges / subgraphs |
|---|---|---|---|---|---|---|
| 6h | var | 0.598 [0.575, 0.621] | 0.329 [0.297, 0.365] | 0.500 [0.500, 0.500] | 0.000 | 276 / 80 |
| 6h | corr | 0.686 [0.662, 0.711] | 0.353 [0.335, 0.372] | 0.500 [0.500, 0.500] | 0.000 | 276 / 80 |
| 6h | pcmci+ | 0.704 [0.678, 0.731] | 0.395 [0.369, 0.422] | 0.558 [0.515, 0.602] | 0.022 | 276 / 80 |
| 6h | varlingam | 0.615 [0.586, 0.641] | 0.334 [0.301, 0.368] | 0.547 [0.486, 0.616] | 0.033 | 276 / 80 |
| 6h | dynotears | 0.711 [0.679, 0.741] | 0.438 [0.399, 0.479] | 0.563 [0.507, 0.620] | 0.029 | 276 / 80 |
| 6h | notears_svar2 | 0.599 [0.574, 0.623] | 0.321 [0.290, 0.356] | 0.453 [0.387, 0.524] | 0.035 | 276 / 80 |
| 6h | var_sortnregress | 0.519 [0.493, 0.544] | 0.252 [0.227, 0.276] | 0.533 [0.475, 0.587] | 0.029 | 276 / 80 |
| 6h | r2_sortnregress | 0.649 [0.623, 0.675] | 0.316 [0.285, 0.348] | 0.739 [0.696, 0.782] | 0.022 | 276 / 80 |
| 6h | random | 0.487 [0.458, 0.516] | 0.230 [0.207, 0.257] | 0.540 [0.480, 0.600] | 0.031 | 276 / 80 |
| 12h | var | 0.568 [0.547, 0.588] | 0.288 [0.260, 0.318] | 0.500 [0.500, 0.500] | 0.000 | 424 / 93 |
| 12h | corr | 0.687 [0.661, 0.710] | 0.347 [0.329, 0.365] | 0.500 [0.500, 0.500] | 0.000 | 424 / 93 |
| 12h | pcmci+ | 0.704 [0.674, 0.733] | 0.389 [0.361, 0.419] | 0.567 [0.529, 0.605] | 0.019 | 424 / 93 |
| 12h | varlingam | 0.634 [0.603, 0.663] | 0.334 [0.300, 0.372] | 0.611 [0.565, 0.660] | 0.025 | 424 / 93 |
| 12h | dynotears | 0.690 [0.659, 0.718] | 0.369 [0.336, 0.405] | 0.495 [0.446, 0.546] | 0.025 | 424 / 93 |
| 12h | notears_svar2 | 0.571 [0.547, 0.592] | 0.289 [0.260, 0.319] | 0.455 [0.401, 0.512] | 0.029 | 424 / 93 |
| 12h | var_sortnregress | 0.501 [0.474, 0.527] | 0.227 [0.203, 0.257] | 0.500 [0.459, 0.542] | 0.021 | 424 / 93 |
| 12h | r2_sortnregress | 0.655 [0.629, 0.681] | 0.324 [0.295, 0.356] | 0.738 [0.705, 0.771] | 0.017 | 424 / 93 |
| 12h | random | 0.515 [0.489, 0.542] | 0.238 [0.210, 0.267] | 0.585 [0.542, 0.626] | 0.022 | 424 / 93 |
| 24h | var | 0.531 [0.509, 0.552] | 0.266 [0.240, 0.294] | 0.500 [0.500, 0.500] | 0.000 | 533 / 99 |
| 24h | corr | 0.684 [0.658, 0.707] | 0.343 [0.325, 0.360] | 0.500 [0.500, 0.500] | 0.000 | 533 / 99 |
| 24h | pcmci+ | 0.689 [0.661, 0.714] | 0.354 [0.330, 0.378] | 0.532 [0.498, 0.565] | 0.017 | 533 / 99 |
| 24h | varlingam | 0.614 [0.582, 0.645] | 0.317 [0.285, 0.353] | 0.598 [0.546, 0.646] | 0.025 | 533 / 99 |
| 24h | dynotears | 0.648 [0.617, 0.678] | 0.307 [0.279, 0.334] | 0.462 [0.423, 0.500] | 0.020 | 533 / 99 |
| 24h | notears_svar2 | 0.541 [0.515, 0.566] | 0.266 [0.240, 0.295] | 0.417 [0.375, 0.457] | 0.020 | 533 / 99 |
| 24h | var_sortnregress | 0.549 [0.521, 0.579] | 0.275 [0.243, 0.307] | 0.561 [0.523, 0.599] | 0.019 | 533 / 99 |
| 24h | r2_sortnregress | 0.654 [0.628, 0.679] | 0.320 [0.290, 0.352] | 0.720 [0.691, 0.751] | 0.016 | 533 / 99 |
| 24h | random | 0.502 [0.475, 0.528] | 0.233 [0.209, 0.259] | 0.497 [0.455, 0.539] | 0.021 | 533 / 99 |

## Paired comparison of directed AUROC against lag-only VAR (paired Wilcoxon, two-sided; Holm over all pairs below)

| res | method | mean AUROC diff vs VAR | raw p | Holm p (3 capable methods x 3 res = 9) | Holm p (incl. reference, 12) |
|---|---|---|---|---|---|
| 6h | pcmci+ | +0.106 | 1.2e-09 | 5.9e-09 | 9.5e-09 |
| 6h | varlingam | +0.016 | 0.22 | 0.22 | 0.65 |
| 6h | dynotears | +0.112 | 4.5e-10 | 2.7e-09 | 4e-09 |
| 6h | notears_svar2 | +0.001 | 0.88 |  | 0.88 |
| 12h | pcmci+ | +0.136 | 1.4e-10 | 9.8e-10 | 1.4e-09 |
| 12h | varlingam | +0.066 | 3.9e-06 | 7.7e-06 | 1.9e-05 |
| 12h | dynotears | +0.123 | 7.7e-11 | 6.2e-10 | 8.5e-10 |
| 12h | notears_svar2 | +0.003 | 0.4 |  | 0.81 |
| 24h | pcmci+ | +0.158 | 1.6e-13 | 1.5e-12 | 1.9e-12 |
| 24h | varlingam | +0.083 | 1.7e-06 | 5e-06 | 1e-05 |
| 24h | dynotears | +0.117 | 1.3e-09 | 5.9e-09 | 9.5e-09 |
| 24h | notears_svar2 | +0.010 | 0.13 |  | 0.51 |

## Hypothesis verdicts - verdict methods (PCMCI+, VAR-LiNGAM, DYNOTEARS)

- H1 condition at 12h: pcmci+=not met, varlingam=met, dynotears=not met
- H1 condition at 24h: pcmci+=not met, varlingam=not met, dynotears=not met
- **H1 (needs 12 h and 24 h): does not hold**
- Negative (i), accuracy within 2 SE of 0.5 for every capable method at 24 h (>=30 subgraphs with contemporaneous edges: True): **not met**
- Negative (ii), VAR directed AUROC >= every capable method at 24 h: **not met**
  - (ii) at 12h: not met
  - (ii) at 6h: not met
- **Overall: AMBIGUOUS**

## Hypothesis verdicts - sensitivity: reference NOTEARS-SVAR included

- H1 condition at 12h: pcmci+=not met, varlingam=met, dynotears=not met, notears_svar2=not met
- H1 condition at 24h: pcmci+=not met, varlingam=not met, dynotears=not met, notears_svar2=not met
- **H1 (needs 12 h and 24 h): does not hold**
- Negative (i), accuracy within 2 SE of 0.5 for every capable method at 24 h (>=30 subgraphs with contemporaneous edges: True): **not met**
- Negative (ii), VAR directed AUROC >= every capable method at 24 h: **not met**
  - (ii) at 12h: not met
  - (ii) at 6h: not met
- **Overall: AMBIGUOUS**

## Secondary: raw orientation by scores (VAR and correlation not forced to 0.5) and orientation on lagged edges

| res | method | accuracy (coefficient/score-based, no by-definition 0.5) [95% CI] | accuracy on lagged edges [95% CI] |
|---|---|---|---|
| 6h | var | 0.507 [0.446, 0.570] | 0.564 [0.511, 0.618] |
| 6h | corr | 0.507 [0.478, 0.537] | 0.550 [0.524, 0.577] |
| 6h | pcmci+ | 0.558 [0.518, 0.601] | 0.680 [0.636, 0.724] |
| 6h | varlingam | 0.547 [0.481, 0.615] | 0.523 [0.472, 0.577] |
| 6h | dynotears | 0.563 [0.505, 0.621] | 0.724 [0.690, 0.760] |
| 6h | notears_svar2 | 0.453 [0.385, 0.521] | 0.567 [0.512, 0.620] |
| 6h | var_sortnregress | 0.533 [0.476, 0.587] | 0.492 [0.448, 0.539] |
| 6h | r2_sortnregress | 0.739 [0.693, 0.780] | 0.714 [0.666, 0.759] |
| 6h | random | 0.540 [0.482, 0.601] | 0.503 [0.449, 0.554] |
| 12h | var | 0.460 [0.410, 0.514] | 0.517 [0.452, 0.586] |
| 12h | corr | 0.512 [0.489, 0.533] | 0.510 [0.483, 0.537] |
| 12h | pcmci+ | 0.567 [0.531, 0.602] | 0.683 [0.628, 0.736] |
| 12h | varlingam | 0.611 [0.562, 0.660] | 0.615 [0.552, 0.677] |
| 12h | dynotears | 0.495 [0.446, 0.548] | 0.713 [0.670, 0.757] |
| 12h | notears_svar2 | 0.455 [0.402, 0.512] | 0.542 [0.483, 0.602] |
| 12h | var_sortnregress | 0.500 [0.455, 0.541] | 0.483 [0.415, 0.549] |
| 12h | r2_sortnregress | 0.738 [0.705, 0.773] | 0.721 [0.653, 0.782] |
| 12h | random | 0.585 [0.542, 0.626] | 0.450 [0.398, 0.502] |
| 24h | var | 0.456 [0.412, 0.500] | 0.435 [0.352, 0.521] |
| 24h | corr | 0.498 [0.480, 0.516] | 0.515 [0.467, 0.559] |
| 24h | pcmci+ | 0.532 [0.499, 0.564] | 0.622 [0.546, 0.699] |
| 24h | varlingam | 0.598 [0.549, 0.647] | 0.519 [0.435, 0.601] |
| 24h | dynotears | 0.462 [0.423, 0.503] | 0.634 [0.578, 0.690] |
| 24h | notears_svar2 | 0.417 [0.375, 0.458] | 0.420 [0.336, 0.500] |
| 24h | var_sortnregress | 0.561 [0.522, 0.601] | 0.489 [0.391, 0.589] |
| 24h | r2_sortnregress | 0.720 [0.690, 0.750] | 0.771 [0.684, 0.850] |
| 24h | random | 0.497 [0.453, 0.539] | 0.511 [0.432, 0.593] |

## Sensitivity to the label rule

| res | method | acc Delta/4 rule (n) | acc Delta/2 primary (n) | acc Delta rule (n) |
|---|---|---|---|---|
| 6h | pcmci+ | 0.534 [0.486, 0.582] (205) | 0.558 [0.516, 0.601] (276) | 0.593 [0.560, 0.627] (424) |
| 6h | varlingam | 0.541 [0.465, 0.627] (205) | 0.547 [0.485, 0.617] (276) | 0.547 [0.497, 0.600] (424) |
| 6h | dynotears | 0.495 [0.433, 0.556] (205) | 0.563 [0.504, 0.620] (276) | 0.630 [0.583, 0.675] (424) |
| 6h | notears_svar2 | 0.410 [0.343, 0.483] (205) | 0.453 [0.387, 0.520] (276) | 0.474 [0.419, 0.532] (424) |
| 12h | pcmci+ | 0.529 [0.480, 0.578] (276) | 0.567 [0.532, 0.606] (424) | 0.610 [0.571, 0.644] (533) |
| 12h | varlingam | 0.620 [0.564, 0.677] (276) | 0.611 [0.564, 0.659] (424) | 0.618 [0.573, 0.662] (533) |
| 12h | dynotears | 0.435 [0.384, 0.493] (276) | 0.495 [0.443, 0.547] (424) | 0.556 [0.511, 0.601] (533) |
| 12h | notears_svar2 | 0.467 [0.399, 0.534] (276) | 0.455 [0.401, 0.513] (424) | 0.482 [0.435, 0.530] (533) |
| 24h | pcmci+ | 0.514 [0.477, 0.549] (424) | 0.532 [0.499, 0.564] (533) | 0.553 [0.518, 0.587] (580) |
| 24h | varlingam | 0.575 [0.522, 0.629] (424) | 0.598 [0.547, 0.645] (533) | 0.589 [0.545, 0.635] (580) |
| 24h | dynotears | 0.426 [0.382, 0.469] (424) | 0.462 [0.422, 0.504] (533) | 0.484 [0.446, 0.520] (580) |
| 24h | notears_svar2 | 0.403 [0.359, 0.451] (424) | 0.417 [0.375, 0.459] (533) | 0.410 [0.370, 0.449] (580) |

## Sensitivity to lag order (p=2, same hyperparameters)

| res | method | AUROC p=1 | AUROC p=2 | acc p=1 | acc p=2 |
|---|---|---|---|---|---|
| 6h | var | 0.598 | 0.625 | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| 6h | pcmci+ | 0.704 | 0.690 | 0.558 [0.514, 0.599] | 0.556 [0.515, 0.598] |
| 6h | varlingam | 0.615 | 0.615 | 0.547 [0.483, 0.621] | 0.482 [0.423, 0.543] |
| 6h | dynotears | 0.711 | 0.708 | 0.563 [0.505, 0.623] | 0.563 [0.506, 0.622] |
| 6h | notears_svar2 | 0.599 | 0.624 | 0.453 [0.388, 0.523] | 0.424 [0.369, 0.489] |
| 12h | var | 0.568 | 0.578 | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| 12h | pcmci+ | 0.704 | 0.718 | 0.567 [0.533, 0.606] | 0.603 [0.569, 0.635] |
| 12h | varlingam | 0.634 | 0.638 | 0.611 [0.564, 0.662] | 0.583 [0.534, 0.635] |
| 12h | dynotears | 0.690 | 0.687 | 0.495 [0.445, 0.545] | 0.492 [0.441, 0.543] |
| 12h | notears_svar2 | 0.571 | 0.586 | 0.455 [0.400, 0.510] | 0.432 [0.374, 0.490] |
| 24h | var | 0.531 | 0.530 | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| 24h | pcmci+ | 0.689 | 0.694 | 0.532 [0.498, 0.565] | 0.553 [0.516, 0.591] |
| 24h | varlingam | 0.614 | 0.594 | 0.598 [0.550, 0.647] | 0.504 [0.460, 0.547] |
| 24h | dynotears | 0.648 | 0.648 | 0.462 [0.423, 0.503] | 0.462 [0.425, 0.501] |
| 24h | notears_svar2 | 0.541 | 0.549 | 0.417 [0.376, 0.458] | 0.435 [0.398, 0.474] |

## Secondary analysis: raw scale (scale leakage)

| res | method | AUROC raw | acc contemp raw [95% CI] | AUROC std | acc contemp std |
|---|---|---|---|---|---|
| 6h | var | 0.761 | 0.500 [0.500, 0.500] | 0.598 | 0.500 |
| 6h | corr | 0.684 | 0.500 [0.500, 0.500] | 0.686 | 0.500 |
| 6h | pcmci+ | 0.704 | 0.558 [0.517, 0.600] | 0.704 | 0.558 |
| 6h | varlingam | 0.764 | 0.757 [0.701, 0.809] | 0.615 | 0.547 |
| 6h | dynotears | 0.726 | 0.683 [0.626, 0.737] | 0.711 | 0.563 |
| 6h | notears_svar2 | 0.815 | 0.859 [0.818, 0.893] | 0.599 | 0.453 |
| 6h | var_sortnregress | 0.808 | 0.920 [0.883, 0.955] | 0.519 | 0.533 |
| 6h | r2_sortnregress | 0.676 | 0.739 [0.693, 0.781] | 0.649 | 0.739 |
| 12h | var | 0.747 | 0.500 [0.500, 0.500] | 0.568 | 0.500 |
| 12h | corr | 0.687 | 0.500 [0.500, 0.500] | 0.687 | 0.500 |
| 12h | pcmci+ | 0.704 | 0.567 [0.532, 0.603] | 0.704 | 0.567 |
| 12h | varlingam | 0.713 | 0.755 [0.704, 0.803] | 0.634 | 0.611 |
| 12h | dynotears | 0.740 | 0.750 [0.699, 0.797] | 0.690 | 0.495 |
| 12h | notears_svar2 | 0.810 | 0.877 [0.843, 0.911] | 0.571 | 0.455 |
| 12h | var_sortnregress | 0.806 | 0.941 [0.910, 0.967] | 0.501 | 0.500 |
| 12h | r2_sortnregress | 0.679 | 0.738 [0.704, 0.768] | 0.655 | 0.738 |
| 24h | var | 0.723 | 0.500 [0.500, 0.500] | 0.531 | 0.500 |
| 24h | corr | 0.684 | 0.500 [0.500, 0.500] | 0.684 | 0.500 |
| 24h | pcmci+ | 0.689 | 0.532 [0.502, 0.566] | 0.689 | 0.532 |
| 24h | varlingam | 0.686 | 0.740 [0.697, 0.787] | 0.614 | 0.598 |
| 24h | dynotears | 0.741 | 0.765 [0.717, 0.809] | 0.648 | 0.462 |
| 24h | notears_svar2 | 0.805 | 0.865 [0.831, 0.900] | 0.541 | 0.417 |
| 24h | var_sortnregress | 0.802 | 0.938 [0.912, 0.960] | 0.549 | 0.561 |
| 24h | r2_sortnregress | 0.679 | 0.720 [0.689, 0.750] | 0.654 | 0.720 |

## Breakdown by region and size (primary analysis)

| res | region | d | method | AUROC | acc contemp (n) |
|---|---|---|---|---|---|
| 6h | east_germany | 5 | var | 0.624 | 0.500 (50) |
| 6h | east_germany | 5 | pcmci+ | 0.652 | 0.500 (50) |
| 6h | east_germany | 5 | varlingam | 0.619 | 0.560 (50) |
| 6h | east_germany | 5 | dynotears | 0.673 | 0.540 (50) |
| 6h | east_germany | 10 | var | 0.573 | 0.500 (120) |
| 6h | east_germany | 10 | pcmci+ | 0.743 | 0.596 (120) |
| 6h | east_germany | 10 | varlingam | 0.633 | 0.467 (120) |
| 6h | east_germany | 10 | dynotears | 0.710 | 0.529 (120) |
| 6h | bavaria | 5 | var | 0.600 | 0.500 (31) |
| 6h | bavaria | 5 | pcmci+ | 0.718 | 0.548 (31) |
| 6h | bavaria | 5 | varlingam | 0.578 | 0.516 (31) |
| 6h | bavaria | 5 | dynotears | 0.719 | 0.565 (31) |
| 6h | bavaria | 10 | var | 0.595 | 0.500 (75) |
| 6h | bavaria | 10 | pcmci+ | 0.705 | 0.540 (75) |
| 6h | bavaria | 10 | varlingam | 0.628 | 0.680 (75) |
| 6h | bavaria | 10 | dynotears | 0.740 | 0.633 (75) |
| 12h | east_germany | 5 | var | 0.604 | 0.500 (58) |
| 12h | east_germany | 5 | pcmci+ | 0.656 | 0.586 (58) |
| 12h | east_germany | 5 | varlingam | 0.657 | 0.681 (58) |
| 12h | east_germany | 5 | dynotears | 0.653 | 0.440 (58) |
| 12h | east_germany | 10 | var | 0.556 | 0.500 (152) |
| 12h | east_germany | 10 | pcmci+ | 0.754 | 0.572 (152) |
| 12h | east_germany | 10 | varlingam | 0.693 | 0.668 (152) |
| 12h | east_germany | 10 | dynotears | 0.713 | 0.513 (152) |
| 12h | bavaria | 5 | var | 0.565 | 0.500 (64) |
| 12h | bavaria | 5 | pcmci+ | 0.685 | 0.555 (64) |
| 12h | bavaria | 5 | varlingam | 0.563 | 0.531 (64) |
| 12h | bavaria | 5 | dynotears | 0.690 | 0.500 (64) |
| 12h | bavaria | 10 | var | 0.547 | 0.500 (150) |
| 12h | bavaria | 10 | pcmci+ | 0.720 | 0.560 (150) |
| 12h | bavaria | 10 | varlingam | 0.623 | 0.560 (150) |
| 12h | bavaria | 10 | dynotears | 0.706 | 0.497 (150) |
| 24h | east_germany | 5 | var | 0.546 | 0.500 (70) |
| 24h | east_germany | 5 | pcmci+ | 0.633 | 0.521 (70) |
| 24h | east_germany | 5 | varlingam | 0.615 | 0.671 (70) |
| 24h | east_germany | 5 | dynotears | 0.573 | 0.386 (70) |
| 24h | east_germany | 10 | var | 0.536 | 0.500 (191) |
| 24h | east_germany | 10 | pcmci+ | 0.726 | 0.537 (191) |
| 24h | east_germany | 10 | varlingam | 0.663 | 0.605 (191) |
| 24h | east_germany | 10 | dynotears | 0.696 | 0.521 (191) |
| 24h | bavaria | 5 | var | 0.521 | 0.500 (84) |
| 24h | bavaria | 5 | pcmci+ | 0.684 | 0.565 (84) |
| 24h | bavaria | 5 | varlingam | 0.551 | 0.524 (84) |
| 24h | bavaria | 5 | dynotears | 0.644 | 0.429 (84) |
| 24h | bavaria | 10 | var | 0.521 | 0.500 (188) |
| 24h | bavaria | 10 | pcmci+ | 0.712 | 0.516 (188) |
| 24h | bavaria | 10 | varlingam | 0.626 | 0.596 (188) |
| 24h | bavaria | 10 | dynotears | 0.680 | 0.447 (188) |

Noise floor reminder: random-score AUROC and orientation are shown in the primary table.
