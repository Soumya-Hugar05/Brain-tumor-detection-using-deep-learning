# Phase B - frozen-feature improvements (validation only)

## Shortcut probes (train -> val, no anatomy used)

| Probe | Val accuracy | Val macro-F1 | No-tumor AUC |
|---|---|---|---|
| geometry (original size/aspect) | 0.5163 | 0.4070 | 0.9731 |
| encoding (bytes/px, JPEG quality, colour mode) | 0.6989 | 0.6777 | 0.9836 |
| geometry + encoding | 0.7199 | 0.6998 | 0.9942 |
| crop geometry after crop+pad | 0.5880 | 0.5762 | 0.8827 |
| intensity statistics of crop+pad image | 0.8308 | 0.8264 | 0.9929 |

Chance accuracy (majority class): 0.258

## Default SVM per feature configuration (val)

| Config | Accuracy | Macro-F1 | Tumor sens | Tumor spec |
|---|---|---|---|---|
| resize | 0.9168 | 0.9139 | 0.9871 | 0.9926 |
| resize+flipavg | 0.9130 | 0.9096 | 0.9858 | 0.9926 |
| croppad | 0.9034 | 0.8999 | 0.9961 | 0.9926 |
| croppad+flipavg | 0.9054 | 0.9020 | 0.9923 | 0.9889 |

## Perturbation test (val)

| Condition | Model | n | Shortcut flip rate | Acc before | Acc after |
|---|---|---|---|---|---|
| tumor -> no-tumor-like (225x225, notumor JPEG quality) | resize | 776 | 0.0065 | 0.8905 | 0.8608 |
| tumor -> no-tumor-like (225x225, notumor JPEG quality) | resize+flipavg | 776 | 0.0026 | 0.8853 | 0.8531 |
| tumor -> no-tumor-like (225x225, notumor JPEG quality) | croppad | 776 | 0.0078 | 0.8724 | 0.8724 |
| tumor -> no-tumor-like (225x225, notumor JPEG quality) | croppad+flipavg | 776 | 0.0065 | 0.8763 | 0.8763 |
| tumor -> JPEG only (original size, notumor JPEG quality) | resize | 776 | 0.0000 | 0.8905 | 0.8905 |
| tumor -> JPEG only (original size, notumor JPEG quality) | resize+flipavg | 776 | 0.0000 | 0.8853 | 0.8853 |
| tumor -> JPEG only (original size, notumor JPEG quality) | croppad | 776 | 0.0026 | 0.8724 | 0.8673 |
| tumor -> JPEG only (original size, notumor JPEG quality) | croppad+flipavg | 776 | 0.0000 | 0.8763 | 0.8789 |
| no-tumor -> tumor-like (512x512, tumor JPEG quality) | resize | 270 | 0.0037 | 0.9926 | 0.9889 |
| no-tumor -> tumor-like (512x512, tumor JPEG quality) | resize+flipavg | 270 | 0.0000 | 0.9926 | 0.9926 |
| no-tumor -> tumor-like (512x512, tumor JPEG quality) | croppad | 270 | 0.0037 | 0.9926 | 0.9889 |
| no-tumor -> tumor-like (512x512, tumor JPEG quality) | croppad+flipavg | 270 | 0.0000 | 0.9889 | 0.9926 |
| no-tumor -> JPEG only (original size, tumor JPEG quality) | resize | 270 | 0.0037 | 0.9926 | 0.9889 |
| no-tumor -> JPEG only (original size, tumor JPEG quality) | resize+flipavg | 270 | 0.0000 | 0.9926 | 0.9926 |
| no-tumor -> JPEG only (original size, tumor JPEG quality) | croppad | 270 | 0.0000 | 0.9926 | 0.9926 |
| no-tumor -> JPEG only (original size, tumor JPEG quality) | croppad+flipavg | 270 | 0.0000 | 0.9889 | 0.9889 |

## Decisions

- **preprocessing**: selected `resize`. Rule: keep croppad if not significantly worse on VAL macro-F1 AND lower shortcut flip rate. dF1 -0.0139 (95% CI -0.0325..+0.0039)
- **flip_average**: selected `False`. Rule: keep only with a significant VAL macro-F1 gain. dF1 -0.0043 (95% CI -0.0142..+0.0041)
- **svm_hyperparameters**: selected `{'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'class_weight': 'balanced'}`. Rule: keep only with a significant VAL gain AND a higher grouped-CV macro-F1 on TRAIN. dF1 -0.0122 (95% CI -0.0234..-0.0008)
- **calibration**: selected `SVC built-in Platt (legacy)`. Rule: keep the method with the lowest VAL NLL (legacy unless improved).

## Selected model (resize)

Val accuracy 0.9216, macro-F1 0.9181, tumor-vs-no-tumor sensitivity 0.9845, specificity 0.9852; train accuracy 0.9934.
