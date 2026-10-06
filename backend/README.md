---
title: Alzheimers MRI Classifier
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Alzheimer's Disease MRI Classifier

FastAPI backend serving three fine-tuned ResNet-18 models that classify cognitive status from axial T1-weighted MRI slices of the OASIS-1 dataset. It powers the web demo at https://alzheimers-mri-classifier.netlify.app.

This is a research prototype. It is not validated for clinical diagnosis or patient care.

## Why there are three models

The original model scored 79% accuracy, but the cohort it was trained on had a flaw: OASIS-1 records no Clinical Dementia Rating (CDR) for participants under 60, and those 201 young adults (mean age 27) were labelled cognitively normal. The healthy group was therefore about 30 years younger than the impaired groups, and the model partly learned age instead of disease.

The corrected models are trained only on the 235 CDR-rated participants, whose group ages are much closer (mean 69, 76 and 78). The original model is kept so the two can be compared side by side.

This work was accepted for presentation at ICAAD 2026.

## Models

| id | Task | Training data | Evaluation | Macro F1 | Macro AUC |
|---|---|---|---|---|---|
| `corrected` | CN / MCI / AD | 235 CDR-rated subjects (135 / 70 / 30) | 5-fold subject-level CV | 0.469 | 0.722 (95% CI 0.670–0.776) |
| `binary` | CN / impaired (CDR ≥ 0.5) | Same 235 subjects (135 / 100) | 5-fold subject-level CV | 0.658 | 0.733 |
| `original` | CN / MCI / AD | Included 201 unrated young adults as CN | Single 66-scan test split | 0.638 | 0.811 |

Macro F1 and AUC are reported instead of accuracy because the classes are imbalanced. In an earlier experiment on the corrected cohort, a decision threshold tuned for accuracy reached 58.3% accuracy while detecting no Alzheimer's cases at all.

MCI is a proxy label (CDR 0.5), not a clinical diagnosis. AD is mostly CDR 1 (mild dementia).

### What the comparison shows

| Cohort | Age-only model AUC | MRI model AUC | MRI minus age (95% CI) |
|---|---|---|---|
| Confounded (436 sessions) | 0.826 | 0.860 | +0.034 (+0.004 to +0.065) |
| Corrected (235 subjects) | 0.628 | 0.722 | +0.095 (+0.037 to +0.153) |

On the confounded cohort, logistic regression on age alone reaches 0.826 of the MRI model's 0.860 AUC, so imaging adds only 0.034. Scored on the CDR-rated participants only, the same cohort-A predictions fall from 0.860 to 0.713. On the corrected cohort, imaging adds nearly three times as much over age. Both rows use the same preprocessing and 5-fold cross-validation; intervals are from 2,000 bootstrap resamples.

## Expected input

The corrected and binary models expect a grayscale 224×224 image of a single mid-axial slice:

- Source: OASIS-1 `T88_111` atlas-registered, skull-stripped, gain-field-corrected volumes (`*_t88_masked_gfc`)
- The middle axial slice, min-max scaled to 0–255
- Copied into three channels and normalized with ImageNet statistics at inference

Colour images return `input_mismatch_warning: true`. Slices that are not atlas-registered and skull-stripped are accepted but give unreliable predictions. The sample scans on the website are in the correct format.

A three-slice variant (axial slices at −12, 0 and +12 stacked as channels) was also tested. It gave no consistent benefit and lower Alzheimer's recall in the 3-class task, so the deployed models use a single slice.

The original model was trained on single slices that were not atlas-registered.

## API

`GET /health` returns server status and the loaded model ids.

`GET /models` lists each model and whether its weights loaded.

`POST /predict` takes multipart form data:

- `file`: the image
- `model`: `corrected` (default), `binary` or `original`

Example response:

```json
{
  "model": "corrected",
  "model_name": "Corrected model (age-matched cohort)",
  "prediction": "Cognitively Normal (CN)",
  "confidence_scores": {
    "Cognitively Normal (CN)": 0.61,
    "Mild Cognitive Impairment (MCI)": 0.27,
    "Alzheimer's Disease (AD)": 0.12
  },
  "low_confidence_warning": false,
  "input_mismatch_warning": false,
  "disclaimer": "Research prototype only. Not validated for clinical use."
}
```

## Files

- `app.py`: FastAPI server and model registry
- `model.py`: ResNet-18 with a Dropout + Linear head; loads both bare and checkpoint-dict weights
- `alz_resnet18_3class_v3.pt`, `alz_resnet18_binary_v3.pt`: corrected single-slice models, trained on all 235 subjects after cross-validation
- `best_alzheimer_model.pt`: original model

Weight paths can be overridden with `CORRECTED_MODEL_PATH`, `BINARY_MODEL_PATH` and `ORIGINAL_MODEL_PATH`.

## Data acknowledgement

Data were provided by OASIS: Cross-Sectional. Principal Investigators: D. Marcus, R. Buckner, J. Csernansky, J. Morris; P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382.

Marcus, D. S., et al. (2007). Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI data in young, middle aged, nondemented, and demented older adults. *Journal of Cognitive Neuroscience*, 19(9), 1498–1507. https://doi.org/10.1162/jocn.2007.19.9.1498
