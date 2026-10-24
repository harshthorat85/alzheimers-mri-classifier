# Age confounding in MRI-based Alzheimer's classification (OASIS-1)

Code, model weights and out-of-fold predictions for *Misleading Accuracies: Age
Confounding and Cohort-Size Limits in MRI-Based Alzheimer's Classification*
(ICAAD 2026).

OASIS-1 has no Clinical Dementia Rating for 181 of its 416 participants (201
of 436 sessions). Labeling them cognitively normal makes the healthy class
decades younger than the impaired classes, so a model can score well by
learning age. In a reconstruction of that labeling (cohort A), a ResNet-18
reaches 75.7% accuracy, below the 77.1% majority-class rate, and a logistic
regression on age alone reaches macro AUC 0.826 against the imaging model's
0.860. On the 235 rated participants (cohort B) the imaging model reaches
0.722 against 0.628 for age: real signal, but about a tenth of an AUC.

**Demo:** https://alzheimers-mri-classifier.netlify.app
**Model weights and demo backend:** https://huggingface.co/spaces/harshthorat85/alzheimers-mri-classifier

## Layout

| Folder | Contents |
|---|---|
| `backend/` | FastAPI inference server; its README documents the models, expected input and API |
| `frontend/` | Static demo page |
| `training/` | Data extraction, cross-validation, ablations and the original model, in run order |
| `oof/` | Out-of-fold predictions behind every reported number |

The weights (about 45 MB each) are in the Hugging Face Space, not in this
repository. `backend/` is mirrored there; this repository is canonical for code.

## Out-of-fold predictions

| File | Shape | Rows |
|---|---|---|
| `oof/oof_A_3class.npz` | 436 x 3 | cohort A, all sessions |
| `oof/oof_B_3class.npz` | 235 x 3 | cohort B, CDR-rated participants |
| `oof/oof_B_binary.npz` | 235 x 2 | cohort B, CN vs impaired |
| `oof/oof_B_weighting.npz` | 3 x (235 x 3) | cohort B, one array per class-weighting scheme |

Each archive holds `ids`, `probs` and `labels` (the weighting file holds `ids`
and one array per scheme). Rows follow the clinical spreadsheet's order; align
by `ids`. See `training/README.md` for the expected figures and for how to
regenerate everything.

## Scope

Research prototype; not validated for clinical use. One dataset, one
architecture, and 235 participants in the corrected cohort, too few for the
three-class problem.

## Data

Data were provided by OASIS: Cross-Sectional. Principal Investigators:
D. Marcus, R. Buckner, J. Csernansky, J. Morris; P50 AG05681, P01 AG03991,
P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382. No OASIS images are
included here.

Marcus, D. S., et al. (2007). Open Access Series of Imaging Studies (OASIS):
Cross-sectional MRI data in young, middle aged, nondemented, and demented older
adults. *Journal of Cognitive Neuroscience*, 19(9), 1498-1507.
https://doi.org/10.1162/jocn.2007.19.9.1498
