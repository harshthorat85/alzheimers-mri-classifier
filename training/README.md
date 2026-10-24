# Training and evaluation code

Everything needed to regenerate the paper's numbers, in run order. Colab
notebooks assume a GPU runtime and the file names used in the code
(`/content/...`); `01_extract.py` runs locally.

| Step | File | Needs | Produces |
|---|---|---|---|
| 1 | `01_extract.py` | OASIS-1 discs 1-12 (`.tar.gz`) and the clinical spreadsheet | `oasis1_slices_1ch_all436.npz`: `ids` (436, sorted) and `imgs` (436, 224, 224) uint8 |
| 2 | `02_cv_and_final_fit.ipynb` | step 1 output, spreadsheet | Out-of-fold (OOF) predictions for cohort A (confounded, 436 sessions) and cohort B (corrected, 235 participants), 3-class and binary; age-only baselines; bootstrap CIs; rescoring on rated participants; the released v3 weights |
| 3 | `03_weighting_ablation.ipynb` | step 1 output, spreadsheet | Class-weighting comparison (full inverse-frequency, square-root, none), all with label smoothing 0.05 |
| 4 | `04_age_mri_stacking.ipynb` | OOF files from step 2 | Age + MRI stacking comparison |
| - | `original_model.ipynb` | OASIS scan folders in Google Drive | The original model (`best_alzheimer_model.pt`) |
| - | `../archive/` | - | Superseded three-slice experiments; see its README |

## Expected results

Cohort A (confounded; 436 sessions, grouped 5-fold CV):
accuracy 0.757; macro AUC 0.860 for the imaging model against 0.826 for age
alone (difference +0.034, 95% CI +0.004 to +0.065). Rescored on CDR-rated
participants only, imaging falls to 0.713 and age to 0.617.

Cohort B (corrected; 235 participants, stratified 5-fold CV):
accuracy 0.562; macro AUC 0.722 (95% CI 0.670-0.776) against 0.628 for age
alone (difference +0.095, 95% CI +0.037 to +0.153). Binary CN vs impaired:
accuracy 0.660, macro F1 0.658, AUC 0.733.

Class weighting on cohort B:

| Weighting | Accuracy | Macro AUC | AD recall |
|---|---|---|---|
| Full inverse-frequency | 0.515 | 0.712 | 17 of 30 |
| Square-root | 0.562 | 0.722 | 11 of 30 |
| None | 0.596 | 0.728 | 7 of 30 |

## Reproducibility

- Step 1 regenerates the slice file exactly (verified identical, pixel for pixel,
  against the file used in the paper).
- Everything downstream of the OOF files (age baselines, bootstrap intervals,
  rescoring, stacking) is deterministic: all splits use seed 42 and the
  bootstrap uses seed 0. Verified with scikit-learn 1.6.1; fold assignment can
  change between versions.
- CNN training is not bit-reproducible on a GPU. One rerun of the single-slice
  pipeline gave a macro AUC of 0.695 against the published 0.722, so retraining
  will land near, not on, the reported figures. The OOF files in `../oof/` are
  the reference for every number above.

## Row order

The slice file is sorted by session ID. The OOF files follow the row order of
the clinical spreadsheet. Align them by `ids`, never by position. Sorting the
spreadsheet by ID would interleave each rescan (MR2) with its MR1 session and
silently misalign cohort A.

## The original model

`original_model.ipynb` trained on whole-head, mid-sagittal slices that include
the skull and come from scans that were not atlas-registered, unlike the axial,
skull-stripped slices used everywhere else here.
It split sessions at random without grouping by participant, so 3 of its 66 test
scans come from a person who also appears in training or validation. It is
kept for provenance; the paper's comparisons use cohort A (a reconstruction of
the confounded labeling), not this model.

## Data

OASIS-1 data are not included and are governed by the OASIS data use
agreement. Slice files are derived from OASIS images and must not be
redistributed; regenerate them with `01_extract.py`.
