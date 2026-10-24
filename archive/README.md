# Archive: three-slice experiments (superseded)

`ai_project_improvement.py` and `aineuro2.py` use `oasis1_slices_3ch*.npz`, in
which the three channels are three different axial slices (offsets -12, 0 and
+12 from the middle) rather than one slice copied three times. `aineuro2.py`
also exported the v2 weights, which are not distributed.

These scripts are kept for the record only:

- The script that built the three-slice `.npz` files is no longer available, so
  these runs cannot be regenerated from this repository.
- The paper does not report a three-slice comparison. The single-slice control
  run did not reproduce the published figure (AUC 0.695 against 0.722), so no
  valid comparison between the two inputs exists.
- Metrics hardcoded in the v2 checkpoint metadata come from this superseded
  pipeline. Do not cite them as results of the paper.

The paper's pipeline is in the parent folder.
