import tarfile, shutil, numpy as np, nibabel as nib, matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(r'C:\AI Project\Datasets\OASIS-1')
TMP  = ROOT / '_tmp'
if TMP.exists(): shutil.rmtree(TMP)
TMP.mkdir()

with tarfile.open(ROOT / 'oasis_cross-sectional_disc1.tar.gz') as t:
    mem = [m for m in t.getmembers()
           if 'OAS1_0001_MR1' in m.name and '111_t88_masked_gfc.' in m.name]
    print('pulling:', [m.name for m in mem])
    t.extractall(TMP, members=mem)

hdr = next(TMP.rglob('*111_t88_masked_gfc.hdr'))
vol = np.squeeze(nib.load(hdr).get_fdata())
print('volume shape:', vol.shape)

fig, ax = plt.subplots(1, 3, figsize=(10, 4))
for i, a in enumerate(ax):
    a.imshow(np.take(vol, vol.shape[i] // 2, axis=i).T, cmap='gray', origin='lower')
    a.set_title(f'axis {i}  (size {vol.shape[i]})')
    a.axis('off')
plt.tight_layout()
plt.savefig(ROOT / 'orientation_check.png', dpi=110)
plt.show()