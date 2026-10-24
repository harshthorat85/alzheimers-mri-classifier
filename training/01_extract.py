import tarfile, shutil, numpy as np, pandas as pd, nibabel as nib
from pathlib import Path
from skimage.transform import resize

ROOT = Path(r'C:\AI Project\Datasets\OASIS-1')
AXIS = 2
XLSX = ROOT / 'oasis_cross-sectional-5708aa0a98d82080.xlsx'
OUT  = ROOT / 'oasis1_slices_1ch_all436.npz'

df = pd.read_excel(XLSX)
keep = set(df['ID'])
print(f'sessions to extract: {len(keep)}')

slices = {}
for d in range(1, 13):
    arc = ROOT / f'oasis_cross-sectional_disc{d}.tar.gz'
    if TMP.exists(): shutil.rmtree(TMP)
    TMP.mkdir()

    with tarfile.open(arc) as t:
        mem = [m for m in t.getmembers()
               if '111_t88_masked_gfc.' in m.name
               and m.name.split('/')[1] in keep]
        t.extractall(TMP, members=mem)

    for hdr in TMP.rglob('*111_t88_masked_gfc.hdr'):
        sid = hdr.name.split('_mpr')[0]
        if sid not in keep or sid in slices:
            continue
        vol = np.squeeze(nib.load(hdr).get_fdata())
        sl  = np.take(vol, vol.shape[AXIS] // 2, axis=AXIS).T
        sl  = resize(sl, (224, 224), anti_aliasing=True)
        rng = sl.max() - sl.min()
        slices[sid] = ((sl - sl.min()) / (rng if rng else 1) * 255).astype(np.uint8)

    print(f'disc {d:2d}  ->  {len(slices)} / {len(keep)}')

if TMP.exists(): shutil.rmtree(TMP)

missing = keep - set(slices)
if missing:
    print(f'\nWARNING: {len(missing)} missing:', sorted(missing)[:10])

ids = sorted(slices)
np.savez_compressed(OUT, ids=np.array(ids), imgs=np.stack([slices[i] for i in ids]))
print(f'\nsaved {OUT.name}  —  {len(ids)} slices, {OUT.stat().st_size / 1e6:.1f} MB')