import tarfile

ARC = r'C:\AI Project\Datasets\OASIS-1\oasis_cross-sectional_disc1.tar.gz'

with tarfile.open(ARC) as t:
    names = t.getnames()

print('total entries:', len(names))
print()
for n in names[:40]:
    print(n)