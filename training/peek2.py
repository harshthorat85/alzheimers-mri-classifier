import tarfile

ARC = r'C:\AI Project\Datasets\OASIS-2\OAS2_RAW_PART1.tar.gz'

with tarfile.open(ARC) as t:
    names = []
    for m in t:
        names.append(m.name)
        if len(names) > 60:
            break

for n in names:
    print(n)