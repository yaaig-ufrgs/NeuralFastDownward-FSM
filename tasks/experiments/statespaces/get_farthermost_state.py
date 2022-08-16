from sys import argv

for file in argv[1:]:
    with open(file) as f:
        fm = -1
        for h in [int(x.strip().split(";")[0]) for x in f.readlines() if not x.startswith("#")]:
            fm = max(fm, h)
        print(file, fm)
