#/usr/bin/env python3

# usage: ./check_sampling_inconsistencies.py statespace_file samples_file [samples_file ...]

from sys import argv

def read_samples(file: str) -> [(int, str)]:
    with open(file) as f:
        return [(int(l.split(";")[0]), l.strip().split(";")[1]) \
                   for l in f.readlines() if l and l[0] != "#"]

def pct(n: int, total: int, decimals: int = 2) -> str:
    return f"{round(100*n/total, decimals)}%"

statespace = {}
duplicates_in_ss = 0
for h, s in read_samples(argv[1]):
    # assert s not in statespace
    if s in statespace:
        assert statespace[s] == h
        duplicates_in_ss += 1
    else:
        statespace[s] = h
if duplicates_in_ss:
    print(f"*** State space has {duplicates_in_ss} duplicated samples.")

for samples_file in argv[2:]:
    print(f"--- {samples_file} ---")
    samples = read_samples(samples_file)
    with_hstar = 0
    underestimates = 0
    not_in_statespace = 0
    for h, s in samples:
        if s in statespace:
            hstar = statespace[s]
            if h == hstar:
                with_hstar += 1
            elif h < hstar:
                underestimates += 1
        else:
            not_in_statespace += 1
    total = len(samples)
    print(f" | Samples: {total}")
    print(f" | Samples with h*: {with_hstar} ({pct(with_hstar, total)})")
    print(f" | Samples that underestimate h*: {underestimates} ({pct(underestimates, total)})")
    print(f" | Samples not in state space: {not_in_statespace} ({pct(not_in_statespace, total)})")
