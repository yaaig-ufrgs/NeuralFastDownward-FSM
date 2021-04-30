#!/usr/bin/env python
"""
analyse h value distribution in the matching data files
"""

from __future__ import print_function

import matplotlib
matplotlib.use('agg')

import argparse
import gzip
import json
import os
import matplotlib.pyplot as plt
import shutil
import sys

DEBUG = False
if DEBUG:
    print("INFO> DEBUG MODE ACTIVE")

# Python 2/3 compatibility
gzip_write_converter = lambda x: x.encode()
gzip_read_converter = lambda x: x.decode()
if sys.version_info[0] == 2:
    gzip_write_converter = lambda x: x
    gzip_read_converter = lambda x: x

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

IS_NEW_MATPLOTLIB = matplotlib.compare_versions(matplotlib.__version__, "2.2.2")

EXITCODE_SUCCESS = 0
EXITCODE_GZIP_ERROR = 1

COLORS = ["r", "g", "b", "m", "c", "dimgrey", "lightcoral", "orange", "lime", "darkorchid"]

parser = argparse.ArgumentParser()
parser.add_argument("--count-unique-samples", action="store_true",
                    help="Counts how many unique samples are in the data sets ("
                         "the same state present in two differnt StateFormats "
                         "will NOT be detect  as duplicate currently).")
parser.add_argument("--debug", action="store_true",
                    help="Run the script in debug mode")
parser.add_argument("-d", "--directory", type=str, action="append", default=[],
                    help="Directory under which all directories containing"
                         "domain.pddl shall be analysed.")
parser.add_argument("-p", "--properties", type=str, action="append", default=[],
                    help="Path to a Lab properties file. The plan lengths are"
                         "extracted from there and their distribution is added.")
parser.add_argument("--redraw-domain-common", action="store_true")
parser.add_argument("--skip-if-counted", action="store_true",
                    help="Skips reading the heuristic values if the suffix is "
                         "already stored in the cache")
parser.add_argument("-s", "--suffix", type=str, nargs="+", action="append", default=[],
                    help="Only files with this suffix are analysed. If multiple"
                         " times given, they are individually analysed. If "
                         "additional parameters are given as only the suffix,"
                         "then only entries are counted where the sample type "
                         "is within those additional parameters")

def find_domains(path):
    path_domains = []
    todo = [path]
    while len(todo) > 0:
        path_dir = todo.pop()
        path_domain_file = os.path.join(path_dir, "domain.pddl")
        if os.path.isfile(path_domain_file):
            path_domains.append(path_dir)
        for item in os.listdir(path_dir):
            path_item = os.path.join(path_dir, item)
            if os.path.isdir(path_item):
                todo.append(path_item)
    return path_domains



def split_meta(line):
    line = line.strip()
    if not line.startswith("{"):
        return None, line

    level = 1
    for i in range(1, len(line)):
        if line[i] == "{":
            level += 1
        elif line[i] == "}":
            level -= 1
            if level == 0:
                return json.loads(line[: i + 1]), line[i + 1:]

    raise ValueError("Entry has an opening, but not closing tag: " + line)


def remove_leading_dots(s):
    while len(s) > 0 and s[0] == ".":
        s = s[1:]
    return s


def plot_hist(path, suffix, h_counts, total_count, unique=False):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    h_counts = h_counts.copy()
    for i in range(max(h_counts.keys())):
        if i not in h_counts:
            h_counts[i] = 0

    ax.bar(h_counts.keys(), h_counts.values())

    ax.set_xlabel("heuristic value")
    ax.set_ylabel("count")

    sum_h_counts = sum(c for _, c in h_counts.items())
    ax.set_title(
        "Histogram of h* occurrences in training data %s\n(%stotal samples %s%i)"
        % (suffix,
           "unique/" if unique else "",
           "%i/" % sum_h_counts if unique else "",
           total_count))

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def join_nested_list(nested):
    l = []
    for n in nested:
        l.extend(n)
    return l

def plot_common_hist(path_dir, all_h_counts, all_total_count, unique=False):
    global IS_NEW_MATPLOTLIB
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    n_bins = int(max([max(all_h_counts[suffix].keys()) for suffix in all_h_counts])) + 1
    datas = []
    for suffix in sorted(all_h_counts.keys()):
        counts = all_h_counts[suffix]
        data = join_nested_list([[int(v) for _ in range(counts[v])]for v in counts])
        datas.append(data)
    labels = ["%s\n(%s samples %s%.2E)" %
              (label,
               "unique/total" if unique else "",
               "%.2E/" % len(datas[no]) if unique else "",
               all_total_count[label])
              for no, label in enumerate(sorted(all_h_counts.keys()))]

    density_arg = ({"density": True} if IS_NEW_MATPLOTLIB else {"normed": True})
    ax.hist(datas, n_bins, color=COLORS[:len(all_h_counts.keys())],
            label=labels, **density_arg)

    ax.set_xlabel("heuristic value")
    ax.set_ylabel("fraction")
    ax.set_title("Histogram of h* occurrences in training data")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(
        path_dir, "data_distribution%s.pdf" %
                  ("_unique" if unique else "")))
    plt.close(fig)


def get_path_cache(path_dir, unique=False):
    return os.path.join(path_dir, "data_distributions%s.json" %
                        ("_unique" if unique else ""))

def fix_cache_int(cache):
    for dkey in cache.keys():
        for h in sorted(cache[dkey].keys()):
            if not isinstance(h, int):
                cache[dkey][int(h)] = cache[dkey][h]
                del cache[dkey][h]
    return cache

def load_cache(path_dir, unique=False):
    path = get_path_cache(path_dir, unique=unique)
    if os.path.exists(path):
        with open(path, "r") as f:
            cache = json.load(f)
            if isinstance(cache, list) and len(cache) == 2:
                return fix_cache_int(cache[0]), cache[1]
            elif isinstance(cache, dict):
                return fix_cache_int(cache), {}
            else:
                assert False

    else:
        return {}, {}

def save_cache(path_dir, h_counts, total_counts, unique=False):
    path_cache = get_path_cache(path_dir, unique=unique)
    path_tmp = path_cache + ".tmp"
    with open(path_tmp, "w") as f:
        json.dump([h_counts, total_counts], f)
    shutil.move(path_tmp, path_cache)


def analyse_domain_data(path_dir, suffixes, skip_existing, redraw_domain_common,
                        count_unique_samples):
    data_files = {}  # {suffix: [paths]}
    for item in os.listdir(path_dir):
        path_item = os.path.join(path_dir, item)
        for suffix in suffixes:
            if item.endswith(suffix):
                if suffix not in data_files:
                    data_files[suffix] = []
                data_files[suffix].append(path_item)

    all_h_counts, all_total_count = load_cache(path_dir, unique=count_unique_samples)
    new_suffixes = False

    for suffix in sorted(data_files.keys()):
        print("\tSuffix: %s" % suffix)
        dset_keys = set()
        stype2dset_keys = {}
        for sample_types in suffixes[suffix]:
            dset_key = suffix if sample_types is None else str((suffix, tuple(sample_types)))
            dset_keys.add(dset_key)
            if sample_types is not None:
                for sample_type in sample_types:
                    if sample_type not in stype2dset_keys:
                        stype2dset_keys[sample_type] = []
                    stype2dset_keys[sample_type].append(dset_key)

        if skip_existing and all([key in all_h_counts for key in dset_keys]):
            print("\t\tSkip")
            continue
        new_suffixes = True

        # Load dataset and count h values
        h_counts = {k: {} for k in dset_keys}  # {dset_key: {h: count}
        total_count = {k: 0 for k in dset_keys}  # {dset_key : count}
        had_ioerror = False
        for file in data_files[suffix][:3] if DEBUG else data_files[suffix]:
            print("file: %s" % file)
            try:
                with gzip.open(file, "r") as f:
                    for line in f:
                        line = gzip_read_converter(line)
                        if line == "" or line.startswith("#"):
                            continue

                        meta, entry = split_meta(line)
                        sample_type = meta["sample_type"]
                        fields = entry.split(meta["delimiter"])
                        # Get h value
                        idx_heur = [i for i in range(len(meta["fields"])) if meta["fields"][i]["name"] == "hplan"]
                        assert len(idx_heur) == 1
                        idx_heur = idx_heur[0]
                        h_value = int(fields[idx_heur].strip())

                        if count_unique_samples:
                            idx_current_state = [i for i in range(len(meta["fields"])) if
                                        meta["fields"][i]["name"] == "current_state"]
                            assert len(idx_current_state) == 1
                            idx_current_state = idx_current_state[0]
                            idx_goals = [i for i in range(len(meta["fields"])) if
                                        meta["fields"][i]["name"] == "goals"]
                            assert len(idx_goals) == 1
                            idx_goals = idx_goals[0]
                            unique_key = (fields[idx_current_state].strip(),
                                          fields[idx_goals].strip())

                        for dset_key in stype2dset_keys.get(sample_type,[]) + [suffix]:
                            total_count[dset_key] += 1
                            if h_value not in h_counts[dset_key]:
                                h_counts[dset_key][h_value] = set() if count_unique_samples else 0
                            if count_unique_samples:
                                h_counts[dset_key][h_value].add(unique_key)
                            else:
                                h_counts[dset_key][h_value] += 1

            except IOError as e:
                had_ioerror = True
                print("IOError in: %s\n%s" % (file, str(e)))
        if had_ioerror:
            sys.exit(EXITCODE_GZIP_ERROR)

        for dset_key in dset_keys:
            if count_unique_samples:
                for h_value in h_counts[dset_key].keys():
                    h_counts[dset_key][h_value] = len(h_counts[dset_key][h_value])
            all_h_counts[dset_key] = h_counts[dset_key]
            all_total_count[dset_key] = total_count[dset_key]
        save_cache(path_dir, all_h_counts, all_total_count,
                   unique=count_unique_samples)
        #plot_hist(os.path.join(
        #    path_dir, "%s_h_distribution%s.pdf" %
        #              (remove_leading_dots(suffix),
        #               "_unique" if count_unique_samples else "")),
        #    suffix, h_counts, total_count, count_unique_samples)

    if (new_suffixes or redraw_domain_common) and len(all_h_counts.keys()) > 0:
        plot_common_hist(path_dir, all_h_counts, all_total_count,
                         unique=count_unique_samples)


def load_properties(properties):
    domain_counts = {}
    for path_prob in properties:
        with open(path_prob, "r") as f:
            lab_properties = json.load(f)
        for name, properties in lab_properties.items():
            domain = properties["domain"]
            algorithm = properties["algorithm"]
            plan_length = properties.get("plan_length", None)
            if plan_length is not None:
                if domain not in domain_counts:
                    domain_counts[domain] = {}
                if algorithm not in domain_counts[domain]:
                    domain_counts[domain][algorithm] = {}
                if plan_length not in domain_counts[domain][algorithm]:
                    domain_counts[domain][algorithm][plan_length] = 0
                domain_counts[domain][algorithm][plan_length] += 1

    return domain_counts

def parse_argv(argv):
    global DEBUG
    options = parser.parse_args(argv)
    if options.debug:
        DEBUG = True
    for path_dir in options.directory:
        assert os.path.isdir(path_dir)
    for path_prob in options.properties:
        assert os.path.isfile(path_prob)
    suffix = {}
    for dataset in options.suffix:
        if dataset[0] not in suffix:
            suffix[dataset[0]] = []
        suffix[dataset[0]].append(None if len(dataset) == 1 else dataset[1:])
    options.suffix = suffix
    return options

def cache_properties_counts(path_dir, algo_counts):
    counts = load_cache(path_dir)
    counts.update(algo_counts)
    save_cache(path_dir, counts)

def run(argv):
    options = parse_argv(argv)

    property_counts = load_properties(options.properties)

    domains = set()
    for d in options.directory:
        domains.update(find_domains(d))


    for domain in sorted(domains):
        print("Domain: %s" % domain)
        domain_identifier = os.path.basename(domain)
        if domain_identifier in property_counts:
            cache_properties_counts(domain, property_counts[os.path.basename(domain)])
        analyse_domain_data(domain, options.suffix, options.skip_if_counted,
                            options.redraw_domain_common,
                            options.count_unique_samples)
    print("Done.")


if __name__ == "__main__":
    run(sys.argv[1:])

