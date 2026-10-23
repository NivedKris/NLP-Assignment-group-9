"""Shared data code for Project 3, Partition C (Group 9).

Implements the dataset construction procedure from the Datasets document, Section 3:
  1. shuffle all 3,914 NLTK treebank sentence indices once with numpy.random.seed(42),
     cut into thirds: [0:1305]=A, [1305:2610]=B, [2610:3914]=C
  2. second, independent shuffle inside the partition with the same seed (42),
     then 80/10/10 train/validation/test
  3. strip functional tags and trace indices from labels (NP-SBJ-1 -> NP)
"""
import json
import os
import re

import numpy as np
import nltk
from nltk.tree import Tree

N_TOTAL = 3914
SEED = 42
PARTITIONS = {"A": (0, 1305), "B": (1305, 2610), "C": (2610, 3914)}
PARTITION = "C"  # Group 9 is assigned Partition C

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_FILE = os.path.join(ROOT_DIR, "data", "splits", "partition_splits_seed42.json")


def ensure_treebank():
    try:
        nltk.data.find("corpora/treebank")
    except LookupError:
        nltk.download("treebank", quiet=True)


def load_all_trees():
    ensure_treebank()
    from nltk.corpus import treebank
    trees = list(treebank.parsed_sents())
    assert len(trees) == N_TOTAL, f"expected {N_TOTAL} sentences, got {len(trees)}"
    return trees


def make_splits(seed=SEED):
    """Returns {partition: {"indices": [...], "train": [...], "validation": [...], "test": [...]}}."""
    np.random.seed(seed)
    perm = np.random.permutation(N_TOTAL)
    out = {}
    for name, (a, b) in PARTITIONS.items():
        part = perm[a:b]
        np.random.seed(seed)  # second, independent shuffle with the same fixed seed
        p = np.random.permutation(part)
        n = len(p)
        n_train, n_trainval = int(0.8 * n), int(0.9 * n)
        out[name] = {
            "indices": [int(i) for i in part],
            "train": [int(i) for i in p[:n_train]],
            "validation": [int(i) for i in p[n_train:n_trainval]],
            "test": [int(i) for i in p[n_trainval:]],
        }
    return out


def save_splits(path=SPLIT_FILE):
    splits = make_splits()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"seed": SEED, "n_total": N_TOTAL, "partitions": splits}, f)
    return splits


def load_splits(partition=PARTITION, path=SPLIT_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run scripts/00_make_splits.py first")
    with open(path) as f:
        return json.load(f)["partitions"][partition]


# ---------------------------------------------------------------- tree cleaning
def strip_label(label):
    """NP-SBJ-1 -> NP, NP=2 -> NP; keeps -LRB-, -RRB-, -NONE- intact.
    '|' and '+' are reserved by the CNF transform, so they are replaced by '_'."""
    if label.startswith("-"):
        return label
    base = re.split(r"[-=]", label)[0] or label
    return base.replace("|", "_").replace("+", "_")


def clean_tree(t):
    """Strip functional tags / trace indices and delete empty elements (-NONE-) and the
    constituents that become empty. Returns None if nothing is left."""
    if isinstance(t, str):
        return t
    if t.label() == "-NONE-":
        return None
    kids = [clean_tree(c) for c in t]
    kids = [k for k in kids if k is not None]
    if not kids:
        return None
    return Tree(strip_label(t.label()), kids)


def load_partition(partition=PARTITION):
    """Returns dict split -> list of (corpus_index, cleaned Tree)."""
    trees = load_all_trees()
    splits = load_splits(partition)
    data = {}
    for name in ("train", "validation", "test"):
        items = []
        for i in splits[name]:
            t = clean_tree(trees[i])
            assert t is not None
            items.append((i, t))
        data[name] = items
    return data


def tree_to_line(t):
    return t.pformat(margin=10**9)
