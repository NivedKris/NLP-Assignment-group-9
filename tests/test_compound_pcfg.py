"""Correctness check: vectorised inside / Viterbi vs. brute-force enumeration of every labelled
binary tree on tiny random grammars. Run: python tests/test_compound_pcfg.py"""
import itertools
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)  # show progress live, also when piped to tee
from common.compound_pcfg import inside, viterbi


def trees(i, j):
    """all binary bracketings of words i..j inclusive, as nested tuples"""
    if i == j:
        yield i
        return
    for k in range(i, j):
        for l in trees(i, k):
            for r in trees(k + 1, j):
                yield (l, r)


def internal(t):
    return [] if isinstance(t, int) else [t] + internal(t[0]) + internal(t[1])


def score(t, labels, unary, rule, NT):
    """labels: dict node -> state; leaves use preterminal state indices NT.."""
    if isinstance(t, int):
        return unary[t, labels[t] - NT]
    a, b, c = labels[id(t)], labels[id(t[0])] if not isinstance(t[0], int) else labels[t[0]], \
        labels[id(t[1])] if not isinstance(t[1], int) else labels[t[1]]
    return rule[a, b, c] + score(t[0], labels, unary, rule, NT) + score(t[1], labels, unary, rule, NT)


def brute(unary, rule, root, NT, T):
    n = unary.shape[0]
    total, best = [], -math.inf
    for t in trees(0, n - 1):
        nodes = internal(t)
        for nt_lab in itertools.product(range(NT), repeat=len(nodes)):
            for t_lab in itertools.product(range(NT, NT + T), repeat=n):
                labels = {id(x): a for x, a in zip(nodes, nt_lab)}
                labels.update({i: s for i, s in enumerate(t_lab)})
                s = (root[labels[id(t)]] + score(t, labels, unary, rule, NT)).item()
                total.append(s)
                best = max(best, s)
    return torch.logsumexp(torch.tensor(total, dtype=torch.float64), 0).item(), best


torch.manual_seed(0)
for n in (2, 3, 4, 5):
    NT, T = 2, 3
    S = NT + T
    unary = torch.log_softmax(torch.randn(1, n, T), -1)
    rule = torch.log_softmax(torch.randn(1, NT, S * S), -1).view(1, NT, S, S)
    root = torch.log_softmax(torch.randn(1, NT), -1)
    lz = inside(unary, rule, root, NT).item()
    spans, tags = viterbi(unary, rule, root, NT)
    ref_lz, ref_max = brute(unary[0], rule[0], root[0], NT, T)
    # score of the Viterbi tree, recomputed from its spans
    assert abs(lz - ref_lz) < 1e-4, (n, lz, ref_lz)
    print(f"n={n}: inside {lz:.5f} == brute {ref_lz:.5f}; viterbi tree spans {spans[0]}; "
          f"brute-force best log-prob {ref_max:.5f}")
    # Viterbi value check: rebuild the score of the returned tree
    lab = {(a, b): s for a, b, s in spans[0]}

    def vscore(i, j):
        if i == j:
            return unary[0, i, tags[0][i]].item(), NT + tags[0][i]
        a = lab[(i, j)]
        for k in range(i, j):
            left_ok = (k == i) or (i, k) in lab
            right_ok = (k + 1 == j) or (k + 1, j) in lab
            if left_ok and right_ok:
                ls, lb = vscore(i, k)
                rs, rb = vscore(k + 1, j)
                return ls + rs + rule[0, a, lb, rb].item(), a
        raise AssertionError("inconsistent spans")

    vs, va = vscore(0, n - 1)
    vs += root[0, va].item()
    assert abs(vs - ref_max) < 1e-4, (n, vs, ref_max)
print("compound PCFG inside/Viterbi: all checks passed")
