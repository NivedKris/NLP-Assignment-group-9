"""CNF conversion, unknown-word signatures, PCFG grammar container and exact CKY (Viterbi) parser.
Shared by Task 1 (count-based PCFG) and Task 3 (neural PCFG decoded with the same CKY)."""
import sys
from collections import Counter

import numpy as np
from nltk.tree import Tree

sys.setrecursionlimit(10000)
TOP = "TOP"


# ------------------------------------------------------------------ CNF transform
def to_cnf(tree, horz_markov):
    """TOP-wrap, collapse unary chains (incl. into POS), then binarize with horizontal
    markovization of order `horz_markov` (0 or 1), no vertical (parent) annotation."""
    t = Tree(TOP, [tree.copy(deep=True)])
    t.collapse_unary(collapsePOS=True, collapseRoot=False, joinChar="+")
    t.chomsky_normal_form(factor="right", horzMarkov=horz_markov, vertMarkov=0,
                          childChar="|", parentChar="^")
    return t


def from_cnf(t):
    """Undo binarization + unary collapsing, drop TOP."""
    t = t.copy(deep=True)
    t.un_chomsky_normal_form(expandUnary=True, childChar="|", parentChar="^", unaryChar="+")
    if t.label() == TOP:
        assert len(t) == 1
        t = t[0]
    return t


def extract_rules(cnf_tree):
    """Yields ('root', X) | ('bin', A, B, C) | ('lex', A, word)."""
    assert cnf_tree.label() == TOP and len(cnf_tree) == 1
    out = [("root", cnf_tree[0].label())]
    stack = [cnf_tree[0]]
    while stack:
        t = stack.pop()
        if len(t) == 1 and isinstance(t[0], str):
            out.append(("lex", t.label(), t[0]))
        elif len(t) == 2 and not isinstance(t[0], str) and not isinstance(t[1], str):
            out.append(("bin", t.label(), t[0].label(), t[1].label()))
            stack.extend([t[0], t[1]])
        else:
            raise ValueError(f"tree not in CNF at node {t.label()} with {len(t)} children")
    return out


# ------------------------------------------------------------------ unknown words
def signature(w):
    """Berkeley-parser-style unknown word class."""
    sig = "UNK"
    if any(c.isdigit() for c in w):
        sig += "-NUM"
    if w[:1].isupper():
        sig += "-CAPS" if any(c.islower() for c in w) else "-ALLCAPS"
    elif any(c.isupper() for c in w):
        sig += "-MIXED"
    if "-" in w:
        sig += "-DASH"
    lw = w.lower()
    for suf in ("ing", "ed", "ly", "ion", "er", "est", "al", "ity", "ive", "able", "ic", "s", "y"):
        if lw.endswith(suf) and len(lw) > len(suf) + 1:
            sig += "-" + suf
            break
    return sig


class WordMapper:
    """Words seen >= min_count times in training stay; rarer/unseen words map to signatures;
    unseen signatures back off to the generic class 'UNK*'."""
    GENERIC = "UNK*"

    def __init__(self, train_words, min_count=2):
        self.counts = Counter(train_words)
        self.min_count = min_count
        self.known = {w for w, c in self.counts.items() if c >= min_count}
        self.vocab = None  # set after building lexicon

    def train_key(self, w):
        return w if w in self.known else signature(w)

    def test_key(self, w):
        if w in self.known:
            return w
        s = signature(w)
        if self.vocab is not None and s in self.vocab:
            return s
        return self.GENERIC


# ------------------------------------------------------------------ grammar + CKY
class Grammar:
    """symbols: list of names (TOP excluded).
    binary: list of (A, B, C, logp) with symbol ids.
    lexicon: dict key -> np.array(len(symbols)) of log P(A -> key) (-inf where impossible).
    root: np.array(len(symbols)) of log P(TOP -> X)."""

    def __init__(self, symbols, binary, lexicon, root, mapper):
        self.symbols = list(symbols)
        self.S = len(self.symbols)
        b = sorted(binary, key=lambda r: (r[0], r[1], r[2]))
        self.P = np.array([r[0] for r in b], dtype=np.int64)
        self.L = np.array([r[1] for r in b], dtype=np.int64)
        self.R = np.array([r[2] for r in b], dtype=np.int64)
        self.logp = np.array([r[3] for r in b], dtype=np.float32)
        self.parents, self.starts = np.unique(self.P, return_index=True)
        self.ends = np.append(self.starts[1:], len(self.P))
        self.seg = {int(a): (int(s), int(e)) for a, s, e in zip(self.parents, self.starts, self.ends)}
        self.lexicon = {k: v.astype(np.float32) for k, v in lexicon.items()}
        self.root = root.astype(np.float32)
        self.mapper = mapper
        self.mapper.vocab = set(self.lexicon)

    def n_rules(self):
        lex = sum(int(np.isfinite(v).sum()) for v in self.lexicon.values())
        return {"binary": int(len(self.P)), "lexical": lex,
                "root": int(np.isfinite(self.root).sum()), "symbols": self.S}

    def lex_vec(self, w, backoff=False):
        v = self.lexicon[self.mapper.test_key(w)]
        if backoff and self.mapper.GENERIC in self.lexicon:
            # lexical back-off used only on a retry: also allow every tag an unknown word may take
            v = np.logaddexp(v, self.lexicon[self.mapper.GENERIC] + np.log(1e-3)).astype(np.float32)
        return v

    def parse_robust(self, words):
        """CKY; if no complete parse exists, retry once with lexical back-off; if that still
        fails return a flat fallback tree. Returns (CNF tree, status) with status in
        {'ok', 'backoff', 'fallback'}."""
        t, _ = self.parse(words)
        if t is not None:
            return t, "ok"
        t, _ = self.parse(words, backoff=True)
        if t is not None:
            return t, "backoff"
        return self.fallback(words), "fallback"

    def parse(self, words, backoff=False):
        """Exact Viterbi CKY, O(n^3 |G|). Returns (CNF tree, logprob) or (None, -inf)."""
        n, S = len(words), self.S
        chart = np.full((n + 1, n + 1, S), -np.inf, dtype=np.float32)
        for i, w in enumerate(words):
            chart[i, i + 1] = self.lex_vec(w, backoff)
        with np.errstate(invalid="ignore"):
            for length in range(2, n + 1):
                for i in range(0, n - length + 1):
                    j = i + length
                    ks = np.arange(i + 1, j)
                    left = chart[i, ks][:, self.L]      # (K, R)
                    right = chart[ks, j][:, self.R]     # (K, R)
                    best = (left + right).max(axis=0) + self.logp
                    chart[i, j, self.parents] = np.maximum.reduceat(best, self.starts)
        top = chart[0, n] + self.root
        x = int(np.argmax(top))
        if not np.isfinite(top[x]):
            return None, -np.inf
        return Tree(TOP, [self._build(chart, words, 0, n, x)]), float(top[x])

    def _build(self, chart, words, i, j, a):
        if j == i + 1:
            return Tree(self.symbols[a], [words[i]])
        s, e = self.seg[a]
        ks = np.arange(i + 1, j)
        sc = chart[i, ks][:, self.L[s:e]] + chart[ks, j][:, self.R[s:e]] + self.logp[s:e]
        k_i, r_i = np.unravel_index(int(np.argmax(sc)), sc.shape)
        k, r = int(ks[k_i]), s + int(r_i)
        return Tree(self.symbols[a], [self._build(chart, words, i, k, int(self.L[r])),
                                      self._build(chart, words, k, j, int(self.R[r]))])

    def fallback(self, words):
        """Flat tree used only if CKY finds no complete parse."""
        kids = []
        for w in words:
            v = self.lex_vec(w)
            kids.append(Tree(self.symbols[int(np.argmax(v))].split("+")[-1], [w]))
        return Tree(TOP, [Tree("S", kids)])
