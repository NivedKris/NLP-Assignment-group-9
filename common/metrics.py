"""Evaluation metrics.

1) PARSEVAL / EVALB-style labeled + unlabeled bracketing F1 (Task 1 and Task 3).
   Mirrors EVALB with COLLINS.prm: words whose GOLD tag is punctuation (, : `` '' .) are
   deleted from both trees,
   the TOP bracket is ignored, ADVP == PRT, brackets are matched as multisets,
   preterminal (POS) nodes are not brackets. Corpus-level P/R/F1.

2) "Compound-PCFG" unlabeled F1 (Task 2, and the common metric used to compare
   Tasks 1/2/3). Mirrors harvardnlp/compound-pcfg exactly: words whose gold POS tag is not
   in WORD_TAGS (punctuation, $, #, ...) are removed (process_ptb.py); spans of length >= 2
   are collected in post-order and the last one (the sentence-level root span) is dropped
   (eval.py: gold_span[:-1]); corpus F1 + PRPN-style sentence F1; length-1 sentences skipped.
"""
from collections import Counter

from nltk.tree import Tree

# ------------------------------------------------------------------ EVALB style
EVALB_DELETE = {",", ":", "``", "''", ".", "-NONE-"}
EVALB_IGNORE_BRACKET = {"TOP", "ROOT", ""}
EVALB_EQ = {"PRT": "ADVP"}


def _evalb_brackets(tree, keep):
    """keep[i]: whether word i survives deletion (decided from the GOLD tags, as EVALB does).
    Returns (list of (label, start, end), list of (word, tag) kept)."""
    brackets, kept = [], []
    pos = [0]

    def rec(t):
        if isinstance(t[0], str):  # preterminal
            i = pos[0]; pos[0] += 1
            if keep[i]:
                kept.append((t[0], t.label()))
            return
        start = len(kept)
        for c in t:
            rec(c)
        end = len(kept)
        if end > start and t.label() not in EVALB_IGNORE_BRACKET:
            lab = EVALB_EQ.get(t.label(), t.label())
            brackets.append((lab, start, end))

    rec(tree)
    return brackets, kept


class EvalbScorer:
    def __init__(self):
        self.lab = [0, 0, 0]  # matched, gold, pred
        self.unl = [0, 0, 0]
        self.tags = [0, 0]  # correct, total
        self.n = 0
        self.exact = 0
        self.le40 = {"lab": [0, 0, 0], "unl": [0, 0, 0], "n": 0}

    def add(self, gold, pred):
        keep = [tag not in EVALB_DELETE for _, tag in gold.pos()]
        gb, gk = _evalb_brackets(gold, keep)
        pb, pk = _evalb_brackets(pred, keep)
        assert [w for w, _ in gk] == [w for w, _ in pk], "gold/pred word mismatch"
        gl, pl = Counter(gb), Counter(pb)
        gu, pu = Counter((s, e) for _, s, e in gb), Counter((s, e) for _, s, e in pb)
        ml, mu = sum((gl & pl).values()), sum((gu & pu).values())
        for acc, m, g, p in ((self.lab, ml, len(gb), len(pb)), (self.unl, mu, len(gb), len(pb))):
            acc[0] += m; acc[1] += g; acc[2] += p
        if len(gk) <= 40:
            self.le40["n"] += 1
            for acc, m in ((self.le40["lab"], ml), (self.le40["unl"], mu)):
                acc[0] += m; acc[1] += len(gb); acc[2] += len(pb)
        self.tags[0] += sum(1 for (_, a), (_, b) in zip(gk, pk) if a == b)
        self.tags[1] += len(gk)
        self.n += 1
        self.exact += int(ml == len(gb) == len(pb))

    @staticmethod
    def prf(acc):
        m, g, p = acc
        prec = m / p if p else 0.0
        rec = m / g if g else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {"precision": 100 * prec, "recall": 100 * rec, "f1": 100 * f1}

    def summary(self):
        return {
            "n_sentences": self.n,
            "labeled": self.prf(self.lab),
            "unlabeled": self.prf(self.unl),
            "tagging_accuracy": 100 * self.tags[0] / max(self.tags[1], 1),
            "exact_match": 100 * self.exact / max(self.n, 1),
            "len_le_40": {"n_sentences": self.le40["n"],
                          "labeled": self.prf(self.le40["lab"]),
                          "unlabeled": self.prf(self.le40["unl"])},
        }


# ------------------------------------------------------- compound-PCFG style
# process_ptb.py (harvardnlp/compound-pcfg) keeps only leaves with these tags
WORD_TAGS = {"CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN",
             "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS",
             "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
             "WDT", "WP", "WP$", "WRB"}


def prune_leaves(tree, keep):
    """Delete leaf positions where keep[i] is False; drop constituents that become empty."""
    pos = [0]

    def rec(t):
        if isinstance(t[0], str):
            i = pos[0]; pos[0] += 1
            return Tree(t.label(), [t[0]]) if keep[i] else None
        kids = [rec(c) for c in t]
        kids = [k for k in kids if k is not None]
        return Tree(t.label(), kids) if kids else None

    return rec(tree)


def gold_keep_mask(gold):
    return [tag in WORD_TAGS for _, tag in gold.pos()]


def strip_for_compare(gold):
    """Gold tree in compound-PCFG format (punctuation etc. removed), or None if empty."""
    return prune_leaves(gold, gold_keep_mask(gold))


def postorder_spans(tree):
    """Spans (i, j) inclusive, length >= 2, in the order REDUCE actions happen."""
    spans, pos = [], [0]

    def rec(t):
        if isinstance(t[0], str):
            pos[0] += 1
            return
        start = pos[0]
        for c in t:
            rec(c)
        if pos[0] - 1 > start:
            spans.append((start, pos[0] - 1))

    rec(tree)
    return spans


def span_set(tree):
    return set(postorder_spans(tree)[:-1])


class CompareScorer:
    """Unlabeled corpus F1 + PRPN sentence F1, identical to compound-pcfg eval.py."""

    def __init__(self):
        self.tp = self.fp = self.fn = 0
        self.sent_f1 = []

    def add_sets(self, pred_set, gold_set):
        self.tp += len(pred_set & gold_set)
        self.fp += len(pred_set - gold_set)
        self.fn += len(gold_set - pred_set)
        overlap = pred_set & gold_set
        prec = float(len(overlap)) / (len(pred_set) + 1e-8)
        reca = float(len(overlap)) / (len(gold_set) + 1e-8)
        if len(gold_set) == 0:
            reca = 1.0
            if len(pred_set) == 0:
                prec = 1.0
        self.sent_f1.append(2 * prec * reca / (prec + reca + 1e-8))

    def add_trees(self, gold, pred):
        """gold, pred: full trees over the same (punctuated) words. Returns False if skipped."""
        keep = gold_keep_mask(gold)
        g, p = prune_leaves(gold, keep), prune_leaves(pred, keep)
        if g is None or len(g.leaves()) <= 1:
            return False
        self.add_sets(span_set(p), span_set(g))
        return True

    def summary(self):
        prec = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        rec = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {"n_sentences": len(self.sent_f1), "corpus_f1": 100 * f1,
                "sentence_f1": 100 * sum(self.sent_f1) / max(len(self.sent_f1), 1),
                "precision": 100 * prec, "recall": 100 * rec}


def right_branching_spans(n):
    # binary right-branching tree over n words: spans (i, n-1) for i = 0..n-2, root last
    return set((i, n - 1) for i in range(1, n - 1))


def left_branching_spans(n):
    return set((0, j) for j in range(1, n - 1))
