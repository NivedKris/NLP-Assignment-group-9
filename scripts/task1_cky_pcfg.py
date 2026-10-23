"""Task 1: treebank PCFG (relative-frequency MLE, add-1 on rules seen < 5 times) + CKY.
Usage: python scripts/task1_cky_pcfg.py --horz 0      (main run, 0th-order markovization)
       python scripts/task1_cky_pcfg.py --horz 1      (ablation, 1st-order markovization)"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)  # show progress live, also when piped to tee
from common.metrics import CompareScorer, EvalbScorer
from common.pcfg import Grammar, WordMapper, extract_rules, from_cnf, to_cnf
from common.ptb import PARTITION, ROOT_DIR, load_partition, tree_to_line

ap = argparse.ArgumentParser()
ap.add_argument("--horz", type=int, default=0, choices=[0, 1])
ap.add_argument("--smooth_below", type=int, default=5, help="add-1 to rules seen fewer than this")
ap.add_argument("--min_word_count", type=int, default=2, help="rarer training words -> UNK signature")
ap.add_argument("--split", default="test", choices=["test", "validation"])
args = ap.parse_args()

out_dir = os.path.join(ROOT_DIR, "results", "task1")
os.makedirs(out_dir, exist_ok=True)
tag = f"h{args.horz}"
data = load_partition(PARTITION)
train = [t for _, t in data["train"]]
evals = [t for _, t in data[args.split]]

# ---- 1. CNF conversion
cnf_train = [to_cnf(t, args.horz) for t in train]
for t, c in zip(train[:50], cnf_train[:50]):  # round-trip sanity check
    assert from_cnf(c) == t

# ---- 2. counts (rare words -> signatures)
mapper = WordMapper([w for t in train for w in t.leaves()], min_count=args.min_word_count)
cnt = defaultdict(Counter)   # lhs -> Counter(rhs)
for c in cnf_train:
    for r in extract_rules(c):
        if r[0] == "root":
            cnt["TOP"][("root", r[1])] += 1
        elif r[0] == "bin":
            cnt[r[1]][("bin", r[2], r[3])] += 1
        else:
            cnt[r[1]][("lex", mapper.train_key(r[2]))] += 1

symbols = sorted({lhs for lhs in cnt if lhs != "TOP"} |
                 {rhs[1] for rhs in cnt["TOP"]} |
                 {x for lhs in cnt for rhs in cnt[lhs] if rhs[0] == "bin" for x in rhs[1:]})
sid = {s: i for i, s in enumerate(symbols)}


def smoothed(c):  # add-1 smoothing on any rule seen fewer than `smooth_below` times
    return c + 1 if c < args.smooth_below else c


binary, lex_counts = [], defaultdict(dict)
root = np.full(len(symbols), -np.inf)
n_smoothed = n_total = 0
for lhs, rhs_counter in cnt.items():
    denom = sum(smoothed(c) for c in rhs_counter.values())
    for rhs, c in rhs_counter.items():
        n_total += 1
        n_smoothed += int(c < args.smooth_below)
        lp = np.log(smoothed(c) / denom)
        if rhs[0] == "root":
            root[sid[rhs[1]]] = lp
        elif rhs[0] == "bin":
            binary.append((sid[lhs], sid[rhs[1]], sid[rhs[2]], lp))
        else:
            lex_counts[rhs[1]][sid[lhs]] = lp

lexicon = {}
for word, d in lex_counts.items():
    v = np.full(len(symbols), -np.inf)
    for a, lp in d.items():
        v[a] = lp
    lexicon[word] = v
# generic unknown class: P(A -> UNK*) = log-sum of A's probability over all signature classes
sig_keys = [k for k in lexicon if k.startswith("UNK")]
with np.errstate(divide="ignore"):
    lexicon[WordMapper.GENERIC] = np.log(np.sum([np.exp(lexicon[k]) for k in sig_keys], axis=0))

G = Grammar(symbols, binary, lexicon, root, mapper)
stats = G.n_rules()
stats.update({"rule_types_total": n_total, "rule_types_add1_smoothed": n_smoothed,
              "train_trees": len(train), "known_words": len(mapper.known)})
print(f"[task1 {tag}] grammar:", stats)

# ---- 3./4. CKY parse + evaluate
evalb, comp = EvalbScorer(), CompareScorer()
status, t0 = Counter(), time.time()
pred_lines = []
for k, gold in enumerate(evals):
    words = gold.leaves()
    cnf, st = G.parse_robust(words)
    status[st] += 1
    pred = from_cnf(cnf)
    evalb.add(gold, pred)
    comp.add_trees(gold, pred)
    pred_lines.append(tree_to_line(pred))
    if (k + 1) % 20 == 0:
        print(f"  parsed {k+1}/{len(evals)}  ({time.time()-t0:.0f}s)", flush=True)

res = {"task": "task1", "markovization": args.horz, "split": args.split,
       "grammar": stats, "parse_status": dict(status), "parse_seconds": time.time() - t0,
       "evalb": evalb.summary(), "compare": comp.summary()}
with open(os.path.join(out_dir, f"task1_{tag}_{args.split}_pred.txt"), "w") as f:
    f.write("\n".join(pred_lines) + "\n")
with open(os.path.join(out_dir, f"task1_{tag}_{args.split}_gold.txt"), "w") as f:
    f.write("\n".join(tree_to_line(t) for t in evals) + "\n")
with open(os.path.join(out_dir, f"task1_{tag}_{args.split}_metrics.json"), "w") as f:
    json.dump(res, f, indent=2)
e = res["evalb"]
print(f"[task1 {tag}] {args.split}: labeled F1 {e['labeled']['f1']:.2f} | unlabeled F1 "
      f"{e['unlabeled']['f1']:.2f} | tag acc {e['tagging_accuracy']:.2f} | "
      f"compare corpus F1 {res['compare']['corpus_f1']:.2f} | parse status {dict(status)}")
