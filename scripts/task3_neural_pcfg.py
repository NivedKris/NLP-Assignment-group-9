"""Task 3: supervised neural PCFG. Task 2's rule parameterisation (symbol embeddings -> networks ->
softmax over rules; binary rules: one linear layer as in Task 2's main run, root/emissions: residual
ReLU MLPs) with the latent z and the BiLSTM inference network removed, trained by MLE
on Partition C gold training trees (same CNF trees as Task 1), decoded with Task 1's exact CKY.

With fully observed trees, the inside-outside expected rule counts equal the observed counts in
the gold tree, so supervised inside-outside MLE = maximising sum_{rules r in gold trees} log p(r).
Usage: python scripts/task3_neural_pcfg.py --horz 0   (and --horz 1 for the markovization ablation)"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)  # show progress live, also when piped to tee
from common.compound_pcfg import ResidualLayer
from common.metrics import CompareScorer, EvalbScorer
from common.pcfg import Grammar, WordMapper, extract_rules, from_cnf, signature, to_cnf
from common.ptb import PARTITION, ROOT_DIR, load_partition, tree_to_line

ap = argparse.ArgumentParser()
ap.add_argument("--horz", type=int, default=0, choices=[0, 1])
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch_size", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--beta1", type=float, default=0.75)
ap.add_argument("--beta2", type=float, default=0.999)
ap.add_argument("--max_grad_norm", type=float, default=3.0)
ap.add_argument("--state_dim", type=int, default=256)
ap.add_argument("--rule_hidden", type=int, default=256)
ap.add_argument("--rule_layers", type=int, default=1, choices=[1, 2],
                help="binary-rule network, same meaning as in Task 2 (1 = official / Task 2 main run)")
ap.add_argument("--min_word_count", type=int, default=2)
ap.add_argument("--prune", type=float, default=1e-3,
                help="decode with binary rules p >= prune (plus every training-seen rule)")
ap.add_argument("--seed", type=int, default=3435)
ap.add_argument("--device", default="auto")
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" \
    else torch.device(args.device)
np.random.seed(args.seed); torch.manual_seed(args.seed)
tag = f"h{args.horz}"
out_dir = os.path.join(ROOT_DIR, "results", "task3")
os.makedirs(out_dir, exist_ok=True)

data = load_partition(PARTITION)
train = [t for _, t in data["train"]]
val = [t for _, t in data["validation"]]
test = [t for _, t in data["test"]]
mapper = WordMapper([w for t in train for w in t.leaves()], min_count=args.min_word_count)

# ---------------------------------------------------------------- inventories from training trees
train_rules = [extract_rules(to_cnf(t, args.horz)) for t in train]
nt_set, pt_set, pairs, words = set(), set(), set(), set()
for rs in train_rules:
    for r in rs:
        if r[0] == "bin":
            nt_set.add(r[1]); pairs.add((r[2], r[3]))
        elif r[0] == "lex":
            pt_set.add(r[1]); words.add(mapper.train_key(r[2]))
roots = {r[1] for rs in train_rules for r in rs if r[0] == "root"}
symbols = sorted(nt_set | pt_set | roots | {x for p in pairs for x in p})
sid = {s: i for i, s in enumerate(symbols)}
nts, pts = sorted(nt_set), sorted(pt_set)
nt_i = {s: i for i, s in enumerate(nts)}
pt_i = {s: i for i, s in enumerate(pts)}
pair_list = sorted(pairs)
pair_i = {p: i for i, p in enumerate(pair_list)}
vocab = sorted(words | {WordMapper.GENERIC})
w_i = {w: i for i, w in enumerate(vocab)}
both = nt_set & pt_set
print(f"[task3 {tag}] device={device} symbols={len(symbols)} NT={len(nts)} PT={len(pts)} "
      f"child-pairs={len(pair_list)} vocab={len(vocab)} NT&PT overlap={len(both)}")


class NeuralPCFG(nn.Module):
    """Task 2 architecture without z / encoder. A symbol that is both NT and PT normalises
    jointly over its binary and lexical expansions (switch probability from an extra logit)."""

    def __init__(self):
        super().__init__()
        D = args.state_dim
        self.nt_emb = nn.Parameter(torch.randn(len(nts), D))
        self.pt_emb = nn.Parameter(torch.randn(len(pts), D))
        self.root_emb = nn.Parameter(torch.randn(1, D))
        if args.rule_layers == 2:
            self.rule_mlp = nn.Sequential(nn.Linear(D, args.rule_hidden), nn.ReLU(),
                                          nn.Linear(args.rule_hidden, len(pair_list)))
        else:
            self.rule_mlp = nn.Linear(D, len(pair_list))
        self.root_mlp = nn.Sequential(nn.Linear(D, D), ResidualLayer(D, D), ResidualLayer(D, D),
                                      nn.Linear(D, len(symbols)))
        self.vocab_mlp = nn.Sequential(nn.Linear(D, D), ResidualLayer(D, D), ResidualLayer(D, D),
                                       nn.Linear(D, len(vocab)))
        # for symbols in both sets: log P(binary | A) vs log P(lexical | A)
        self.both = sorted(both)
        self.switch = nn.Parameter(torch.zeros(len(self.both))) if self.both else None

    def forward(self):
        rule = F.log_softmax(self.rule_mlp(self.nt_emb), -1)          # NT x pairs
        root = F.log_softmax(self.root_mlp(self.root_emb), -1)[0]     # symbols
        emit = F.log_softmax(self.vocab_mlp(self.pt_emb), -1)         # PT x vocab
        if self.both:
            ls = F.logsigmoid(self.switch)                            # log P(binary)
            ln = F.logsigmoid(-self.switch)
            nt_idx = torch.tensor([nt_i[s] for s in self.both], device=rule.device)
            pt_idx = torch.tensor([pt_i[s] for s in self.both], device=rule.device)
            rule = rule.index_add(0, nt_idx, ls.unsqueeze(1).expand(-1, rule.size(1)))
            emit = emit.index_add(0, pt_idx, ln.unsqueeze(1).expand(-1, emit.size(1)))
        return rule, root, emit


def index_rules(trees, wkey):
    """-> (bin idx (nt, pair), lex idx (pt, word), root idx, n_words, n_uncovered)"""
    b, l, r, nw, unc = [], [], [], 0, 0
    for t in trees:
        nw += len(t.leaves())
        for x in extract_rules(to_cnf(t, args.horz)):
            if x[0] == "root":
                if x[1] in sid:
                    r.append(sid[x[1]])
                else:
                    unc += 1
            elif x[0] == "bin":
                if x[1] in nt_i and (x[2], x[3]) in pair_i:
                    b.append((nt_i[x[1]], pair_i[(x[2], x[3])]))
                else:
                    unc += 1
            else:
                k = wkey(x[2])
                if x[1] in pt_i:
                    l.append((pt_i[x[1]], w_i.get(k, w_i[WordMapper.GENERIC])))
                else:
                    unc += 1
    return b, l, r, nw, unc


def test_key(w):
    """known word -> itself; otherwise its signature if seen in training, else generic UNK*."""
    if w in mapper.known:
        return w
    s = signature(w)
    return s if s in w_i else WordMapper.GENERIC


def nll(model, idx):
    b, l, r, nw, _ = idx
    rule, root, emit = model()
    dev = rule.device
    bt = torch.tensor(b, device=dev, dtype=torch.long).view(-1, 2)
    lt = torch.tensor(l, device=dev, dtype=torch.long).view(-1, 2)
    rt = torch.tensor(r, device=dev, dtype=torch.long)
    lp = rule[bt[:, 0], bt[:, 1]].sum() + emit[lt[:, 0], lt[:, 1]].sum() + root[rt].sum()
    return -lp, nw


model = NeuralPCFG()
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

train_idx = [index_rules([t], mapper.train_key) for t in train]
val_idx = index_rules(val, test_key)
print(f"[task3 {tag}] validation rule occurrences not covered by the inventory: {val_idx[4]}")
ckpt = os.path.join(out_dir, f"neural_pcfg_{tag}_best.pt")
best, history, t_start = float("inf"), [], time.time()
for epoch in range(1, args.epochs + 1):
    model.train()
    perm = np.random.permutation(len(train_idx))
    tot, totw = 0.0, 0
    for s in range(0, len(perm), args.batch_size):
        batch = [train_idx[i] for i in perm[s:s + args.batch_size]]
        merged = (sum((x[0] for x in batch), []), sum((x[1] for x in batch), []),
                  sum((x[2] for x in batch), []), sum(x[3] for x in batch), 0)
        opt.zero_grad()
        loss, nw = nll(model, merged)
        (loss / len(batch)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        opt.step()
        tot += loss.item(); totw += nw
    model.eval()
    with torch.no_grad():
        vl, vw = nll(model, val_idx)
    v = vl.item() / vw
    saved = v < best
    if saved:
        best = v
        torch.save(model.state_dict(), ckpt)
    history.append({"epoch": epoch, "train_nll_per_word": tot / totw, "val_nll_per_word": v})
    print(f"[task3 {tag}] epoch {epoch}/{args.epochs} train nll/word {tot/totw:.4f} "
          f"val nll/word {v:.4f} {'*saved*' if saved else ''}", flush=True)

# ---------------------------------------------------------------- neural grammar -> CKY tables
model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
model.eval()
with torch.no_grad():
    rule, root_lp, emit = [x.double().cpu().numpy() for x in model()]
seen = {(r[1], r[2], r[3]) for rs in train_rules for r in rs if r[0] == "bin"}
binary = []
logthr = np.log(args.prune)
for a, A in enumerate(nts):
    for p, (B, C) in enumerate(pair_list):
        lp = rule[a, p]
        if lp >= logthr or (A, B, C) in seen:
            binary.append((sid[A], sid[B], sid[C], lp))
lexicon = {}
for wi, w in enumerate(vocab):
    v = np.full(len(symbols), -np.inf)
    for pi, P in enumerate(pts):
        v[sid[P]] = emit[pi, wi]
    lexicon[w] = v


class NeuralMapper(WordMapper):
    """Same word -> lexicon-key mapping as used for training/validation (test_key above)."""

    def test_key(self, w):
        return test_key(w)


G = Grammar(symbols, binary, lexicon, root_lp, NeuralMapper([w for t in train for w in t.leaves()],
                                                            min_count=args.min_word_count))
stats = G.n_rules()
stats.update({"seen_binary_rules": len(seen), "child_pairs": len(pair_list), "nt": len(nts), "pt": len(pts)})
print(f"[task3 {tag}] decode grammar:", stats)

evalb, comp = EvalbScorer(), CompareScorer()
pred_lines, status, t0 = [], Counter(), time.time()
for k, gold in enumerate(test):
    words_ = gold.leaves()
    cnf, st = G.parse_robust(words_)
    status[st] += 1
    pred = from_cnf(cnf)
    evalb.add(gold, pred); comp.add_trees(gold, pred)
    pred_lines.append(tree_to_line(pred))
    if (k + 1) % 20 == 0:
        print(f"  parsed {k+1}/{len(test)} ({time.time()-t0:.0f}s)", flush=True)

res = {"task": "task3", "markovization": args.horz, "args": vars(args), "history": history,
       "best_val_nll_per_word": best, "train_seconds": t0 - t_start, "parse_seconds": time.time() - t0,
       "grammar": stats, "parse_status": dict(status), "evalb": evalb.summary(), "compare": comp.summary()}
with open(os.path.join(out_dir, f"task3_{tag}_test_pred.txt"), "w") as f:
    f.write("\n".join(pred_lines) + "\n")
with open(os.path.join(out_dir, f"task3_{tag}_test_metrics.json"), "w") as f:
    json.dump(res, f, indent=2)
e = res["evalb"]
print(f"[task3 {tag}] test: labeled F1 {e['labeled']['f1']:.2f} | unlabeled F1 {e['unlabeled']['f1']:.2f} "
      f"| tag acc {e['tagging_accuracy']:.2f} | compare corpus F1 {res['compare']['corpus_f1']:.2f} "
      f"| parse status {dict(status)}")
