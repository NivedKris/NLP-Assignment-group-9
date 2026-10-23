"""Task 2: unsupervised Compound PCFG on Partition C training *sentences only*, evaluated
against Partition C gold test trees (unlabeled bracketing F1, compound-pcfg protocol).

Data processing mirrors harvardnlp/compound-pcfg (process_ptb.py + preprocess.py):
  punctuation / non-word tokens removed, lowercased, numbers -> 'N', length-1 sentences skipped.
Training mirrors train.py: Adam(lr 1e-3, betas 0.75/0.999), grad-clip 3, xavier init,
  length curriculum 30 -> 40 (+1 per epoch), same-length batches, model selection on
  validation ELBO perplexity. Decoding mirrors eval.py: Viterbi with z = posterior mean.
Usage: python scripts/task2_compound_pcfg.py                                  (main run)
       python scripts/task2_compound_pcfg.py --rule_layers 2 --tag rule2layer (ablation)"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)  # show progress live, also when piped to tee
from common.compound_pcfg import CompPCFG
from common.metrics import (CompareScorer, left_branching_spans, right_branching_spans,
                            span_set, strip_for_compare)
from common.ptb import PARTITION, ROOT_DIR, load_partition, tree_to_line

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=20)            # assignment: 15-20
ap.add_argument("--batch_size", type=int, default=4)         # assignment: 4-8
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--beta1", type=float, default=0.75)
ap.add_argument("--beta2", type=float, default=0.999)
ap.add_argument("--max_grad_norm", type=float, default=3.0)
ap.add_argument("--max_length", type=int, default=30)        # curriculum start
ap.add_argument("--len_incr", type=int, default=1)
ap.add_argument("--final_max_length", type=int, default=40)
ap.add_argument("--nt_states", type=int, default=30)
ap.add_argument("--t_states", type=int, default=60)
ap.add_argument("--z_dim", type=int, default=64)
ap.add_argument("--w_dim", type=int, default=256)
ap.add_argument("--h_dim", type=int, default=256)
ap.add_argument("--state_dim", type=int, default=256)
ap.add_argument("--rule_hidden", type=int, default=256)
ap.add_argument("--rule_layers", type=int, default=1, choices=[1, 2],
                help="binary-rule network: 1 = official code (main run), 2 = 2-layer ReLU MLP (ablation)")
ap.add_argument("--tag", default="", help="suffix for output files (extra runs)")
ap.add_argument("--vocab_min_count", type=int, default=2, help="rarer train words -> <unk>")
ap.add_argument("--seed", type=int, default=3435)            # official code's seed
ap.add_argument("--device", default="auto")
ap.add_argument("--print_every", type=int, default=50)
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" \
    else torch.device(args.device)
np.random.seed(args.seed); torch.manual_seed(args.seed)
out_dir = os.path.join(ROOT_DIR, "results", "task2" + (f"_{args.tag}" if args.tag else ""))
os.makedirs(out_dir, exist_ok=True)


def clean_number(w):  # identical to compound-pcfg preprocess.py / eval.py
    return re.sub("[0-9]{1,}([,.]?[0-9]*)*", "N", w)


def prep(items):
    """-> list of (words, gold_stripped_tree); trees with < 2 words after stripping are dropped."""
    out = []
    for _, t in items:
        g = strip_for_compare(t)
        if g is None or len(g.leaves()) < 2:
            continue
        out.append(([clean_number(w.lower()) for w in g.leaves()], g))
    return out


data = load_partition(PARTITION)
train, val, test = prep(data["train"]), prep(data["validation"]), prep(data["test"])
# gold trees are NOT used for training: only `words` of train are ever given to the model
counts = Counter(w for ws, _ in train for w in ws)
itos = ["<pad>", "<unk>", "<s>", "</s>"] + sorted(w for w, c in counts.items() if c >= args.vocab_min_count)
stoi = {w: i for i, w in enumerate(itos)}
enc = lambda ws: [stoi.get(w, 1) for w in ws]
print(f"[task2] device={device} train={len(train)} val={len(val)} test={len(test)} vocab={len(itos)}")


def make_batches(sents, bs):
    by_len = {}
    for i, (ws, _) in enumerate(sents):
        by_len.setdefault(len(ws), []).append(i)
    batches = []
    for L in sorted(by_len):
        ids = by_len[L]
        for s in range(0, len(ids), bs):
            batches.append(ids[s:s + bs])
    return batches


train_batches, val_batches = make_batches(train, args.batch_size), make_batches(val, args.batch_size)
model = CompPCFG(len(itos), h_dim=args.h_dim, w_dim=args.w_dim, z_dim=args.z_dim,
                 state_dim=args.state_dim, t_states=args.t_states, nt_states=args.nt_states,
                 rule_hidden=args.rule_hidden, rule_layers=args.rule_layers)
for p in model.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)
model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))


def to_tensor(sents, ids):
    return torch.tensor([enc(sents[i][0]) for i in ids], dtype=torch.long, device=device)


def evaluate(sents, batches, use_mean):
    model.eval()
    nll = kl = words = 0.0
    sc = CompareScorer()
    with torch.no_grad():
        for ids in batches:
            x = to_tensor(sents, ids)
            n_, k_, spans, _ = model(x, argmax=True, use_mean=use_mean)
            nll += n_.sum().item(); kl += k_.sum().item(); words += len(ids) * (x.size(1) + 1)
            for bb, i in enumerate(ids):
                sc.add_sets(set((a, b) for a, b, _ in spans[bb][:-1]), span_set(sents[i][1]))
    model.train()
    s = sc.summary()
    return {"recon_ppl": float(np.exp(nll / words)), "elbo_ppl": float(np.exp((nll + kl) / words)),
            "kl_per_sent": kl / sum(len(b) for b in batches),
            "corpus_f1": s["corpus_f1"], "sentence_f1": s["sentence_f1"]}


ckpt_path = os.path.join(out_dir, "compound_pcfg_best.pt")
best_val, history, max_len = float("inf"), [], args.max_length
t_start = time.time()
for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    tr_nll = tr_kl = tr_words = 0.0
    n_b = 0
    for bi in np.random.permutation(len(train_batches)):
        ids = train_batches[bi]
        L = len(train[ids[0]][0])
        if L > max_len or L == 1:  # curriculum length filter (as in train.py)
            continue
        x = to_tensor(train, ids)
        opt.zero_grad()
        nll, kl = model(x)
        (nll + kl).mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        opt.step()
        tr_nll += nll.sum().item(); tr_kl += kl.sum().item(); tr_words += len(ids) * (L + 1)
        n_b += 1
        if n_b % args.print_every == 0:
            print(f"  epoch {epoch} batch {n_b} recon_ppl {np.exp(tr_nll/tr_words):.1f} "
                  f"kl {tr_kl/max(1, tr_words):.4f} ({time.time()-t0:.0f}s)", flush=True)
    v = evaluate(val, val_batches, use_mean=False)  # sampled z: valid upper bound for selection
    rec = {"epoch": epoch, "max_len": max_len, "train_recon_ppl": float(np.exp(tr_nll / max(tr_words, 1))),
           "train_elbo_ppl": float(np.exp((tr_nll + tr_kl) / max(tr_words, 1))),
           "val": v, "seconds": time.time() - t0}
    history.append(rec)
    saved = v["elbo_ppl"] < best_val
    if saved:
        best_val = v["elbo_ppl"]
        torch.save({"state_dict": model.state_dict(), "itos": itos, "args": vars(args),
                    "epoch": epoch}, ckpt_path)
    print(f"[task2] epoch {epoch}/{args.epochs} max_len {max_len} train_ppl {rec['train_elbo_ppl']:.1f} "
          f"val_ppl {v['elbo_ppl']:.1f} val_corpusF1 {v['corpus_f1']:.2f} "
          f"val_sentF1 {v['sentence_f1']:.2f} {'*saved*' if saved else ''} ({rec['seconds']:.0f}s)",
          flush=True)
    max_len = min(args.final_max_length, max_len + args.len_incr)

# ---------------------------------------------------------------- test (eval.py protocol)
ck = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ck["state_dict"])
model.eval()
sc, rb, lb = CompareScorer(), CompareScorer(), CompareScorer()
nll_t = kl_t = words_t = 0.0
pred_lines, gold_lines = [], []
with torch.no_grad():
    for ws, gold in test:
        x = torch.tensor([enc(ws)], dtype=torch.long, device=device)
        n_, k_, spans, tags = model(x, argmax=True, use_mean=True)
        nll_t += n_.item(); kl_t += k_.item(); words_t += len(ws) + 1
        gset = span_set(gold)
        sc.add_sets(set((a, b) for a, b, _ in spans[0][:-1]), gset)
        rb.add_sets(right_branching_spans(len(ws)), gset)
        lb.add_sets(left_branching_spans(len(ws)), gset)
        # predicted tree string with NT-k / T-k labels (as eval.py)
        node = {i: f"(T-{tags[0][i]+1} {ws[i]})" for i in range(len(ws))}
        built = {}
        for a, b, lab in spans[0]:  # post-order => children are built before parents
            # left child = largest already-built span starting at a and ending before b
            left = max([e for (s, e) in built if s == a and e < b], default=a)
            ltree = built.get((a, left), node[a])
            rtree = built.get((left + 1, b), node[left + 1])
            built[(a, b)] = f"(NT-{lab+1} {ltree} {rtree})"
        pred_lines.append(built[(0, len(ws) - 1)])
        gold_lines.append(tree_to_line(gold))
res = {"task": "task2", "best_epoch": ck["epoch"], "train_seconds": time.time() - t_start,
       "n_train_sents": len(train), "vocab": len(itos), "args": vars(args), "history": history,
       "test": {**sc.summary(), "recon_ppl_mean_z": float(np.exp(nll_t / words_t))},
       "baselines": {"right_branching": rb.summary(), "left_branching": lb.summary()}}
with open(os.path.join(out_dir, "task2_test_pred.txt"), "w") as f:
    f.write("\n".join(pred_lines) + "\n")
with open(os.path.join(out_dir, "task2_test_gold.txt"), "w") as f:
    f.write("\n".join(gold_lines) + "\n")
with open(os.path.join(out_dir, "task2_test_metrics.json"), "w") as f:
    json.dump(res, f, indent=2)
print(f"[task2] TEST (best epoch {ck['epoch']}): corpus F1 {res['test']['corpus_f1']:.2f} | "
      f"sentence F1 {res['test']['sentence_f1']:.2f} | right-branching corpus F1 "
      f"{res['baselines']['right_branching']['corpus_f1']:.2f}")
