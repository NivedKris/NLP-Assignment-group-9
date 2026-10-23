"""Collects every metrics JSON in results/ and writes results/RESULTS.md and REPORT.md
(the final report, with all numbers filled in from the runs). Also plots Task 2/3 curves."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)  # show progress live, also when piped to tee
from common.ptb import PARTITION, ROOT_DIR, load_splits

R = os.path.join(ROOT_DIR, "results")


def load(p):
    p = os.path.join(R, p)
    if not os.path.exists(p):
        sys.exit(f"missing {p} - run the earlier steps first")
    with open(p) as f:
        return json.load(f)


t1 = {h: load(f"task1/task1_h{h}_test_metrics.json") for h in (0, 1)}
t2 = load("task2/task2_test_metrics.json")
t2a = load("task2_rule2layer/task2_test_metrics.json")  # ablation: 2-layer ReLU binary-rule MLP
t3 = {h: load(f"task3/task3_h{h}_test_metrics.json") for h in (0, 1)}
sp = load_splits(PARTITION)
f = lambda x: f"{x:.2f}"

# ------------------------------------------------------------------ plots (optional)
plots = []
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    hist = t2["history"]
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(ep, [h["train_elbo_ppl"] for h in hist], marker="o", label="train (ELBO)")
    ax[0].plot(ep, [h["val"]["elbo_ppl"] for h in hist], marker="o", label="validation (ELBO)")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("perplexity bound"); ax[0].legend(); ax[0].set_yscale("log")
    ax[1].plot(ep, [h["val"]["corpus_f1"] for h in hist], marker="o", label="val corpus F1")
    ax[1].plot(ep, [h["val"]["sentence_f1"] for h in hist], marker="o", label="val sentence F1")
    ha = t2a["history"]
    ax[1].plot([h["epoch"] for h in ha], [h["val"]["corpus_f1"] for h in ha], "--", color="gray",
               label="val corpus F1, 2-layer rule MLP (ablation)")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("unlabeled F1"); ax[1].legend(loc="center right", fontsize=8)
    for a in ax:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle("Task 2: Compound PCFG training (Partition C)")
    fig.tight_layout(); fig.savefig(os.path.join(R, "task2_training_curves.png"), dpi=130)
    plots.append("results/task2_training_curves.png")
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for h in (0, 1):
        hh = t3[h]["history"]
        ax.plot([x["epoch"] for x in hh], [x["val_nll_per_word"] for x in hh], label=f"val, h={h}")
        ax.plot([x["epoch"] for x in hh], [x["train_nll_per_word"] for x in hh], "--", label=f"train, h={h}")
    ax.set_xlabel("epoch"); ax.set_ylabel("NLL per word (gold trees)"); ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Task 3: supervised neural PCFG")
    fig.tight_layout(); fig.savefig(os.path.join(R, "task3_training_curves.png"), dpi=130)
    plots.append("results/task3_training_curves.png")
except Exception as e:  # matplotlib missing or headless problems: report still gets written
    print("plots skipped:", e)

# ------------------------------------------------------------------ tables
rows = []
for name, r in [("Task 1: count PCFG + CKY, h=0 (main)", t1[0]), ("Task 1: count PCFG + CKY, h=1 (ablation)", t1[1]),
                ("Task 3: supervised neural PCFG, h=0", t3[0]), ("Task 3: supervised neural PCFG, h=1", t3[1])]:
    e, c = r["evalb"], r["compare"]
    rows.append(f"| {name} | {f(e['labeled']['precision'])} | {f(e['labeled']['recall'])} | **{f(e['labeled']['f1'])}** "
                f"| {f(e['unlabeled']['f1'])} | {f(e['tagging_accuracy'])} | {f(c['corpus_f1'])} | {f(c['sentence_f1'])} |")
tt = t2["test"]
rows.append(f"| Task 2: compound PCFG (unsupervised) | - | - | - | - | - | {f(tt['corpus_f1'])} | {f(tt['sentence_f1'])} |")
ta = t2a["test"]
rows.append(f"| Task 2 ablation: 2-layer ReLU binary-rule MLP | - | - | - | - | - | {f(ta['corpus_f1'])} | {f(ta['sentence_f1'])} |")
rb, lb = t2["baselines"]["right_branching"], t2["baselines"]["left_branching"]
rows.append(f"| Baseline: right-branching trees | - | - | - | - | - | {f(rb['corpus_f1'])} | {f(rb['sentence_f1'])} |")
rows.append(f"| Baseline: left-branching trees | - | - | - | - | - | {f(lb['corpus_f1'])} | {f(lb['sentence_f1'])} |")
main_table = ("| System | Labeled P | Labeled R | **Labeled F1** | Unlabeled F1 | POS acc | Compare corpus F1 | Compare sentence F1 |\n"
              "|---|---|---|---|---|---|---|---|\n" + "\n".join(rows))

with open(os.path.join(R, "RESULTS.md"), "w") as fh:
    fh.write(f"# Results - Partition {PARTITION} test split ({len(sp['test'])} sentences)\n\n{main_table}\n")

# ------------------------------------------------------------------ narrative helpers
from nltk.tree import Tree
from common.metrics import CompareScorer, left_branching_spans, span_set


def lb_agreement(pred_file):
    """corpus F1 between a model's predicted trees and left-branching trees over the same words"""
    sc = CompareScorer()
    with open(os.path.join(R, pred_file)) as fh:
        for line in fh:
            t = Tree.fromstring(line)
            sc.add_sets(span_set(t), left_branching_spans(len(t.leaves())))
    return sc.summary()["corpus_f1"]


lb_agree = lb_agreement("task2_rule2layer/task2_test_pred.txt")
lb_agree_main = lb_agreement("task2/task2_test_pred.txt")
b1 = max((0, 1), key=lambda h: t1[h]["evalb"]["labeled"]["f1"])
b3 = max((0, 1), key=lambda h: t3[h]["evalb"]["labeled"]["f1"])
d_mark = t1[1]["evalb"]["labeled"]["f1"] - t1[0]["evalb"]["labeled"]["f1"]
d31 = {h: t3[h]["evalb"]["labeled"]["f1"] - t1[h]["evalb"]["labeled"]["f1"] for h in (0, 1)}
d31u = {h: t3[h]["evalb"]["unlabeled"]["f1"] - t1[h]["evalb"]["unlabeled"]["f1"] for h in (0, 1)}
best3, best1 = t3[b3]["evalb"]["labeled"]["f1"], t1[b1]["evalb"]["labeled"]["f1"]
if best3 - best1 >= 1.0:
    verdict = "**Yes** - the neurally-parameterised supervised grammar beats the best count-based grammar"
elif best3 > best1:
    verdict = ("**Only marginally** - the best neural grammar is ahead of the best count-based grammar, but by less "
               "than 1 labeled-F1 point, which is within the noise of a 131-sentence test set")
else:
    verdict = "**No** - the neurally-parameterised supervised grammar does not beat the best count-based grammar"
sign = lambda x: f"+{x:.2f}" if x >= 0 else f"{x:.2f}"
gap_rb = tt["sentence_f1"] - rb["sentence_f1"]


def gstat(r, k):
    return r["grammar"][k]


report = f"""# NLP Assignment - Project 3: From CKY to Neural Grammar Induction

**Group 9 - Penn Treebank, Partition C**

| Member (institute e-mail) |
|---|
| nivedkrishna26@kgpian.iitkgp.ac.in (Nived Krishna) |
| srimangarlapati@kgpian.iitkgp.ac.in |
| mmodi1610@kgpian.iitkgp.ac.in |

All numbers in this report are produced by the code in this folder (the commands in Section 8 regenerate
them) and are read automatically from `results/**/*.json`.

## 1. Data (Dataset 3.3 - Partition C)

* Source: `nltk.corpus.treebank.parsed_sents()` - the 3,914-sentence NLTK sample of the Penn Treebank.
* Construction (`common/ptb.py`, `scripts/00_make_splits.py`): `numpy.random.seed(42)` then one permutation
  of all 3,914 indices, cut into thirds: A = `[0:1305]`, B = `[1305:2610]`, **C = `[2610:3914]` (1,304 sentences)**.
  Inside Partition C a second, independent permutation with the same seed (42) is split 80/10/10:
  **train {len(sp['train'])} / validation {len(sp['validation'])} / test {len(sp['test'])}** sentences.
  Every index list is saved in `data/splits/partition_splits_seed42.json` and reused by all tasks;
  Partitions A and B are never read.
* Preprocessing: functional tags and trace indices are stripped (`NP-SBJ-1` -> `NP`, `NP=2` -> `NP`), empty
  elements (`-NONE-` traces) and constituents that become empty are deleted. Cleaned trees are in `data/partition_C/`.

## 2. Evaluation protocol

* **Labeled / unlabeled bracketing F1 (PARSEVAL, EVALB-style)** - `common/metrics.py:EvalbScorer`. Mirrors
  EVALB with `COLLINS.prm`: punctuation tokens (gold tags comma, colon, period and both quote tags) are deleted, POS nodes are not
  brackets, the TOP bracket is ignored, ADVP = PRT, brackets matched as multisets, corpus-level P/R/F1.
  This is the Task 1 / Task 3 metric.
* **"Compare" unlabeled F1 (compound-PCFG protocol)** - `common/metrics.py:CompareScorer`, a line-by-line
  reimplementation of `harvardnlp/compound-pcfg/eval.py`: punctuation and other non-word tokens removed as in
  `process_ptb.py`, spans of length >= 2, sentence-level span dropped, corpus F1 and PRPN-style sentence F1,
  length-1 sentences skipped. This is the only metric that applies to Task 2 (its labels are anonymous
  clusters), so all three tasks are also scored with it to make the comparison fair.

## 3. Results (Partition C test split, {len(sp['test'])} sentences)

{main_table}

## 4. Task 1 - Baseline: CKY with a count-based PCFG

**Method** (`scripts/task1_cky_pcfg.py`, `common/pcfg.py`).
1. Training trees are wrapped in `TOP`, unary chains are collapsed (`collapse_unary(collapsePOS=True)`, labels
   joined with `+`), then binarized with NLTK `chomsky_normal_form(factor="right", horzMarkov=h, vertMarkov=0)`,
   with **h = 0 (main)** and **h = 1 (ablation)**. Round-trip `CNF -> un-CNF` is asserted to reproduce the
   original trees.
2. Rule probabilities are relative frequencies P(A -> beta) = c'(A -> beta) / sum_beta' c'(A -> beta'), where
   **c' = c + 1 for every rule seen fewer than 5 times** and c' = c otherwise (binary, lexical and TOP rules).
   Training words seen once are replaced by a Berkeley-style unknown-word signature (e.g. `UNK-CAPS-s`), so
   unknown test words get a distribution; unseen signatures back off to the pooled class `UNK*`.
3. Exact Viterbi CKY over the CNF grammar, O(n^3 |G|): for each span and split point all binary rules are
   scored with vectorised NumPy, and back-pointers are recovered by re-maximisation. If no complete parse
   exists the sentence is re-parsed once with a lexical back-off that lets every word also take any
   unknown-word tag (probability x 1e-3). The CNF output is un-binarized and unary chains re-expanded before scoring.
4. Labeled and unlabeled bracketing F1 on the test split.

| | h = 0 | h = 1 |
|---|---|---|
| symbols | {gstat(t1[0],'symbols')} | {gstat(t1[1],'symbols')} |
| binary rules | {gstat(t1[0],'binary')} | {gstat(t1[1],'binary')} |
| rule types with add-1 (count < 5) / total | {gstat(t1[0],'rule_types_add1_smoothed')} / {gstat(t1[0],'rule_types_total')} | {gstat(t1[1],'rule_types_add1_smoothed')} / {gstat(t1[1],'rule_types_total')} |
| test sentences parsed directly / with back-off / flat fallback | {t1[0]['parse_status'].get('ok',0)} / {t1[0]['parse_status'].get('backoff',0)} / {t1[0]['parse_status'].get('fallback',0)} | {t1[1]['parse_status'].get('ok',0)} / {t1[1]['parse_status'].get('backoff',0)} / {t1[1]['parse_status'].get('fallback',0)} |
| **labeled F1** | **{f(t1[0]['evalb']['labeled']['f1'])}** | **{f(t1[1]['evalb']['labeled']['f1'])}** |
| unlabeled F1 | {f(t1[0]['evalb']['unlabeled']['f1'])} | {f(t1[1]['evalb']['unlabeled']['f1'])} |
| labeled F1, sentences <= 40 words | {f(t1[0]['evalb']['len_le_40']['labeled']['f1'])} | {f(t1[1]['evalb']['len_le_40']['labeled']['f1'])} |

**Markovization ablation.** Moving from 0th- to 1st-order horizontal markovization changes labeled F1 by
{sign(d_mark)} points. h = 0 forgets every sibling inside a binarized constituent (one intermediate symbol
per parent, e.g. `NP|<>`), which over-generalises: any child sequence is possible. h = 1 remembers the previous
sibling (`NP|<DT>`), so the grammar has {gstat(t1[1],'symbols')} instead of {gstat(t1[0],'symbols')} symbols, models
local ordering better, but its rules are sparser on ~1k training trees (more sentences need the back-off retry).

## 5. Task 2 - Paper reproduction: Compound PCFG (unsupervised)

**Method** (`scripts/task2_compound_pcfg.py`, `common/compound_pcfg.py`). A device-agnostic port of
`harvardnlp/compound-pcfg` (`models.py`, `PCFG.py`, `train.py`, `eval.py`), with the inside algorithm
vectorised over all spans of the same width (same result as the official loop version, much faster on CPU).
The model never sees a tree: training uses only the words of the {t2['n_train_sents']} Partition C training sentences
(punctuation removed, lower-cased, digits -> `N`, length-1 sentences skipped, as in the official preprocessing).

| setting | value |
|---|---|
| nonterminals / preterminals | {t2['args']['nt_states']} / {t2['args']['t_states']} |
| latent z | R^{t2['args']['z_dim']}, q(z given x) = N(mu, sigma^2) from a 1-layer BiLSTM (hidden {t2['args']['h_dim']}) over word embeddings of dim {t2['args']['w_dim']}, max-pooled |
| rule probabilities | all conditioned on [symbol embedding (256); z]. Root rules S -> A and emissions T -> w: MLP Linear -> 2 residual ReLU layers (hidden 256) -> Linear. Binary rules A -> B C: one linear layer -> softmax over (NT+T)^2 child pairs (official `models.py`) |
| objective | ELBO = E_q[log p(x given z)] minus the KL divergence between q(z given x) and N(0, I); log p(x given z) by the inside algorithm |
| optimiser | Adam, lr {t2['args']['lr']}, betas ({t2['args']['beta1']}, {t2['args']['beta2']}), grad-norm clip {t2['args']['max_grad_norm']} |
| batch / epochs | {t2['args']['batch_size']} sentences of equal length / {t2['args']['epochs']} epochs; curriculum max length {t2['args']['max_length']} -> {t2['args']['final_max_length']} (+1 per epoch) |
| model selection | validation ELBO perplexity (best epoch: {t2['best_epoch']}) |
| decoding | Viterbi (max-probability tree) with z = posterior mean, as in `eval.py` |
| vocabulary | {t2['vocab']} types (training words seen >= 2 times; rest `<unk>`) |

**Results.** Test corpus F1 **{f(tt['corpus_f1'])}**, sentence F1 **{f(tt['sentence_f1'])}** (right-branching:
{f(rb['corpus_f1'])} / {f(rb['sentence_f1'])}, left-branching: {f(lb['corpus_f1'])} / {f(lb['sentence_f1'])}; the induced grammar is at
{sign(gap_rb)} sentence-F1 points relative to right-branching). For reference, Kim et al. (2019) report sentence F1 55.2 (mean) / 60.1 (max) for the
compound PCFG, 50.8 for the neural PCFG and 39.5 for right-branching on the full PTB test set, training on
~40k sentences of WSJ 02-21. Our grammar is trained on {t2['n_train_sents']} sentences - about 40x less raw text -
which is the expected, intentional weakness of the one-partition-per-group design: the grammar has to induce
constituency from co-occurrence statistics alone, and with ~1k sentences most word types are seen only a
handful of times. {"Training curves: `results/task2_training_curves.png`." if plots else ""}

**Ablation: a 2-layer ReLU MLP for the binary rules.** The assignment describes the rule network as a
"2-layer MLP (hidden 256, ReLU)". In the official code this describes the root and emission networks; the
binary-rule network is a single linear layer. We also trained the literal variant, where the binary rules use
Linear(320, 256) -> ReLU -> Linear (`--rule_layers 2`, everything else identical). It reaches test corpus F1
**{f(ta['corpus_f1'])}** / sentence F1 **{f(ta['sentence_f1'])}** (best epoch {t2a['best_epoch']}). Its trees agree with purely
left-branching trees at {f(lb_agree)} corpus F1 (main model: {f(lb_agree_main)}): on this small corpus it collapses to a largely
left-branching grammar after a few epochs (see the dashed curve in the training plot), while the official parameterisation
keeps improving. This is why the main run uses the official single-layer binary-rule network.

**Other differences from the official code (all deliberate).** (i) the inference-network LSTM and word
embeddings are 256-dimensional as the assignment specifies (official default 512); (ii) {t2['args']['epochs']} epochs (assignment 15-20;
the paper uses 10 epochs on full PTB); (iii) on ~1k sentences a 10k-word cap keeps every word, so `<unk>` would never be
trained - words seen once are mapped to `<unk>` instead; (iv) inside algorithm vectorised and device-agnostic (verified
against brute-force enumeration in `tests/test_compound_pcfg.py`); (v) checkpoints store a `state_dict`.

## 6. Task 3 - Combine & improve: supervised neural PCFG

**Method** (`scripts/task3_neural_pcfg.py`). Task 2's parameterisation with the latent z and the BiLSTM
inference network removed: every grammar symbol has a 256-d embedding; binary rule probabilities
P(A -> B C) = softmax over the inventory of child pairs (B, C) of the same single linear layer as Task 2's main run,
applied to the embedding of A; emissions P(T -> w) and root probabilities P(TOP -> X) use the same residual ReLU
MLPs (hidden 256) as Task 2.
The symbols are the **real treebank symbols of Task 1's CNF trees** (h = 0 and h = 1), not 30/60 anonymous
states, so labeled F1 is defined and the comparison with Task 1 isolates *how* rule probabilities are
parameterised (counts vs. a shared neural function).

*Training.* Supervised MLE on the gold training trees: with fully observed trees the expected rule counts
that inside-outside would compute are exactly the rule counts of the gold tree, so the inside-outside MLE
objective reduces to maximising sum log P(rule) over all rules in the gold CNF trees (plus root and emission
terms). Adam (lr 1e-3, betas 0.75/0.999, clip 3), batches of {t3[0]['args']['batch_size']} trees, {t3[0]['args']['epochs']} epochs,
model selection on validation negative log-likelihood per word.

*Decoding.* The neural grammar is exported to the same tables as Task 1 and parsed with the same exact CKY.
Unlike counts, the network gives non-zero probability to child pairs never seen with a parent (generalisation
through shared embeddings); to keep CKY tractable binary rules with probability < {t3[0]['args']['prune']} are pruned unless
the rule was seen in training ({gstat(t3[0],'binary')} binary rules for h = 0 vs. {gstat(t3[0],'seen_binary_rules')} seen; {gstat(t3[1],'binary')} vs. {gstat(t3[1],'seen_binary_rules')} for h = 1).
Emissions are dense: every preterminal can emit every word with a learned probability.

| | Task 1 (counts) h=0 | Task 3 (neural) h=0 | Task 1 (counts) h=1 | Task 3 (neural) h=1 |
|---|---|---|---|---|
| labeled F1 | {f(t1[0]['evalb']['labeled']['f1'])} | {f(t3[0]['evalb']['labeled']['f1'])} ({sign(d31[0])}) | {f(t1[1]['evalb']['labeled']['f1'])} | {f(t3[1]['evalb']['labeled']['f1'])} ({sign(d31[1])}) |
| unlabeled F1 | {f(t1[0]['evalb']['unlabeled']['f1'])} | {f(t3[0]['evalb']['unlabeled']['f1'])} ({sign(d31u[0])}) | {f(t1[1]['evalb']['unlabeled']['f1'])} | {f(t3[1]['evalb']['unlabeled']['f1'])} ({sign(d31u[1])}) |
| POS accuracy | {f(t1[0]['evalb']['tagging_accuracy'])} | {f(t3[0]['evalb']['tagging_accuracy'])} | {f(t1[1]['evalb']['tagging_accuracy'])} | {f(t3[1]['evalb']['tagging_accuracy'])} |
| compare corpus F1 | {f(t1[0]['compare']['corpus_f1'])} | {f(t3[0]['compare']['corpus_f1'])} | {f(t1[1]['compare']['corpus_f1'])} | {f(t3[1]['compare']['corpus_f1'])} |

**Does a neurally-parameterised, fully-supervised grammar beat the classical count-based one?** {verdict}
(best Task 3 labeled F1 {f(best3)} with h = {b3} vs. best Task 1 labeled F1 {f(best1)} with h = {b1}).
Both are far above the unsupervised Task 2 grammar on the shared unlabeled metric
({f(max(t3[0]['compare']['corpus_f1'], t3[1]['compare']['corpus_f1']))} and {f(max(t1[0]['compare']['corpus_f1'], t1[1]['compare']['corpus_f1']))} vs. {f(tt['corpus_f1'])} corpus F1),
which quantifies how much the gold trees are worth on a corpus this small.

## 7. Discussion

* **Supervision.** On the shared unlabeled metric the supervised grammars reach {f(max(t1[0]['compare']['corpus_f1'], t1[1]['compare']['corpus_f1'], t3[0]['compare']['corpus_f1'], t3[1]['compare']['corpus_f1']))}
  corpus F1 against {f(tt['corpus_f1'])} for the grammar induced from the same ~1k sentences without trees (Task 2);
  right-branching trees score {f(rb['corpus_f1'])}, the standard baseline for English unsupervised parsing.
* **Parameterisation.** Counts are the exact MLE for the rules they have seen and give nothing to rules they
  have not; the neural grammar shares statistical strength between symbols through embeddings (similar
  symbols get similar rule distributions) and smooths both rules and emissions. Measured effect of switching
  counts -> neural: {sign(d31[0])} labeled F1 with h = 0 and {sign(d31[1])} with h = 1.
* **Markovization.** 1st-order horizontal markovization adds sibling context at the cost of sparser rules;
  measured effect of h = 0 -> h = 1: {sign(d_mark)} labeled F1 for counts and
  {sign(t3[1]['evalb']['labeled']['f1'] - t3[0]['evalb']['labeled']['f1'])} for the neural grammar.
* **Limitations.** Single run per model (no seed variance on this budget); a small test set ({len(sp['test'])} sentences)
  means differences of ~1 F1 point are within noise; CKY is exact but pure NumPy, so long sentences are slow.

## 8. Reproducing

```
pip install -r requirements.txt
python scripts/00_make_splits.py
python scripts/task1_cky_pcfg.py --horz 0 ; python scripts/task1_cky_pcfg.py --horz 1
python scripts/task2_compound_pcfg.py
python scripts/task2_compound_pcfg.py --rule_layers 2 --tag rule2layer
python scripts/task3_neural_pcfg.py --horz 0 ; python scripts/task3_neural_pcfg.py --horz 1
python scripts/make_report.py
```

## References

* Y. Kim, C. Dyer, A. M. Rush. *Compound Probabilistic Context-Free Grammars for Grammar Induction.* ACL 2019, arXiv:1906.10225. Code: https://github.com/harvardnlp/compound-pcfg
* M. Marcus, B. Santorini, M. A. Marcinkiewicz. *Building a Large Annotated Corpus of English: The Penn Treebank.* Computational Linguistics 19(2), 1993.
* S. Bird, E. Klein, E. Loper. *Natural Language Processing with Python* (NLTK), 2009.
"""
with open(os.path.join(ROOT_DIR, "REPORT.md"), "w") as fh:
    fh.write(report)
print("wrote REPORT.md and results/RESULTS.md")
print(main_table)
