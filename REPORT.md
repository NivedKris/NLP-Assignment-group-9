# NLP Assignment - Project 3: From CKY to Neural Grammar Induction

**Group 9 - Penn Treebank, Partition C**

| Member (institute e-mail) |
|---|
| nivedkrishna26@kgpian.iitkgp.ac.in (Nived Krishna)(26CS91R05) |
| srimangarlapati@kgpian.iitkgp.ac.in(Garlapati Srimannadh Reddy)(23CS10020)  |
| mmodi1610@kgpian.iitkgp.ac.in (Mayank Modi)(23CS10089)|

All numbers in this report are produced by the code in this folder (the commands in Section 8 regenerate
them) and are read automatically from `results/**/*.json`.

## 1. Data (Dataset 3.3 - Partition C)

* Source: `nltk.corpus.treebank.parsed_sents()` - the 3,914-sentence NLTK sample of the Penn Treebank.
* Construction (`common/ptb.py`, `scripts/00_make_splits.py`): `numpy.random.seed(42)` then one permutation
  of all 3,914 indices, cut into thirds: A = `[0:1305]`, B = `[1305:2610]`, **C = `[2610:3914]` (1,304 sentences)**.
  Inside Partition C a second, independent permutation with the same seed (42) is split 80/10/10:
  **train 1043 / validation 130 / test 131** sentences.
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

## 3. Results (Partition C test split, 131 sentences)

| System | Labeled P | Labeled R | **Labeled F1** | Unlabeled F1 | POS acc | Compare corpus F1 | Compare sentence F1 |
|---|---|---|---|---|---|---|---|
| Task 1: count PCFG + CKY, h=0 (main) | 65.09 | 63.61 | **64.34** | 67.26 | 90.90 | 64.38 | 66.89 |
| Task 1: count PCFG + CKY, h=1 (ablation) | 70.16 | 66.83 | **68.46** | 71.24 | 91.01 | 69.11 | 71.53 |
| Task 3: supervised neural PCFG, h=0 | 67.60 | 64.56 | **66.05** | 68.41 | 90.90 | 65.38 | 67.46 |
| Task 3: supervised neural PCFG, h=1 | 70.64 | 66.71 | **68.62** | 70.95 | 91.01 | 68.20 | 70.33 |
| Task 2: compound PCFG (unsupervised) | - | - | - | - | - | 30.45 | 33.17 |
| Task 2 ablation: 2-layer ReLU binary-rule MLP | - | - | - | - | - | 4.90 | 5.31 |
| Baseline: right-branching trees | - | - | - | - | - | 34.64 | 38.07 |
| Baseline: left-branching trees | - | - | - | - | - | 7.11 | 8.66 |

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
| symbols | 183 | 483 |
| binary rules | 1642 | 2419 |
| rule types with add-1 (count < 5) / total | 4099 / 5290 | 4750 / 6067 |
| test sentences parsed directly / with back-off / flat fallback | 128 / 3 / 0 | 121 / 10 / 0 |
| **labeled F1** | **64.34** | **68.46** |
| unlabeled F1 | 67.26 | 71.24 |
| labeled F1, sentences <= 40 words | 64.67 | 68.49 |

**Markovization ablation.** Moving from 0th- to 1st-order horizontal markovization changes labeled F1 by
+4.11 points. h = 0 forgets every sibling inside a binarized constituent (one intermediate symbol
per parent, e.g. `NP|<>`), which over-generalises: any child sequence is possible. h = 1 remembers the previous
sibling (`NP|<DT>`), so the grammar has 483 instead of 183 symbols, models
local ordering better, but its rules are sparser on ~1k training trees (more sentences need the back-off retry).

## 5. Task 2 - Paper reproduction: Compound PCFG (unsupervised)

**Method** (`scripts/task2_compound_pcfg.py`, `common/compound_pcfg.py`). A device-agnostic port of
`harvardnlp/compound-pcfg` (`models.py`, `PCFG.py`, `train.py`, `eval.py`), with the inside algorithm
vectorised over all spans of the same width (same result as the official loop version, much faster on CPU).
The model never sees a tree: training uses only the words of the 1038 Partition C training sentences
(punctuation removed, lower-cased, digits -> `N`, length-1 sentences skipped, as in the official preprocessing).

| setting | value |
|---|---|
| nonterminals / preterminals | 30 / 60 |
| latent z | R^64, q(z given x) = N(mu, sigma^2) from a 1-layer BiLSTM (hidden 256) over word embeddings of dim 256, max-pooled |
| rule probabilities | all conditioned on [symbol embedding (256); z]. Root rules S -> A and emissions T -> w: MLP Linear -> 2 residual ReLU layers (hidden 256) -> Linear. Binary rules A -> B C: one linear layer -> softmax over (NT+T)^2 child pairs (official `models.py`) |
| objective | ELBO = E_q[log p(x given z)] minus the KL divergence between q(z given x) and N(0, I); log p(x given z) by the inside algorithm |
| optimiser | Adam, lr 0.001, betas (0.75, 0.999), grad-norm clip 3.0 |
| batch / epochs | 4 sentences of equal length / 20 epochs; curriculum max length 30 -> 40 (+1 per epoch) |
| model selection | validation ELBO perplexity (best epoch: 10) |
| decoding | Viterbi (max-probability tree) with z = posterior mean, as in `eval.py` |
| vocabulary | 2117 types (training words seen >= 2 times; rest `<unk>`) |

**Results.** Test corpus F1 **30.45**, sentence F1 **33.17** (right-branching:
34.64 / 38.07, left-branching: 7.11 / 8.66; the induced grammar is at
-4.90 sentence-F1 points relative to right-branching). For reference, Kim et al. (2019) report sentence F1 55.2 (mean) / 60.1 (max) for the
compound PCFG, 50.8 for the neural PCFG and 39.5 for right-branching on the full PTB test set, training on
~40k sentences of WSJ 02-21. Our grammar is trained on 1038 sentences - about 40x less raw text -
which is the expected, intentional weakness of the one-partition-per-group design: the grammar has to induce
constituency from co-occurrence statistics alone, and with ~1k sentences most word types are seen only a
handful of times. Training curves: `results/task2_training_curves.png`.

**Ablation: a 2-layer ReLU MLP for the binary rules.** The assignment describes the rule network as a
"2-layer MLP (hidden 256, ReLU)". In the official code this describes the root and emission networks; the
binary-rule network is a single linear layer. We also trained the literal variant, where the binary rules use
Linear(320, 256) -> ReLU -> Linear (`--rule_layers 2`, everything else identical). It reaches test corpus F1
**4.90** / sentence F1 **5.31** (best epoch 9). Its trees agree with purely
left-branching trees at 63.73 corpus F1 (main model: 11.57): on this small corpus it collapses to a largely
left-branching grammar after a few epochs (see the dashed curve in the training plot), while the official parameterisation
keeps improving. This is why the main run uses the official single-layer binary-rule network.

**Other differences from the official code (all deliberate).** (i) the inference-network LSTM and word
embeddings are 256-dimensional as the assignment specifies (official default 512); (ii) 20 epochs (assignment 15-20;
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
terms). Adam (lr 1e-3, betas 0.75/0.999, clip 3), batches of 8 trees, 40 epochs,
model selection on validation negative log-likelihood per word.

*Decoding.* The neural grammar is exported to the same tables as Task 1 and parsed with the same exact CKY.
Unlike counts, the network gives non-zero probability to child pairs never seen with a parent (generalisation
through shared embeddings); to keep CKY tractable binary rules with probability < 0.001 are pruned unless
the rule was seen in training (5175 binary rules for h = 0 vs. 1642 seen; 17146 vs. 2419 for h = 1).
Emissions are dense: every preterminal can emit every word with a learned probability.

| | Task 1 (counts) h=0 | Task 3 (neural) h=0 | Task 1 (counts) h=1 | Task 3 (neural) h=1 |
|---|---|---|---|---|
| labeled F1 | 64.34 | 66.05 (+1.70) | 68.46 | 68.62 (+0.16) |
| unlabeled F1 | 67.26 | 68.41 (+1.15) | 71.24 | 70.95 (-0.29) |
| POS accuracy | 90.90 | 90.90 | 91.01 | 91.01 |
| compare corpus F1 | 64.38 | 65.38 | 69.11 | 68.20 |

**Does a neurally-parameterised, fully-supervised grammar beat the classical count-based one?** **Only marginally** - the best neural grammar is ahead of the best count-based grammar, but by less than 1 labeled-F1 point, which is within the noise of a 131-sentence test set
(best Task 3 labeled F1 68.62 with h = 1 vs. best Task 1 labeled F1 68.46 with h = 1).
Both are far above the unsupervised Task 2 grammar on the shared unlabeled metric
(68.20 and 69.11 vs. 30.45 corpus F1),
which quantifies how much the gold trees are worth on a corpus this small.

## 7. Discussion

* **Supervision.** On the shared unlabeled metric the supervised grammars reach 69.11
  corpus F1 against 30.45 for the grammar induced from the same ~1k sentences without trees (Task 2);
  right-branching trees score 34.64, the standard baseline for English unsupervised parsing.
* **Parameterisation.** Counts are the exact MLE for the rules they have seen and give nothing to rules they
  have not; the neural grammar shares statistical strength between symbols through embeddings (similar
  symbols get similar rule distributions) and smooths both rules and emissions. Measured effect of switching
  counts -> neural: +1.70 labeled F1 with h = 0 and +0.16 with h = 1.
* **Markovization.** 1st-order horizontal markovization adds sibling context at the cost of sparser rules;
  measured effect of h = 0 -> h = 1: +4.11 labeled F1 for counts and
  +2.57 for the neural grammar.
* **Limitations.** Single run per model (no seed variance on this budget); a small test set (131 sentences)
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
