# NLP Assignment - Group 9 - Project 3 (Penn Treebank, Partition C)

From CKY to Neural Grammar Induction. The full write-up with all results is in **`REPORT.md`**;
the results table alone is in `results/RESULTS.md`.

## Layout

```
common/ptb.py              seeded 3-way partition (seed 42) + 80/10/10 split, label cleaning
common/metrics.py          EVALB-style labeled/unlabeled bracketing F1; compound-PCFG unlabeled F1
common/pcfg.py             CNF transform (collapse unary + markovized binarization), UNK signatures, exact CKY
common/compound_pcfg.py    Compound PCFG model, vectorised inside algorithm, Viterbi
scripts/00_make_splits.py  step 0: build + save split index lists (data/splits/)
scripts/task1_cky_pcfg.py  Task 1: count-based PCFG (MLE, add-1 on rules seen < 5 times) + CKY; --horz 0|1
scripts/task2_compound_pcfg.py  Task 2: unsupervised compound PCFG (raw training sentences only)
scripts/task3_neural_pcfg.py    Task 3: supervised neural PCFG (Task 2 parameterisation, no z); --horz 0|1
scripts/make_report.py     builds REPORT.md + results/RESULTS.md + plots from results/**/*.json
scripts/make_zip.py        builds ../NLP_ASSIGNMENT_GROUP_9.zip
results/                   metrics JSON, predicted/gold trees, checkpoints, plots
logs/                      console logs of every run
```

## Run everything

```
pip install -r requirements.txt
python scripts/00_make_splits.py
python scripts/task1_cky_pcfg.py --horz 0
python scripts/task1_cky_pcfg.py --horz 1
python scripts/task2_compound_pcfg.py          # uses a GPU automatically if available
python scripts/task2_compound_pcfg.py --rule_layers 2 --tag rule2layer   # ablation
python scripts/task3_neural_pcfg.py --horz 0
python scripts/task3_neural_pcfg.py --horz 1
python scripts/make_report.py
python scripts/make_zip.py
```

Only Partition C (indices `[2610:3914]` of the seed-42 permutation) is ever used.
