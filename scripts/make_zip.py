"""Builds ../NLP_ASSIGNMENT_GROUP_9.zip (next to the project folder) containing the folder
NLP_ASSIGNMENT_GROUP_9/ with code, data splits, results, logs, report. Works on Windows/Linux/macOS.
Model checkpoints (*.pt, ~55 MB) are left out to keep the upload small (they can be regenerated);
add --with-checkpoints to include them."""
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)  # show progress live, also when piped to tee
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NAME = "NLP_ASSIGNMENT_GROUP_9"
WITH_CKPT = "--with-checkpoints" in sys.argv
SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", ".idea", ".vscode", "nltk_data", ".ipynb_checkpoints"}
REQUIRED = ["REPORT.md", "README.md", "requirements.txt", "results/RESULTS.md",
            "data/splits/partition_splits_seed42.json",
            "results/task1/task1_h0_test_metrics.json", "results/task1/task1_h1_test_metrics.json",
            "results/task2/task2_test_metrics.json",
            "results/task2_rule2layer/task2_test_metrics.json",
            "results/task3/task3_h0_test_metrics.json", "results/task3/task3_h1_test_metrics.json"]
missing = [p for p in REQUIRED if not os.path.exists(os.path.join(ROOT_DIR, p))]
if missing:
    sys.exit(f"not ready to zip, missing: {missing}")

out = os.path.join(os.path.dirname(ROOT_DIR), NAME + ".zip")
n = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for d, dirs, files in os.walk(ROOT_DIR):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
        for fn in files:
            if fn.endswith((".pyc", ".zip")) or fn == ".DS_Store" or (fn.endswith(".pt") and not WITH_CKPT):
                continue
            p = os.path.join(d, fn)
            z.write(p, os.path.join(NAME, os.path.relpath(p, ROOT_DIR)))
            n += 1
print(f"wrote {out} ({n} files, {os.path.getsize(out)/1e6:.1f} MB)")
