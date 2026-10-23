"""Step 0: build the shared seeded 3-way partition + 80/10/10 splits and save index lists.
Also writes the cleaned Partition C trees (one bracketed tree per line) for inspection."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)  # show progress live, also when piped to tee
from common.ptb import (PARTITION, ROOT_DIR, SPLIT_FILE, load_partition, save_splits,
                        tree_to_line)

splits = save_splits()
print(f"saved {SPLIT_FILE}")
for name, s in splits.items():
    print(f"Partition {name}: total={len(s['indices'])} train={len(s['train'])} "
          f"validation={len(s['validation'])} test={len(s['test'])}")

# disjointness sanity checks
allidx = [set(s["indices"]) for s in splits.values()]
assert not (allidx[0] & allidx[1]) and not (allidx[0] & allidx[2]) and not (allidx[1] & allidx[2])
c = splits[PARTITION]
assert set(c["train"]) | set(c["validation"]) | set(c["test"]) == set(c["indices"])

data = load_partition(PARTITION)
out_dir = os.path.join(ROOT_DIR, "data", f"partition_{PARTITION}")
os.makedirs(out_dir, exist_ok=True)
for name, items in data.items():
    with open(os.path.join(out_dir, f"{name}_trees.txt"), "w") as f:
        for _, t in items:
            f.write(tree_to_line(t) + "\n")
    lens = [len(t.leaves()) for _, t in items]
    print(f"Partition {PARTITION}/{name}: {len(items)} trees, avg len {sum(lens)/len(lens):.1f}, "
          f"max len {max(lens)}")
print("example cleaned tree:", tree_to_line(data["train"][0][1])[:200])
