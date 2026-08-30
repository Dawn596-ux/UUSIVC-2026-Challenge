"""Generate + verify the disjoint K-fold CV manifests against the REAL labeled data.

Regenerates every fold's manifests through the exact code path training will use
(``write_fixed_manifests`` with cv_num_folds/cv_fold), then asserts the three
invariants on the real dataset:

- pairwise disjoint val group sets across folds (patient-level, per task);
- union of all fold val groups == every labeled group of that task;
- no group appears on both sides of any fold's train/val split.

Prints a per-task/per-fold stats table (samples / groups / label counts) and
writes a JSON summary for the run report. CPU-only — safe in AutoDL no-GPU mode.

Run (server):
    python -B scripts/verify_cv_manifests.py --data-root /root/autodl-tmp/data \
        --num-folds 3 --seed 2024
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for direct invocation

from datasets.uusivc2026_paths import (
    _collect_labeled_training_entries,
    derive_group_key,
    resolve_data_root,
    split_entries,
    write_fixed_manifests,
)


def group_sets(entries: List[Dict[str, Any]]) -> set:
    return {derive_group_key(e) for e in entries}


def label_counts(entries: List[Dict[str, Any]]) -> Dict[str, int]:
    return dict(sorted(Counter(str(e.get("class_label_index")) for e in entries).items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and verify K-fold CV manifests on real data.")
    parser.add_argument("--data-root", type=str, default=None,
                        help="UUSIVC2026 data root. Defaults to UUSIVC2026_DATA_ROOT.")
    parser.add_argument("--num-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--manifest-root", type=str, default="./outputs/uusivc2026_fixed",
                        help="Manifest parent dir; folds land in manifests_cv{K}/fold{i} — "
                             "the same locations train.py/run_cv.sh will use.")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Summary json path (default outputs/cv/K{K}_manifest_check.json).")
    args = parser.parse_args()

    k, seed = args.num_folds, args.seed
    data_root = resolve_data_root(args.data_root)
    print(f"[Info] Data root: {data_root}")
    print(f"[Info] K={k} seed={seed}")

    # 1) Regenerate every fold's manifests via the training code path.
    for fold in range(k):
        out_dir = Path(args.manifest_root) / f"manifests_cv{k}" / f"fold{fold}"
        write_fixed_manifests(
            str(data_root), out_dir, phases=("train", "val"),
            local_val_fraction=0.1, seed=seed,
            split_mode="grouped_stratified", cv_num_folds=k, cv_fold=fold,
        )
        print(f"[Info] fold{fold}: manifests written to {out_dir}")

    # 2) Verify invariants on the real data.
    labeled = _collect_labeled_training_entries(data_root)
    summary: Dict[str, Any] = {"num_folds": k, "seed": seed, "data_root": str(data_root), "tasks": {}}
    failures: List[str] = []

    for task, entries in sorted(labeled.items()):
        all_groups = group_sets(entries)
        per_fold: Dict[str, Any] = {}
        val_group_sets: List[set] = []
        for fold in range(k):
            out_dir = Path(args.manifest_root) / f"manifests_cv{k}" / f"fold{fold}"
            with (out_dir / "val_entries.json").open("r", encoding="utf-8") as f:
                val_entries = [e for e in json.load(f) if e.get("task") == task]
            train_entries, val_split = split_entries(
                entries, 0.1, seed, mode="grouped_stratified", cv_num_folds=k, cv_fold=fold
            )
            tg, vg = group_sets(train_entries), group_sets(val_entries)
            if tg & vg:
                failures.append(f"{task} fold{fold}: {len(tg & vg)} groups in BOTH train and val")
            if vg != group_sets(val_split):
                failures.append(f"{task} fold{fold}: manifest val_groups != split_entries val_groups")
            val_group_sets.append(vg)
            per_fold[f"fold{fold}"] = {
                "train_samples": len(train_entries), "train_groups": len(tg),
                "val_samples": len(val_entries), "val_groups": len(vg),
                "val_labels": label_counts(val_entries),
            }
        for i in range(k):
            for j in range(i + 1, k):
                overlap = val_group_sets[i] & val_group_sets[j]
                if overlap:
                    failures.append(f"{task}: fold{i} & fold{j} share {len(overlap)} val groups")
        covered = set().union(*val_group_sets)
        if covered != all_groups:
            failures.append(f"{task}: fold union covers {len(covered)}/{len(all_groups)} groups")

        summary["tasks"][task] = {
            "total_samples": len(entries), "total_groups": len(all_groups),
            "invariants_ok": not any(x.startswith(task) for x in failures),
            **per_fold,
        }
        vals = [per_fold[f"fold{f}"]["val_samples"] for f in range(k)]
        grps = [per_fold[f"fold{f}"]["val_groups"] for f in range(k)]
        print(f"\n[Task] {task}: total {len(entries)} samples / {len(all_groups)} groups")
        print(f"  val_samples/fold: {vals}   val_groups/fold: {grps}")
        for f in range(k):
            print(f"  fold{f} val_labels: {per_fold[f'fold{f}']['val_labels']}")

    print("\n" + "=" * 60)
    if failures:
        print("[FAIL] invariant violations:")
        for x in failures:
            print(f"  - {x}")
        sys.exit(1)
    print("[PASS] all invariants hold on REAL data: folds disjoint, union complete, "
          "no group split across train/val")

    out_json = Path(args.output_json) if args.output_json else Path(f"outputs/cv/K{k}_manifest_check.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Info] summary written to {out_json}")


if __name__ == "__main__":
    main()
