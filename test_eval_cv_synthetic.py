"""Synthetic self-check for the local evaluation ruler (no real dataset needed).

Covers the pieces added on feature-eval-cv:
- disjoint patient-level K-fold split invariants (uusivc2026_paths.split_entries)
- per-sample record export wiring (score_submission.collect_records)
- evaluate_ab patient-level paired bootstrap (identical arms -> zero delta;
  uniform per-patient improvement -> CI excludes 0 with worst > 0)
- aggregate_cv load/summarize + the 3-rule adoption check
- default (non-CV) split path stays legacy (fraction-based holdout)

Run: python -B test_eval_cv_synthetic.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from aggregate_cv import evaluate_rules, load_run, render_comparison, render_single_run
from datasets.uusivc2026_paths import derive_group_key, split_entries
from evaluate_ab import bootstrap_deltas, paired_records
from score_submission import score_submission_dir


def make_cls_entries() -> list[dict]:
    """6 groups x 2 samples, labels balanced across groups."""
    entries = []
    for g in range(6):
        label = g % 2
        for s in range(2):
            entries.append({
                "task": "image_cls", "dataset_name": "DS_CLS",
                "sample_id": f"g{g}_s{s}", "class_label_index": label,
                "input_path_relative": f"imgs/cls_img_{g}_{s}.png",
                "data_partition_group": "private_train", "organ": "thyroid",
            })
    return entries


def make_seg_entries() -> list[dict]:
    """3 groups x 2 frames (video_seg-like group semantics via 2-digit frame suffix)."""
    entries = []
    for g in range(3):
        for f in (1, 2):
            entries.append({
                "task": "image_seg", "dataset_name": "DS_IMG",
                "sample_id": f"v{g}_{f}",
                "input_path_relative": f"imgs/seg_img_v{g}_{f}.png",
                "target_path_relative": f"masks/seg_mask_v{g}_{f}.png",
                "data_partition_group": "public_all", "organ": "breast",
            })
    return entries


def check_kfold_split() -> None:
    k = 3
    val_groups_per_fold = []
    for fold in range(k):
        all_groups, train_groups, val_groups = set(), set(), set()
        for task_entries in (make_cls_entries(), make_seg_entries()):  # split per task, as production does
            train, val = split_entries(task_entries, 0.1, 2024, mode="grouped_stratified",
                                       cv_num_folds=k, cv_fold=fold)
            assert train and val, fold
            # patient-level guarantee: no group on both sides
            tg, vg = {derive_group_key(e) for e in train}, {derive_group_key(e) for e in val}
            assert not (tg & vg), fold
            # every entry accounted for
            assert len(train) + len(val) == len(task_entries)
            train_groups |= tg
            val_groups |= vg
            all_groups |= {derive_group_key(e) for e in task_entries}
        val_groups_per_fold.append(val_groups)
        assert train_groups | val_groups == all_groups
    # disjoint folds whose union covers all groups
    all_groups = {derive_group_key(e) for e in make_cls_entries() + make_seg_entries()}
    for i in range(k):
        for j in range(i + 1, k):
            assert not (val_groups_per_fold[i] & val_groups_per_fold[j]), (i, j)
    assert set().union(*val_groups_per_fold) == all_groups
    # determinism
    _, val_again = split_entries(make_cls_entries(), 0.1, 2024, mode="grouped_stratified",
                                 cv_num_folds=k, cv_fold=0)
    assert {derive_group_key(e) for e in val_again} <= val_groups_per_fold[0]
    # label stratification: cls labels balanced across folds (round-robin, slack <= 2 samples)
    cls_by_fold = []
    cls_entries = make_cls_entries()
    for fold in range(k):
        _, val = split_entries(cls_entries, 0.1, 2024, mode="grouped_stratified",
                               cv_num_folds=k, cv_fold=fold)
        cls_by_fold.append([e["class_label_index"] for e in val])
    for label in (0, 1):
        counts = [sum(1 for l in labels if l == label) for labels in cls_by_fold]
        assert max(counts) - min(counts) <= 2, counts
    # invalid args rejected
    for bad in ((2, None), (2, 2), (1, 0)):
        try:
            split_entries(cls_entries, 0.1, 2024, cv_num_folds=bad[0], cv_fold=bad[1])
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass
    print("[PASS] K-fold split: disjoint/union/deterministic/stratified")


def check_legacy_split_unchanged() -> None:
    entries = make_cls_entries()
    train, val = split_entries(entries, 0.1, 2024, mode="grouped_stratified")
    assert val and train
    assert not ({derive_group_key(e) for e in train} & {derive_group_key(e) for e in val})
    # stratified minimum: >=1 val group per label bucket on tiny data (legacy rule)
    assert len(val) == 4  # 3 groups/label -> round(3*0.1)->max(1,0)=1 group per label = 4 entries
    train_full, val_empty = split_entries(entries, 0.0, 2024)
    assert len(train_full) == len(entries) and val_empty == []
    print("[PASS] legacy fraction split path unchanged")


def check_records_export() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="eval_cv_selfcheck_"))
    root, pred = tmp / "data_root", tmp / "pred"
    (root / "masks").mkdir(parents=True)
    (pred / "masks").mkdir(parents=True)

    yy, xx = np.mgrid[0:32, 0:32]
    disk = ((yy - 16) ** 2 + (xx - 16) ** 2 <= 6**2).astype(np.uint8) * 255
    entries = []
    for g in range(2):
        gt = np.roll(disk, 4 * g, axis=1)  # group 1 GT shifted -> imperfect for identical pred
        Image.fromarray(gt).save(root / "masks" / f"m{g}.png")
        Image.fromarray(np.roll(disk, 4 * g - 1, axis=1)).save(pred / "masks" / f"m{g}.png")
        entries.append({
            "task": "image_seg", "dataset_name": "DS_IMG", "sample_id": f"s{g}",
            "input_path_relative": f"imgs/i{g}.png",
            "target_path_relative": f"masks/m{g}.png",
            "data_partition_group": "public_all", "organ": "breast", "_root": str(root),
        })
    cls = {f"c{g}": {"prediction": 1, "probability": [0.2, 0.8]} for g in range(2)}
    (pred / "classification.json").write_text(json.dumps(cls), encoding="utf-8")
    for g in range(2):
        entries.append({
            "task": "image_cls", "dataset_name": "DS_CLS", "sample_id": f"c{g}",
            "class_label_index": g % 2,
            "input_path_relative": f"imgs/c{g}.png",
            "data_partition_group": "private_train", "organ": "thyroid", "_root": str(root),
        })

    metrics = score_submission_dir(pred, entries, tolerance=1, collect_records=True)
    records = metrics["records"]
    assert len(records) == 4
    by_task = {r["task"]: r for r in records}
    # group keys must match derive_group_key (patient-level resampling unit)
    for e in entries:
        rec = next(r for r in records if r["key"] == (e["sample_id"]))
        assert rec["group"] == derive_group_key(e)
        assert rec["dataset"] == e["dataset_name"]
    assert "score" in by_task["image_seg"] and "prob" in by_task["image_cls"]
    print("[PASS] record export: keys/groups/datasets wired through")


def check_evaluate_ab() -> None:
    # Deterministic synthetic cls records with per-group spread so bootstrap
    # resamples genuinely vary. Positives: 0.40..0.90; negatives: 0.35..0.85.
    n_groups = 24

    def arm(neg_shift: float) -> list[dict]:
        records = []
        for g in range(n_groups):
            frac = g / (n_groups - 1)
            for i in range(2):
                if g % 2 == 1:  # positive group
                    label, prob = 1, 0.40 + 0.50 * frac
                else:  # negative group, shifted down by the candidate
                    label, prob = 0, max(0.35 + 0.50 * frac - neg_shift, 0.01)
                records.append({"key": f"g{g}_s{i}", "task": "image_cls", "dataset": "DS_CLS",
                                "group": f"grp{g}", "gt": label,
                                "prob": min(prob, 0.99), "pred": int(prob > 0.5)})
        return records

    a, b = arm(0.0), arm(0.0)
    rows = bootstrap_deltas(a, b, n_boot=200, seed=0, alpha=0.05)
    assert abs(rows["overall"]["delta"]) < 1e-12
    assert rows["overall"]["ci_low"] == 0.0 and rows["overall"]["ci_high"] == 0.0
    assert rows["overall"]["worst"] == 0.0

    a2, b2 = arm(0.0), arm(0.30)  # candidate pushes every negative prob down 0.30
    rows2 = bootstrap_deltas(a2, b2, n_boot=300, seed=0, alpha=0.05)
    assert rows2["overall"]["delta"] > 0.0, rows2["overall"]
    assert rows2["overall"]["ci_low"] > 0.0, rows2["overall"]
    assert rows2["overall"]["worst"] > 0.0  # uniform per-group improvement -> resamples stay positive
    assert rows2["overall"]["p_pos"] == 1.0

    # pairing guard: mismatched key sets must raise
    try:
        paired_records(a2, a2[:10])
        raise AssertionError("expected ValueError on key mismatch")
    except ValueError:
        pass

    # mixed seg+cls records: segments must stay task-homogeneous (regression:
    # cls buckets once received seg records and crashed on missing 'gt')
    mixed_a = arm(0.0) + [
        {"key": f"vg{g}", "task": "image_seg", "dataset": "DS_IMG", "group": f"grp{g}",
         "dsc": 0.5 + 0.1 * (g % 3), "nsd": 0.6, "score": 0.7} for g in range(n_groups // 2)]
    mixed_b = arm(0.30) + [
        {"key": f"vg{g}", "task": "image_seg", "dataset": "DS_IMG", "group": f"grp{g}",
         "dsc": 0.6 + 0.1 * (g % 3), "nsd": 0.6, "score": 0.75} for g in range(n_groups // 2)]
    rows3 = bootstrap_deltas(mixed_a, mixed_b, n_boot=100, seed=0, alpha=0.05)
    assert all(np.isfinite(v) for r in rows3.values() for v in r.values()), rows3
    assert rows3["image_seg"]["delta"] > 0 and rows3["image_cls"]["delta"] > 0
    print("[PASS] evaluate_ab: zero-delta identity + improvement CI/worst + mixed-task segments")


def write_fold(root: Path, name: str, overall: float, tasks: dict[str, float], datasets: dict[str, float]) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({
        "per_task": {t: {"score": s} for t, s in tasks.items()},
        "per_dataset": {k: {"score": s, "n": 30} for k, s in datasets.items()},
        "overall_score": overall,
    }), encoding="utf-8")


def check_aggregate_cv() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="cv_agg_selfcheck_"))
    tasks = ["image_seg", "ceus_seg", "video_seg", "image_cls", "ceus_cls"]
    datasets = {f"{t}/DS": 0.5 + 0.01 * i for i, t in enumerate(tasks)}

    base = tmp / "K3_baseline"
    cand = tmp / "K3_candidate"
    base_overall, cand_overall = [], []
    for fold, boost in enumerate((0.004, 0.006, 0.008)):  # unanimous, increasing improvement
        b_overall = 0.70 + 0.01 * fold
        c_overall = b_overall + boost
        write_fold(base, f"fold{fold}", b_overall,
                   {t: b_overall + 0.001 * i for i, t in enumerate(tasks)}, datasets)
        cand_datasets = {k: v + boost for k, v in datasets.items()}
        write_fold(cand, f"fold{fold}", c_overall,
                   {t: c_overall + 0.001 * i for i, t in enumerate(tasks)}, cand_datasets)
        base_overall.append(b_overall)
        cand_overall.append(c_overall)

    run, base_run = load_run(cand), load_run(base)
    assert run["fold_names"] == ["fold0", "fold1", "fold2"]
    text = render_single_run(run)
    assert "mean" in text and "overall" in text

    result = evaluate_rules(run, base_run)
    assert result["adopted"] is True, result
    assert result["rule1_majority_improve"] and result["rule2_gt_std"] and result["rule3_no_task_regression"]
    assert abs(result["overall"]["mean"] - np.mean(np.array(cand_overall) - np.array(base_overall))) < 1e-12

    # degraded candidate must fail
    worse = tmp / "K3_worse"
    for fold, b_overall in enumerate(base_overall):
        write_fold(worse, f"fold{fold}", b_overall - 0.01,
                   {t: b_overall - 0.01 for t in tasks}, datasets)
    result_bad = evaluate_rules(load_run(worse), base_run)
    assert result_bad["adopted"] is False
    render_comparison(run, base_run, result)
    render_comparison(load_run(worse), base_run, result_bad)
    print("[PASS] aggregate_cv: load/summarize + 3-rule adopt/reject")


def main() -> None:
    check_kfold_split()
    check_legacy_split_unchanged()
    check_records_export()
    check_evaluate_ab()
    check_aggregate_cv()
    print("[PASS] eval-cv synthetic self-check: all assertions OK")


if __name__ == "__main__":
    main()
