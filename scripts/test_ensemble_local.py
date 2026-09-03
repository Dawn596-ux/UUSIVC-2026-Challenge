"""合成门：验证 predict.py 三任务函数 extra_models 改动的正确性（CPU、无数据、无 checkpoint）。

① 默认路径不变性：extra_models=None 输出与内嵌的 bc132e4 原版参考实现逐位一致；
② ensemble 路径 = 手工加权平均期望值；
③ 空 list 退化为默认路径；
④ cls 忽略 tta 位；
⑤ seg 成员 tta=False 只做 1 次前向、True 做 2 次（主模型恒 2 次）；
⑥ 第三成员加权正确（分母 = 1 + Σw）。

Run:
  python -B scripts/test_ensemble_local.py        # 本地或服务器均可，无需 GPU/数据/checkpoint
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import predict as P  # noqa: E402

DEVICE = torch.device("cpu")
W_COORD = torch.linspace(-1.0, 1.0, 224).view(1, 1, 1, 224)


class FakeSegModel(torch.nn.Module):
    """确定性 seg 假模型：固定 base + 翻转敏感的输入信号（验证 TTA 真实生效）。"""

    def __init__(self, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.base = torch.randn(2, 224, 224, generator=g)
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        x = batch[P.IMAGE]
        signal = (x * W_COORD).sum(dim=(1, 2, 3)) * 0.01  # 水平翻转后变号
        logits = self.base.unsqueeze(0) + signal.view(-1, 1, 1)
        return {"seg_logits": logits}


class FakeClsModel(torch.nn.Module):
    """确定性 cls 假模型：固定 4 类 logits（> n 类时验证逐成员截断）。"""

    def __init__(self, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.logits = torch.randn(1, 4, generator=g) * 2.0
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        return {"cls_logits": self.logits}


# ---------------------------------------------------------------------------
# bc132e4 原版参考实现（逐字抄录）——默认路径不变性的对照基准
# ---------------------------------------------------------------------------

@torch.no_grad()
def ref_predict_image_seg(model, entry, phase_root, out_dir, device):
    path = phase_root / P.normalize_rel(entry["input_path_relative"])
    image = P.read_image(path)
    tensor, _ = P.BasicImageTransform((224, 224), binary_mask=False)(image, None)
    batch = P.make_seg_batch(tensor, entry["task"], P.BUS_IMAGE, P.TASK_DATASET_NAME["image_seg"], path.stem, device)
    logits = model(batch)["seg_logits"]
    logits_flip = model(P._flip_batch_image(batch))["seg_logits"]
    prob = torch.softmax(logits[0], dim=0) + torch.softmax(logits_flip[0], dim=0).flip(-1)
    pred = torch.argmax(prob, dim=0).detach().cpu().numpy()
    mask = P.resize_binary(pred, P.image_hw(entry, phase_root))
    out_path = out_dir / P.output_rel(entry)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(out_path)


@torch.no_grad()
def ref_predict_image_cls(model, entry, phase_root, device):
    path = phase_root / P.normalize_rel(entry["input_path_relative"])
    image = P.read_image(path)
    tensor, _ = P.BasicImageTransform((224, 224), binary_mask=True)(image, None)
    batch = P.make_cls_batch(tensor, entry["task"], P.BUS_IMAGE, P.TASK_DATASET_NAME["image_cls"], path.stem, device)
    probs = torch.softmax(model(batch)["cls_logits"], dim=1)[0].detach().cpu().numpy()
    n = P.class_count(entry)
    probs = probs[:n]
    probs = probs / max(float(probs.sum()), 1e-12)
    return {"prediction": int(np.argmax(probs)), "probability": [float(x) for x in probs]}


@torch.no_grad()
def ref_predict_ceus_cls(model, entry, phase_root, device, num_frames):
    path = phase_root / P.normalize_rel(entry["input_path_relative"])
    video = P.load_video_npy(path)
    frames = P.sampled_frames(video, num_frames)
    tensor, _ = P.BasicVideoTransform((224, 224), binary_mask=True)(frames, None)
    batch = P.make_cls_batch(tensor, entry["task"], P.CEUS_VIDEO, P.TASK_DATASET_NAME["ceus_cls"], path.stem, device)
    probs = torch.softmax(model(batch)["cls_logits"], dim=1)[0].detach().cpu().numpy()
    n = P.class_count(entry)
    probs = probs[:n]
    probs = probs / max(float(probs.sum()), 1e-12)
    return {"prediction": int(np.argmax(probs)), "probability": [float(x) for x in probs]}


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def make_image_fixture(root: Path, rel: str, size=(32, 48)) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(7)
    arr = rng.randint(0, 255, size=(size[0], size[1], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


def make_video_fixture(root: Path, rel: str, shape=(5, 16, 16, 3)) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(11)
    np.save(path, rng.randint(0, 255, size=shape, dtype=np.uint8))
    return path


def seg_entry(rel: str, target_rel: str) -> dict:
    return {"task": "image_seg", "dataset_name": "breast", "input_path_relative": rel,
            "target_path_relative": target_rel}


def cls_entry(rel: str, n_classes: int) -> dict:
    return {"task": "image_cls", "dataset_name": "breast", "input_path_relative": rel,
            "class_config": {str(i): f"c{i}" for i in range(n_classes - 1)}}


def ceus_cls_entry(rel: str, n_classes: int) -> dict:
    return {"task": "ceus_cls", "dataset_name": "thyroid_ceus", "input_path_relative": rel,
            "class_config": {str(i): f"c{i}" for i in range(n_classes - 1)}}


def read_png(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path))


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_default_seg_unchanged(tmp: Path) -> None:
    make_image_fixture(tmp, "imgs/a.png")
    entry = seg_entry("imgs/a.png", "masks/a.png")
    model = FakeSegModel(1)
    ref_predict_image_seg(model, entry, tmp, tmp / "ref", DEVICE)
    P.predict_image_seg(model, entry, tmp, tmp / "new", DEVICE, extra_models=None)
    assert np.array_equal(read_png(tmp / "ref" / "masks" / "a.png"),
                          read_png(tmp / "new" / "masks" / "a.png")), "seg 默认路径与参考实现不一致"
    P.predict_image_seg(model, entry, tmp, tmp / "empty", DEVICE, extra_models=[])
    assert np.array_equal(read_png(tmp / "ref" / "masks" / "a.png"),
                          read_png(tmp / "empty" / "masks" / "a.png")), "extra_models=[] 未退化为默认路径"


def test_ensemble_seg(tmp: Path) -> None:
    make_image_fixture(tmp, "imgs/a.png")
    entry = seg_entry("imgs/a.png", "masks/a.png")
    phase_root = tmp

    for use_tta, tag in ((True, "tta"), (False, "notta")):
        f, g = FakeSegModel(1), FakeSegModel(2)
        out_dir = tmp / f"ens_{tag}"
        P.predict_image_seg(f, entry, phase_root, out_dir, DEVICE,
                            extra_models=[(g, 0.5, use_tta)])
        f_calls, g_calls = f.calls, g.calls  # 先快照：下面的期望值计算也会调用假模型
        # 手工期望值：P = [1.0·TTA(f) + w·member(g)] / (1 + w)
        path = phase_root / P.normalize_rel(entry["input_path_relative"])
        tensor, _ = P.BasicImageTransform((224, 224), binary_mask=False)(P.read_image(path), None)
        batch = P.make_seg_batch(tensor, entry["task"], P.BUS_IMAGE,
                                 P.TASK_DATASET_NAME["image_seg"], path.stem, DEVICE)
        member = P._seg_tta_prob(g, batch) if use_tta else torch.softmax(g(batch)["seg_logits"][0], dim=0)
        prob = (P._seg_tta_prob(f, batch) + 0.5 * member) / 1.5
        expected = P.resize_binary(torch.argmax(prob, dim=0).detach().cpu().numpy(),
                                   P.image_hw(entry, phase_root))
        actual = read_png(out_dir / "masks" / "a.png")
        assert np.array_equal(expected, actual), f"seg ensemble({tag}) 与手工加权平均不一致"
        assert f_calls == 2, f"主模型前向次数应为 2（原+翻转），实际 {f_calls}"
        assert g_calls == (2 if use_tta else 1), f"辅模型前向次数错误（tta={use_tta}），实际 {g_calls}"


def test_ensemble_seg_third_member(tmp: Path) -> None:
    make_image_fixture(tmp, "imgs/a.png")
    entry = seg_entry("imgs/a.png", "masks/a.png")
    f, g, h = FakeSegModel(1), FakeSegModel(2), FakeSegModel(3)
    out_dir = tmp / "ens_third"
    P.predict_image_seg(f, entry, tmp, out_dir, DEVICE,
                        extra_models=[(g, 0.5, False), (h, 0.5, False)])
    path = tmp / P.normalize_rel(entry["input_path_relative"])
    tensor, _ = P.BasicImageTransform((224, 224), binary_mask=False)(P.read_image(path), None)
    batch = P.make_seg_batch(tensor, entry["task"], P.BUS_IMAGE,
                             P.TASK_DATASET_NAME["image_seg"], path.stem, DEVICE)
    prob = (P._seg_tta_prob(f, batch)
            + 0.5 * torch.softmax(g(batch)["seg_logits"][0], dim=0)
            + 0.5 * torch.softmax(h(batch)["seg_logits"][0], dim=0)) / 2.0
    expected = P.resize_binary(torch.argmax(prob, dim=0).detach().cpu().numpy(),
                               P.image_hw(entry, tmp))
    assert np.array_equal(expected, read_png(out_dir / "masks" / "a.png")), "第三成员加权不一致"


def test_default_cls_unchanged(tmp: Path) -> None:
    make_image_fixture(tmp, "imgs/b.png")
    entry = cls_entry("imgs/b.png", 3)
    model = FakeClsModel(4)
    ref = ref_predict_image_cls(model, entry, tmp, DEVICE)
    new = P.predict_image_cls(model, entry, tmp, DEVICE, extra_models=None)
    assert ref == new, f"image_cls 默认路径与参考实现不一致: {ref} vs {new}"
    new_empty = P.predict_image_cls(model, entry, tmp, DEVICE, extra_models=[])
    assert ref == new_empty, "image_cls extra_models=[] 未退化为默认路径"


def test_ensemble_cls(tmp: Path) -> None:
    make_image_fixture(tmp, "imgs/b.png")
    entry = cls_entry("imgs/b.png", 3)  # n=3 < 4 logits，覆盖逐成员截断
    f, g = FakeClsModel(4), FakeClsModel(5)
    res = P.predict_image_cls(f, entry, tmp, DEVICE, extra_models=[(g, 0.5, True)])
    f_calls, g_calls = f.calls, g.calls  # 快照：期望值计算也会调用假模型
    path = tmp / P.normalize_rel(entry["input_path_relative"])
    tensor, _ = P.BasicImageTransform((224, 224), binary_mask=True)(P.read_image(path), None)
    batch = P.make_cls_batch(tensor, entry["task"], P.BUS_IMAGE,
                             P.TASK_DATASET_NAME["image_cls"], path.stem, DEVICE)
    n = P.class_count(entry)
    expected = (P._cls_member_prob(f, batch, n) + 0.5 * P._cls_member_prob(g, batch, n)) / 1.5
    assert res["probability"] == [float(x) for x in expected], "image_cls ensemble 概率与手工平均不一致"
    assert res["prediction"] == int(np.argmax(expected)), "image_cls ensemble prediction 不一致"
    assert f_calls == 1 and g_calls == 1, f"cls 成员各只应做 1 次前向，实际 f={f_calls} g={g_calls}"
    # tta 位被忽略（新模型对，避免计数串扰）
    f2, g2 = FakeClsModel(4), FakeClsModel(5)
    res_no_tta = P.predict_image_cls(f2, entry, tmp, DEVICE, extra_models=[(g2, 0.5, False)])
    assert res == res_no_tta, "image_cls ensemble 不应受 tta 位影响"


def test_default_ceus_cls_unchanged(tmp: Path) -> None:
    make_video_fixture(tmp, "videos/c.npy")
    entry = ceus_cls_entry("videos/c.npy", 2)
    model = FakeClsModel(6)
    ref = ref_predict_ceus_cls(model, entry, tmp, DEVICE, num_frames=4)
    new = P.predict_ceus_cls(model, entry, tmp, DEVICE, num_frames=4, extra_models=None)
    assert ref == new, f"ceus_cls 默认路径与参考实现不一致: {ref} vs {new}"
    new_empty = P.predict_ceus_cls(model, entry, tmp, DEVICE, num_frames=4, extra_models=[])
    assert ref == new_empty, "ceus_cls extra_models=[] 未退化为默认路径"


def test_ensemble_ceus_cls(tmp: Path) -> None:
    make_video_fixture(tmp, "videos/c.npy")
    entry = ceus_cls_entry("videos/c.npy", 2)
    f, g = FakeClsModel(6), FakeClsModel(7)
    res = P.predict_ceus_cls(f, entry, tmp, DEVICE, num_frames=4, extra_models=[(g, 0.5, False)])
    video = P.load_video_npy(tmp / "videos" / "c.npy")
    frames = P.sampled_frames(video, 4)
    tensor, _ = P.BasicVideoTransform((224, 224), binary_mask=True)(frames, None)
    batch = P.make_cls_batch(tensor, entry["task"], P.CEUS_VIDEO,
                             P.TASK_DATASET_NAME["ceus_cls"], "c", DEVICE)
    n = P.class_count(entry)
    expected = (P._cls_member_prob(f, batch, n) + 0.5 * P._cls_member_prob(g, batch, n)) / 1.5
    assert res["probability"] == [float(x) for x in expected], "ceus_cls ensemble 概率与手工平均不一致"


def main() -> None:
    tests = [
        test_default_seg_unchanged,
        test_ensemble_seg,
        test_ensemble_seg_third_member,
        test_default_cls_unchanged,
        test_ensemble_cls,
        test_default_ceus_cls_unchanged,
        test_ensemble_ceus_cls,
    ]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for test in tests:
            test(tmp)
            print(f"[PASS] {test.__name__}")
    print(f"\nAll {len(tests)} ensemble synthetic tests passed.")


if __name__ == "__main__":
    main()
