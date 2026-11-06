"""Models, loss, metrics, NDWI baseline, ONNX export and an end-to-end train run (tiny, CPU)."""

import json

import numpy as np
import pytest
import torch

from impactlens.data import packed
from impactlens.data.preprocess import FLOOD_BAND_CLIPS
from impactlens.models import baseline, export, train
from impactlens.models.losses import bce_dice_loss
from impactlens.models.metrics import ConfusionMeter
from impactlens.models.siamese_unet import SiameseUNetDiff
from impactlens.models.unet import UNet

# ---------------------------------------------------------------------------- networks


def test_unet_shape_and_size_check():
    net = UNet(in_ch=6, base=4).eval()
    assert net(torch.rand(2, 6, 32, 48)).shape == (2, 1, 32, 48)
    with pytest.raises(ValueError):
        net(torch.rand(1, 6, 30, 32))  # not divisible by 16


def test_siamese_shape_and_it_reads_both_images():
    net = SiameseUNetDiff(base=4).eval()
    a, b = torch.rand(2, 4, 32, 32), torch.rand(2, 4, 32, 32)
    out = net(a, b)
    assert out.shape == (2, 1, 32, 32)
    # Identical dates -> every skip difference is exactly 0, whatever the image is.
    same_1, same_2 = net(a, a), net(torch.rand(2, 4, 32, 32).mul(0) + a, a)
    assert torch.allclose(same_1, same_2)
    assert not torch.allclose(out, net(a, torch.rand(2, 4, 32, 32)))  # image 2 matters


def test_siamese_encoder_is_shared():
    net = SiameseUNetDiff(base=4)
    assert sum(1 for n, _ in net.named_parameters() if n.startswith("encoder")) > 0
    assert not hasattr(net, "encoder2")  # one set of weights for both dates


# ------------------------------------------------------------------------ loss, metrics


def test_loss_ignores_invalid_pixels():
    target = torch.zeros(1, 1, 8, 8)
    target[..., :4] = 1
    valid = torch.ones(1, 1, 8, 8)
    valid[..., 6:] = 0
    logits = torch.randn(1, 1, 8, 8)
    base = bce_dice_loss(logits, target, valid)
    scrambled = logits.clone()
    scrambled[..., 6:] = 100.0  # wildly wrong, but only on ignored pixels
    assert torch.allclose(base, bce_dice_loss(scrambled, target, valid))


def test_loss_is_small_when_right_and_large_when_wrong():
    target = (torch.rand(1, 1, 16, 16) > 0.5).float()
    valid = torch.ones_like(target)
    good = bce_dice_loss((target * 2 - 1) * 12, target, valid)
    bad = bce_dice_loss((1 - target * 2) * 12, target, valid)
    assert good < 0.05 and bad > 5


def test_loss_backward_works_with_all_pixels_invalid():
    logits = torch.randn(1, 1, 4, 4, requires_grad=True)
    loss = bce_dice_loss(logits, torch.ones(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_confusion_meter_counts_valid_pixels_only():
    pred = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0])
    target = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])
    valid = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])  # last pixel ignored
    m = ConfusionMeter()
    m.update_pred(pred, target, valid)
    assert (m.tp, m.fp, m.fn, m.tn) == (1, 1, 1, 1)
    r = m.compute()
    assert r["precision"] == r["recall"] == 0.5 and r["f1"] == 0.5
    assert r["iou"] == pytest.approx(1 / 3)


def test_confusion_meter_empty_is_zero_not_nan():
    assert ConfusionMeter().compute() == {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}


# ---------------------------------------------------------------------------- baseline


def _fake_img(green, nir, swir=0.0):
    """Normalized (1,6,2,2) image; reflectance in green / nir / swir16 is the given constant."""
    img = torch.zeros(1, 6, 2, 2)
    img[:, 1] = green / FLOOD_BAND_CLIPS[1]
    img[:, 3] = nir / FLOOD_BAND_CLIPS[3]
    img[:, 4] = swir / FLOOD_BAND_CLIPS[4]
    return img


def test_ndwi_is_positive_for_water_and_negative_for_vegetation():
    assert ndwi_mean(_fake_img(0.10, 0.03)) > 0.4  # water: bright green, dark NIR
    assert ndwi_mean(_fake_img(0.06, 0.40)) < -0.5  # vegetation: bright NIR


def test_mndwi_uses_swir_not_nir():
    # Dark-NIR shadow that is still bright in SWIR: NDWI is fooled (positive), MNDWI is not.
    shadow = _fake_img(green=0.05, nir=0.02, swir=0.20)
    assert float(baseline.ndwi(shadow).mean()) > 0.4
    assert float(baseline.mndwi(shadow).mean()) < -0.5
    water = _fake_img(green=0.10, nir=0.03, swir=0.02)
    assert float(baseline.mndwi(water).mean()) > 0.5


def ndwi_mean(img):
    return float(baseline.ndwi(img).mean())


# ----------------------------------------------------------------- end-to-end tiny runs


def _write_fake_npz(tmp_path):
    """Tiny packed datasets (64x64 scenes) so a full train run takes seconds."""
    rng = np.random.default_rng(0)
    img = lambda *s: rng.integers(0, 256, s, dtype=np.uint8)  # noqa: E731

    arrays = {}
    for city in ("a", "b", "c", "d"):
        arrays[f"{city}__img1"], arrays[f"{city}__img2"] = img(4, 64, 64), img(4, 64, 64)
        arrays[f"{city}__mask"] = (rng.random((64, 64)) > 0.8).astype(np.uint8)
    arrays["meta"] = packed._meta(
        "oscd", official_train=["a", "b", "c"], official_test=["d"], val_cities=["c"]
    )
    np.savez_compressed(tmp_path / "oscd.npz", **arrays)

    arrays = {}
    for split, n in (("train", 4), ("valid", 2), ("test", 2), ("bolivia", 2)):
        arrays[f"{split}_img"] = img(n, 6, 64, 64)
        arrays[f"{split}_mask"] = (rng.random((n, 64, 64)) > 0.7).astype(np.uint8)
        arrays[f"{split}_valid"] = np.ones((n, 64, 64), np.uint8)
        arrays[f"{split}_ids"] = np.array([f"{split}_{i}" for i in range(n)])
    arrays["meta"] = packed._meta("sen1floods11", splits=["train", "valid", "test"])
    np.savez_compressed(tmp_path / "sen1floods11.npz", **arrays)


def _run(task, tmp_path, *extra):
    _write_fake_npz(tmp_path)
    out = tmp_path / f"out_{task}"
    train.main(["--task", task, "--data", str(tmp_path), "--out", str(out), "--epochs", "2",
                "--device", "cpu", "--base", "4", "--tile", "64", "--batch-size", "4",
                "--crops", "8", *extra])  # fmt: skip
    return out, json.loads((out / "metrics.json").read_text())


def test_train_oscd_end_to_end(tmp_path):
    out, m = _run("oscd", tmp_path)
    assert (out / "best.pt").exists() and (out / "oscd.onnx").exists()
    assert len(m["history"]) == 2
    assert {"loss", "iou", "f1", "precision", "recall"} <= set(m["history"][0]["val"])
    assert "test" in m and "bolivia" not in m and "baseline_ndwi" not in m
    assert m["val"] == m["history"][m["best_epoch"] - 1]["val"]  # val kept apart from test
    assert m["onnx"]["ok"]


def test_train_floods_end_to_end_with_baseline(tmp_path):
    out, m = _run("floods", tmp_path)
    assert (out / "floods.onnx").exists()
    assert {"test", "bolivia", "val"} <= set(m)
    for index in ("baseline_ndwi", "baseline_mndwi"):
        for split in ("val", "test", "bolivia"):
            assert {"fixed", "tuned"} <= set(m[index][split])
    assert m["round"] == 1 and "threshold" not in m  # no tuning unless asked
    assert m["onnx"]["ok"]


def test_early_stopping_stops_when_val_f1_stalls(tmp_path):
    # lr=0 -> the model never changes -> val F1 never improves after epoch 1
    _, m = _run("floods", tmp_path, "--lr", "0", "--patience", "2")
    assert len(m["history"]) < 6 and m["best_epoch"] == 1


def test_test_split_is_not_touched_while_training(tmp_path, monkeypatch):
    seen = []
    real = train.make_dataset
    monkeypatch.setattr(
        train,
        "make_dataset",
        lambda task, source, split, *a, **k: (
            seen.append(split),
            real(task, source, split, *a, **k),
        )[1],
    )
    _run("oscd", tmp_path, "--epochs", "1")
    assert seen.index("test") > seen.index("val") and seen.count("test") == 1


# ---------------------------------------------------------------------------------- ONNX


def test_onnx_matches_pytorch_for_both_models(tmp_path):
    for task in ("oscd", "floods"):
        model = train.build_model(task, 4).eval()
        path = export.export_onnx(model, task, tmp_path / f"{task}.onnx")
        inputs = [
            torch.rand(3, export.N_BANDS[task], 256, 256) for _ in export.input_names(task)
        ]  # batch != 1
        result = export.verify_onnx(model, task, path, inputs)
        assert result["ok"] and result["mask_agreement"] == 1.0
        small = [
            torch.rand(2, export.N_BANDS[task], 64, 96) for _ in inputs
        ]  # other height/width also works
        assert export.verify_onnx(model, task, path, small)["ok"]


# ------------------------------------------------------------------ round 2 additions


def test_lr_factor_warms_up_then_decays_to_floor():
    total, warm = 100, 10
    f = [train.lr_factor(i, total, warm) for i in range(total)]
    assert f[0] == pytest.approx(0.1) and f[warm - 1] == pytest.approx(1.0)
    assert f[warm] == pytest.approx(1.0, abs=1e-3)
    assert all(a >= b for a, b in zip(f[warm:], f[warm + 1 :], strict=False))  # only decreases
    assert f[-1] == pytest.approx(0.01, abs=2e-3)


def test_tune_threshold_picks_the_best_val_cutoff():
    mask = torch.cat([torch.ones(50), torch.zeros(50)]).view(1, 1, 10, 10)
    probs = torch.cat([torch.full((50,), 0.30), torch.full((50,), 0.10)]).view(1, 1, 10, 10)
    valid = torch.ones_like(mask)
    t = train.tune_threshold(probs, mask, valid)  # 0.5 would miss all water; 0.15-0.25 is perfect
    assert 0.10 <= t < 0.30
    assert train.metrics_at(probs, mask, valid, t)["f1"] == 1.0
    assert train.metrics_at(probs, mask, valid, 0.5)["f1"] == 0.0


def test_round_two_run_records_threshold_and_round_note(tmp_path):
    _, m = _run("floods", tmp_path, "--round", "2", "--tune-threshold", "--schedule", "cosine")
    assert m["round"] == 2 and m["earlier_round_test_seen"] is True
    assert "validation" in m["round_note"] and "round-1 test" in m["round_note"]
    th = m["threshold"]
    assert th["tuned_on"] == "valid" and 0 < th["value"] < 1
    assert {"valid", "test", "bolivia"} <= set(th) and "f1" in th["test"]
    assert "test" in m and m["history"][0]["lr"] > 0  # plain 0.5 numbers are kept apart
    # cosine: the last epoch's learning rate is far below the first one's
    assert m["history"][-1]["lr"] < m["history"][0]["lr"]
