import numpy as np
import pytest

from impactlens.data.preprocess import BAND_CLIPS, dn_to_reflectance, normalize


def test_each_band_hits_one_exactly_at_its_own_ceiling():
    refl = np.array(BAND_CLIPS, dtype=np.float32).reshape(4, 1, 1) * np.ones((4, 2, 2))
    assert np.allclose(normalize(refl), 1.0)


def test_nir_has_more_headroom_than_visible_bands():
    # Reflectance 0.55: saturated in blue/green/red, but NIR (ceiling 0.6) still has resolution.
    refl = np.full((4, 3, 3), 0.55, dtype=np.float32)
    out = normalize(refl)
    assert np.allclose(out[:3], 1.0)
    assert out[3].max() == pytest.approx(0.55 / 0.6, rel=1e-6) and out[3].max() < 1.0
    # ...and two bright vegetation NIR values stay distinguishable (they would both be 1.0
    # with the old single 0.4 ceiling).
    nir = np.stack([np.full((2, 2), v, np.float32) for v in (0.45, 0.55)])
    both = normalize(np.concatenate([np.zeros((3, 2, 2), np.float32), nir[:1]]))[3]
    other = normalize(np.concatenate([np.zeros((3, 2, 2), np.float32), nir[1:]]))[3]
    assert other.mean() > both.mean()


def test_range_monotonic_and_negative_values_clip_to_zero():
    x = np.linspace(-0.1, 2.0, 50, dtype=np.float32)
    refl = np.tile(x[None, :, None], (4, 1, 1))
    out = normalize(refl)
    assert out.min() == 0.0 and out.max() == 1.0 and out.dtype == np.float32
    assert (np.diff(out, axis=1) >= 0).all()


def test_batched_input_and_bad_shape():
    batch = np.random.default_rng(0).uniform(0, 1, (5, 4, 8, 8)).astype(np.float32)
    assert np.allclose(normalize(batch)[2], normalize(batch[2]))
    with pytest.raises(ValueError):
        normalize(np.zeros((3, 8, 8), dtype=np.float32))


def test_dn_to_reflectance_scale_offset_and_nodata():
    dn = np.array([0, 1000, 3000], dtype=np.uint16)
    assert dn_to_reflectance(dn).tolist() == pytest.approx([0.0, 0.1, 0.3])
    shifted = dn_to_reflectance(dn, offset=-0.1)
    assert shifted.tolist() == pytest.approx([0.0, 0.0, 0.2])  # DN 0 stays nodata, never negative
