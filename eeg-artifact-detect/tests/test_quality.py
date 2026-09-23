"""Tests for eeg_autoclean.quality.compute_quality_score."""

import mne
import numpy as np
import pytest

from eeg_autoclean.detector import detect_artifacts
from eeg_autoclean.quality import DEFAULT_LAYER_WEIGHTS, compute_quality_score


def _make_raw(data_uv, sfreq=250.0, ch_names=None):
    n_channels = data_uv.shape[0]
    if ch_names is None:
        ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * n_channels)
    return mne.io.RawArray(data_uv * 1e-6, info, verbose=False)


def _clean_raw(seed=0, n_ch=8, duration_s=60.0, sfreq=250.0):
    rng = np.random.default_rng(seed)
    data_uv = rng.normal(0.0, 5.0, size=(n_ch, int(duration_s * sfreq)))
    return _make_raw(data_uv, sfreq=sfreq)


def _heavily_contaminated_raw(seed=0, n_ch=8, duration_s=60.0, sfreq=250.0):
    # Half the channels are completely flat/disconnected for the whole
    # recording, and the other half swing wildly on every other window --
    # both are intermittent-relative-to-nothing-else, gross, unambiguous
    # contamination that Layer 1's adaptive threshold reliably catches
    # (unlike a *continuous* artifact spanning 100% of a channel's own
    # windows, which an adaptive per-channel baseline would normalize away
    # -- see eeg_autoclean.muscle/cardiac's own documented limitations).
    rng = np.random.default_rng(seed)
    n_samples = int(duration_s * sfreq)
    data_uv = rng.normal(0.0, 5.0, size=(n_ch, n_samples))
    half = n_ch // 2
    data_uv[:half, :] = 0.0

    window_samples = int(sfreq)
    n_windows = n_samples // window_samples
    for ch in range(half, n_ch):
        for w in range(0, n_windows, 2):
            start, stop = w * window_samples, (w + 1) * window_samples
            t = np.arange(stop - start) / sfreq
            data_uv[ch, start:stop] = 200.0 * np.sign(np.sin(2 * np.pi * 5 * t))
    return _make_raw(data_uv, sfreq=sfreq)


def test_clean_recording_scores_high():
    raw = _clean_raw()
    result = detect_artifacts(raw)
    summary = compute_quality_score(raw, result)

    assert summary["score"] > 95.0
    assert 0.0 <= summary["score"] <= 100.0


def test_heavily_contaminated_recording_scores_much_lower_than_clean():
    clean_raw = _clean_raw(seed=0)
    clean_result = detect_artifacts(clean_raw)
    clean_summary = compute_quality_score(clean_raw, clean_result)

    dirty_raw = _heavily_contaminated_raw(seed=1)
    dirty_result = detect_artifacts(dirty_raw)
    dirty_summary = compute_quality_score(dirty_raw, dirty_result)

    assert dirty_summary["score"] < clean_summary["score"] - 10.0
    assert dirty_summary["breakdown"]["amplitude_flat"]["badness_fraction"] > 0.3


def test_breakdown_has_all_four_layers_with_weights():
    raw = _clean_raw()
    result = detect_artifacts(raw)
    summary = compute_quality_score(raw, result)

    assert set(summary["breakdown"].keys()) == {"amplitude_flat", "ocular", "muscle", "cardiac"}
    for entry in summary["breakdown"].values():
        assert "badness_fraction" in entry
        assert "skipped" in entry
        assert entry["weight"] == pytest.approx(0.25)


def test_skipped_layer_weight_is_redistributed_not_free():
    # No EOG channel -> Layer 2 is skipped. Its badness_fraction must be
    # None (nothing was measured), not silently treated as 0 (which would
    # inflate the score for having less to measure, not for being cleaner).
    raw = _clean_raw(n_ch=4)
    result = detect_artifacts(raw)
    summary = compute_quality_score(raw, result)

    assert summary["breakdown"]["ocular"]["skipped"] is True
    assert summary["breakdown"]["ocular"]["badness_fraction"] is None
    # Still a well-formed score computed from the 3 layers that did run.
    assert 0.0 <= summary["score"] <= 100.0


def test_custom_weights_change_the_score():
    dirty_raw = _heavily_contaminated_raw(seed=2)
    result = detect_artifacts(dirty_raw)

    default_summary = compute_quality_score(dirty_raw, result)
    # Zero out amplitude_flat's weight entirely (renormalized among the
    # rest) -- since that's the dominant contaminated layer here, the score
    # should rise noticeably once it stops counting.
    custom_weights = dict(DEFAULT_LAYER_WEIGHTS)
    custom_weights["amplitude_flat"] = 0.0
    custom_summary = compute_quality_score(dirty_raw, result, weights=custom_weights)

    assert custom_summary["score"] > default_summary["score"]


def test_raises_without_eeg_channels():
    # detect_artifacts() itself already requires EEG channels (raises
    # first), so to exercise compute_quality_score's own guard directly, a
    # minimal placeholder result is passed instead of a real one -- a
    # legitimate use since compute_quality_score takes a plain dict, not
    # necessarily one that came from detect_artifacts().
    rng = np.random.default_rng(0)
    data_uv = rng.normal(0.0, 5.0, size=(1, 2000))
    info = mne.create_info(ch_names=["MISC000"], sfreq=250.0, ch_types=["misc"])
    raw = mne.io.RawArray(data_uv * 1e-6, info, verbose=False)
    placeholder_result = {
        "annotations": mne.Annotations(onset=[], duration=[], description=[]),
        "ica": None,
        "ica_components_flagged": None,
        "notes": [],
    }

    with pytest.raises(ValueError, match="No EEG channels"):
        compute_quality_score(raw, placeholder_result)
