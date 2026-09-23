"""Tests for eeg_autoclean.eye_state.detect_eye_state using synthetic data
with a known, injected ground truth -- independent of network access (see
scripts/validate_eye_state.py for validation against real labeled data).
"""

import mne
import numpy as np

from eeg_autoclean.eye_state import DEFAULT_BUMP_RATIO_THRESHOLD, detect_eye_state


def _make_raw(sfreq=160.0, duration_s=30.0, ch_names=("O1", "O2", "Oz", "Fz"), alpha_amp_uv=0.0, seed=0):
    """Build a Raw with flat (1/f-free) background noise on every channel,
    plus an optional 10 Hz sinusoid added only to occipital-named channels
    -- a clean stand-in for a posterior alpha rhythm.
    """
    rng = np.random.default_rng(seed)
    n_ch = len(ch_names)
    n_samples = int(duration_s * sfreq)
    data_uv = rng.normal(0.0, 5.0, size=(n_ch, n_samples))

    if alpha_amp_uv > 0:
        t = np.arange(n_samples) / sfreq
        alpha_wave = alpha_amp_uv * np.sin(2 * np.pi * 10.0 * t)
        for i, name in enumerate(ch_names):
            if name.strip(". \t").upper() in ("O1", "O2", "OZ"):
                data_uv[i] += alpha_wave

    info = mne.create_info(ch_names=list(ch_names), sfreq=sfreq, ch_types=["eeg"] * n_ch)
    return mne.io.RawArray(data_uv * 1e-6, info, verbose=False)


def test_strong_occipital_alpha_classified_eyes_closed():
    raw = _make_raw(alpha_amp_uv=20.0)

    result = detect_eye_state(raw)

    assert result["state"] == "eyes_closed"
    assert result["bump_ratio"] > DEFAULT_BUMP_RATIO_THRESHOLD
    assert set(result["channels_used"]) == {"O1", "O2", "Oz"}
    assert set(result["channel_ratios"].keys()) == {"O1", "O2", "Oz"}


def test_no_occipital_alpha_classified_eyes_open():
    raw = _make_raw(alpha_amp_uv=0.0)

    result = detect_eye_state(raw)

    assert result["state"] == "eyes_open"
    assert result["bump_ratio"] < DEFAULT_BUMP_RATIO_THRESHOLD


def test_missing_occipital_channels_skips_gracefully():
    raw = _make_raw(ch_names=("Fz", "Cz", "Pz"), alpha_amp_uv=20.0)

    result = detect_eye_state(raw)

    assert result["state"] is None
    assert result["bump_ratio"] is None
    assert result["channels_used"] == []
    assert any("no occipital channels" in note.lower() for note in result["notes"])


def test_partial_occipital_channels_still_works():
    # Only O1 present (e.g. a montage missing O2/Oz) -- should still classify
    # using whatever occipital channel is available, not raise or skip.
    raw = _make_raw(ch_names=("O1", "Fz", "Cz"), alpha_amp_uv=20.0)

    result = detect_eye_state(raw)

    assert result["state"] == "eyes_closed"
    assert result["channels_used"] == ["O1"]


def test_dot_padded_channel_names_are_matched():
    # PhysioNet's eegbci dataset pads channel names with dots, e.g. "O1..".
    raw = _make_raw(ch_names=("O1..", "O2..", "Oz..", "Fz.."), alpha_amp_uv=20.0)

    result = detect_eye_state(raw)

    assert result["state"] == "eyes_closed"
    assert set(result["channels_used"]) == {"O1..", "O2..", "Oz.."}


def test_flat_occipital_channel_excluded_not_crashed():
    raw = _make_raw(alpha_amp_uv=20.0)
    # Overwrite O1 with an exactly-flat (zero-variance) signal, simulating a
    # disconnected electrode -- flanking-band power is then exactly zero for
    # that channel, which must be excluded rather than causing a division
    # by zero.
    o1_idx = raw.ch_names.index("O1")
    raw._data[o1_idx, :] = 0.0

    result = detect_eye_state(raw)

    assert "O1" not in result["channel_ratios"]
    assert any("O1" in note and "zero" in note.lower() for note in result["notes"])
    # The other two occipital channels still carry the alpha signal, so the
    # overall classification is unaffected.
    assert result["state"] == "eyes_closed"


def test_custom_threshold_changes_classification():
    raw = _make_raw(alpha_amp_uv=20.0)
    baseline = detect_eye_state(raw)
    assert baseline["state"] == "eyes_closed"

    # An unreasonably high threshold should flip the same data to eyes_open.
    strict = detect_eye_state(raw, bump_ratio_threshold=baseline["bump_ratio"] * 10)
    assert strict["state"] == "eyes_open"
