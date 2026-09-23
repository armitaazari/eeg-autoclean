"""Unit tests for eeg_autoclean.muscle (Layer 3: muscle/EMG detection),
independent of the full synthetic-battery evaluation in
tests/test_synthetic_evaluation.py.
"""

import mne
import numpy as np
import pytest

from eeg_autoclean.detector import detect_artifacts
from eeg_autoclean.muscle import MUSCLE_BAND, _scan_muscle_windows


def _make_raw(data_uv, sfreq=250.0, ch_names=None):
    n_channels = data_uv.shape[0]
    if ch_names is None:
        ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * n_channels)
    return mne.io.RawArray(data_uv * 1e-6, info, verbose=False)


def _inject_muscle_burst(data_uv, ch_idx, start, stop, sfreq, amplitude_uv, band=MUSCLE_BAND, n_tones=8, rng=None):
    rng = rng or np.random.default_rng(0)
    t = np.arange(stop - start) / sfreq
    freqs = np.linspace(band[0], band[1], n_tones, endpoint=False) + (band[1] - band[0]) / (2 * n_tones)
    burst = np.zeros(stop - start)
    for f in freqs:
        burst += np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    burst *= amplitude_uv / np.max(np.abs(burst))
    data_uv[ch_idx, start:stop] += burst


def test_strong_muscle_burst_is_flagged():
    rng = np.random.default_rng(0)
    sfreq = 250.0
    n_ch = 4
    duration_s = 30
    n_samples = int(duration_s * sfreq)
    data_uv = rng.normal(0.0, 5.0, size=(n_ch, n_samples))

    start, stop = int(10 * sfreq), int(11 * sfreq)
    _inject_muscle_burst(data_uv, ch_idx=1, start=start, stop=stop, sfreq=sfreq, amplitude_uv=60.0, rng=rng)

    raw = _make_raw(data_uv, sfreq=sfreq)
    annotations = _scan_muscle_windows(raw)

    muscle_cells = {
        (ch, round(onset / 1.0))
        for onset, desc, chs in zip(annotations.onset, annotations.description, annotations.ch_names)
        for ch in chs
        if desc == "BAD_muscle"
    }
    assert ("EEG001", 10) in muscle_cells


def test_persistence_filter_drops_isolated_flag_but_keeps_sustained_one():
    rng = np.random.default_rng(9)
    sfreq = 250.0
    n_ch = 4
    duration_s = 30
    n_samples = int(duration_s * sfreq)
    data_uv = rng.normal(0.0, 5.0, size=(n_ch, n_samples))

    # Isolated: a single 1s window burst on EEG001, with clean windows on
    # both sides -- nothing for a persistence_n=2 requirement to group with.
    isolated_start, isolated_stop = int(5 * sfreq), int(6 * sfreq)
    _inject_muscle_burst(data_uv, ch_idx=1, start=isolated_start, stop=isolated_stop, sfreq=sfreq, amplitude_uv=60.0, rng=rng)

    # Sustained: a continuous burst spanning 3 consecutive 1s windows on
    # EEG002.
    sustained_start, sustained_stop = int(15 * sfreq), int(18 * sfreq)
    _inject_muscle_burst(
        data_uv, ch_idx=2, start=sustained_start, stop=sustained_stop, sfreq=sfreq, amplitude_uv=60.0, rng=rng
    )

    raw = _make_raw(data_uv, sfreq=sfreq)

    def _cells(annotations):
        return {
            (ch, round(onset / 1.0))
            for onset, desc, chs in zip(annotations.onset, annotations.description, annotations.ch_names)
            for ch in chs
            if desc == "BAD_muscle"
        }

    no_persistence = _cells(_scan_muscle_windows(raw, persistence_n=1))
    assert ("EEG001", 5) in no_persistence  # isolated flag present without filtering
    assert ("EEG002", 15) in no_persistence
    assert ("EEG002", 16) in no_persistence
    assert ("EEG002", 17) in no_persistence

    with_persistence = _cells(_scan_muscle_windows(raw, persistence_n=2))
    assert ("EEG001", 5) not in with_persistence  # isolated flag dropped
    assert ("EEG002", 15) in with_persistence  # sustained run of 3 survives
    assert ("EEG002", 16) in with_persistence
    assert ("EEG002", 17) in with_persistence


def test_flat_no_muscle_signal_is_not_flagged():
    rng = np.random.default_rng(1)
    sfreq = 250.0
    n_ch = 4
    n_samples = int(30 * sfreq)
    data_uv = rng.normal(0.0, 5.0, size=(n_ch, n_samples))  # pure background, no burst

    raw = _make_raw(data_uv, sfreq=sfreq)
    annotations = _scan_muscle_windows(raw)

    assert list(annotations.description) == []


def test_detect_artifacts_includes_muscle_annotations_alongside_layer1():
    rng = np.random.default_rng(2)
    sfreq = 250.0
    n_ch = 4
    n_samples = int(30 * sfreq)
    data_uv = rng.normal(0.0, 5.0, size=(n_ch, n_samples))

    start, stop = int(10 * sfreq), int(11 * sfreq)
    _inject_muscle_burst(data_uv, ch_idx=2, start=start, stop=stop, sfreq=sfreq, amplitude_uv=60.0, rng=rng)

    raw = _make_raw(data_uv, sfreq=sfreq)
    result = detect_artifacts(raw)

    descriptions = set(result["annotations"].description)
    assert "BAD_muscle" in descriptions


def test_muscle_scan_raises_without_eeg_channels():
    rng = np.random.default_rng(3)
    sfreq = 100.0
    n_samples = int(5 * sfreq)
    data_uv = rng.normal(0.0, 5.0, size=(1, n_samples))
    info = mne.create_info(ch_names=["MISC000"], sfreq=sfreq, ch_types=["misc"])
    raw = mne.io.RawArray(data_uv * 1e-6, info, verbose=False)

    with pytest.raises(ValueError, match="No EEG channels"):
        _scan_muscle_windows(raw)


def test_muscle_scan_raises_on_too_short_recording():
    rng = np.random.default_rng(4)
    sfreq = 100.0
    data_uv = rng.normal(0.0, 5.0, size=(2, int(0.3 * sfreq)))
    raw = _make_raw(data_uv, sfreq=sfreq)

    with pytest.raises(ValueError, match="too short for even one detection window"):
        _scan_muscle_windows(raw)


def test_muscle_scan_raises_on_nan_data():
    data_uv = np.full((2, 500), np.nan)
    raw = _make_raw(data_uv, sfreq=100.0)

    with pytest.raises(ValueError, match="NaN or Inf"):
        _scan_muscle_windows(raw)


def test_fully_flat_channel_excluded_not_crashed():
    # A channel that's flat for the whole recording has zero power in every
    # band -- the ratio is undefined (0/0), and must be excluded rather than
    # raising or silently flagging every window.
    rng = np.random.default_rng(5)
    sfreq = 250.0
    n_ch = 3
    n_samples = int(30 * sfreq)
    data_uv = rng.normal(0.0, 5.0, size=(n_ch, n_samples))
    data_uv[1, :] = 0.0

    raw = _make_raw(data_uv, sfreq=sfreq)
    annotations = _scan_muscle_windows(raw)

    flagged_channels = {ch for chs in annotations.ch_names for ch in chs}
    assert "EEG001" not in flagged_channels
