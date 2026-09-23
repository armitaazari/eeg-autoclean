"""Unit tests for eeg_autoclean.cardiac (Layer 4: cardiac/pulse detection),
independent of the full synthetic-battery evaluation in
tests/test_synthetic_evaluation.py.
"""

import mne
import numpy as np
import pytest

from eeg_autoclean.cardiac import DEFAULT_CARDIAC_WINDOW_DURATION, _scan_cardiac_windows
from eeg_autoclean.detector import detect_artifacts


def _make_raw(data_uv, sfreq=250.0, ch_names=None, ch_types="eeg"):
    n_channels = data_uv.shape[0]
    if ch_names is None:
        ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
    if isinstance(ch_types, str):
        ch_types = [ch_types] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    return mne.io.RawArray(data_uv * 1e-6, info, verbose=False)


def _inject_pulse_train(data_uv, ch_idx, start, stop, sfreq, amplitude_uv, bpm=75.0, pulse_width_s=0.025):
    t = np.arange(stop - start) / sfreq
    period = 60.0 / bpm
    pulse_times = np.arange(0.0, (stop - start) / sfreq, period)
    train = np.zeros(stop - start)
    for pt in pulse_times:
        train += amplitude_uv * np.exp(-0.5 * ((t - pt) / pulse_width_s) ** 2)
    data_uv[ch_idx, start:stop] += train


def test_periodic_pulse_train_is_flagged():
    # Spans 2 consecutive cardiac windows: the default persistence_n=2 (see
    # DEFAULT_CARDIAC_PERSISTENCE_N) requires a flag to persist across at
    # least that many consecutive windows to be confirmed, matching the
    # validated synthetic battery's ground truth shape (real cardiac
    # contamination is sustained, not a single 6s burst -- see module
    # docstring).
    rng = np.random.default_rng(0)
    sfreq = 250.0
    n_ch = 4
    duration_s = 60
    n_samples = int(duration_s * sfreq)
    data_uv = rng.normal(0.0, 5.0, size=(n_ch, n_samples))

    start, stop = int(24 * sfreq), int(36 * sfreq)  # two consecutive 6.0s cardiac windows
    _inject_pulse_train(data_uv, ch_idx=1, start=start, stop=stop, sfreq=sfreq, amplitude_uv=20.0)

    raw = _make_raw(data_uv, sfreq=sfreq)
    annotations, used_ecg, note = _scan_cardiac_windows(raw)

    assert used_ecg is False
    assert note is None
    cardiac_cells = {
        (ch, round(onset / DEFAULT_CARDIAC_WINDOW_DURATION))
        for onset, desc, chs in zip(annotations.onset, annotations.description, annotations.ch_names)
        for ch in chs
        if desc == "BAD_cardiac"
    }
    assert ("EEG001", 4) in cardiac_cells  # window index 24/6 = 4
    assert ("EEG001", 5) in cardiac_cells  # window index 30/6 = 5


def test_persistence_filter_drops_isolated_flag_but_keeps_sustained_one():
    rng = np.random.default_rng(9)
    sfreq = 250.0
    n_ch = 4
    duration_s = 90  # 15 cardiac (6.0s) windows, plenty of room to separate events
    n_samples = int(duration_s * sfreq)
    data_uv = rng.normal(0.0, 5.0, size=(n_ch, n_samples))

    # Isolated: one cardiac window (index 2 -> 12-18s) on EEG001, clean on
    # both sides.
    isolated_start, isolated_stop = int(12 * sfreq), int(18 * sfreq)
    _inject_pulse_train(data_uv, ch_idx=1, start=isolated_start, stop=isolated_stop, sfreq=sfreq, amplitude_uv=20.0)

    # Sustained: a continuous pulse train spanning 2 consecutive cardiac
    # windows (indices 6-7 -> 36-48s) on EEG002.
    sustained_start, sustained_stop = int(36 * sfreq), int(48 * sfreq)
    _inject_pulse_train(data_uv, ch_idx=2, start=sustained_start, stop=sustained_stop, sfreq=sfreq, amplitude_uv=20.0)

    raw = _make_raw(data_uv, sfreq=sfreq)

    def _cells(annotations):
        return {
            (ch, round(onset / DEFAULT_CARDIAC_WINDOW_DURATION))
            for onset, desc, chs in zip(annotations.onset, annotations.description, annotations.ch_names)
            for ch in chs
            if desc == "BAD_cardiac"
        }

    no_persistence, _, _ = _scan_cardiac_windows(raw, persistence_n=1)
    no_persistence_cells = _cells(no_persistence)
    assert ("EEG001", 2) in no_persistence_cells  # isolated flag present without filtering
    assert ("EEG002", 6) in no_persistence_cells
    assert ("EEG002", 7) in no_persistence_cells

    with_persistence, _, _ = _scan_cardiac_windows(raw, persistence_n=2)
    with_persistence_cells = _cells(with_persistence)
    assert ("EEG001", 2) not in with_persistence_cells  # isolated flag dropped
    assert ("EEG002", 6) in with_persistence_cells  # sustained run of 2 survives
    assert ("EEG002", 7) in with_persistence_cells


def test_clean_signal_with_no_periodicity_is_not_flagged():
    rng = np.random.default_rng(1)
    sfreq = 250.0
    n_ch = 4
    n_samples = int(60 * sfreq)
    data_uv = rng.normal(0.0, 5.0, size=(n_ch, n_samples))  # pure background, no pulse

    raw = _make_raw(data_uv, sfreq=sfreq)
    annotations, _, _ = _scan_cardiac_windows(raw)

    assert list(annotations.description) == []


def test_detect_artifacts_includes_cardiac_annotations():
    # Spans 2 consecutive cardiac windows -- see test_periodic_pulse_train_is_flagged.
    rng = np.random.default_rng(2)
    sfreq = 250.0
    n_ch = 4
    n_samples = int(60 * sfreq)
    data_uv = rng.normal(0.0, 5.0, size=(n_ch, n_samples))

    start, stop = int(24 * sfreq), int(36 * sfreq)
    _inject_pulse_train(data_uv, ch_idx=2, start=start, stop=stop, sfreq=sfreq, amplitude_uv=20.0)

    raw = _make_raw(data_uv, sfreq=sfreq)
    result = detect_artifacts(raw)

    assert "BAD_cardiac" in set(result["annotations"].description)
    assert result["cardiac_used_ecg_channel"] is False


def test_ecg_channel_cross_correlation_path():
    # Spans 2 consecutive cardiac windows -- see test_periodic_pulse_train_is_flagged.
    rng = np.random.default_rng(3)
    sfreq = 250.0
    n_ch = 3
    n_samples = int(60 * sfreq)
    data_uv = rng.normal(0.0, 5.0, size=(n_ch + 1, n_samples))  # last row = ECG

    start, stop = int(24 * sfreq), int(36 * sfreq)
    t = np.arange(stop - start) / sfreq
    period = 60.0 / 75.0
    pulse_times = np.arange(0.0, (stop - start) / sfreq, period)
    pulse = np.zeros(stop - start)
    for pt in pulse_times:
        pulse += np.exp(-0.5 * ((t - pt) / 0.025) ** 2)
    data_uv[n_ch, start:stop] += pulse * 500.0  # strong, clean ECG channel signal
    data_uv[1, start:stop] += pulse * 15.0  # weak, same-phase leakage into one EEG channel

    ch_names = [f"EEG{i:03d}" for i in range(n_ch)] + ["ECG000"]
    ch_types = ["eeg"] * n_ch + ["ecg"]
    raw = _make_raw(data_uv, sfreq=sfreq, ch_names=ch_names, ch_types=ch_types)

    annotations, used_ecg, note = _scan_cardiac_windows(raw)

    assert used_ecg is True
    assert note is None
    cardiac_cells = {
        (ch, round(onset / DEFAULT_CARDIAC_WINDOW_DURATION))
        for onset, desc, chs in zip(annotations.onset, annotations.description, annotations.ch_names)
        for ch in chs
        if desc == "BAD_cardiac"
    }
    assert ("EEG001", 4) in cardiac_cells


def test_recording_too_short_for_cardiac_window_skips_gracefully():
    # 2s recording: long enough for Layer 1/3's 1.0s windows but under half
    # of Layer 4's 6.0s window (the minimum _compute_window_bounds needs for
    # even one partial window) -- this must degrade gracefully (skip, with
    # a note), not raise and abort the whole detect_artifacts() call.
    rng = np.random.default_rng(4)
    sfreq = 100.0
    data_uv = rng.normal(0.0, 5.0, size=(2, int(2 * sfreq)))
    raw = _make_raw(data_uv, sfreq=sfreq)

    annotations, used_ecg, note = _scan_cardiac_windows(raw)
    assert list(annotations.description) == []
    assert used_ecg is False
    assert note is not None and "shorter" in note.lower()

    result = detect_artifacts(raw)
    assert "BAD_cardiac" not in set(result["annotations"].description)
    assert any("cardiac" in n.lower() for n in result["notes"])


def test_cardiac_scan_raises_without_eeg_channels():
    rng = np.random.default_rng(5)
    sfreq = 100.0
    data_uv = rng.normal(0.0, 5.0, size=(1, int(10 * sfreq)))
    info = mne.create_info(ch_names=["MISC000"], sfreq=sfreq, ch_types=["misc"])
    raw = mne.io.RawArray(data_uv * 1e-6, info, verbose=False)

    with pytest.raises(ValueError, match="No EEG channels"):
        _scan_cardiac_windows(raw)


def test_cardiac_scan_raises_on_nan_data():
    data_uv = np.full((2, 2000), np.nan)
    raw = _make_raw(data_uv, sfreq=100.0)

    with pytest.raises(ValueError, match="NaN or Inf"):
        _scan_cardiac_windows(raw)


def test_fully_flat_channel_excluded_not_crashed():
    rng = np.random.default_rng(6)
    sfreq = 250.0
    n_ch = 3
    n_samples = int(60 * sfreq)
    data_uv = rng.normal(0.0, 5.0, size=(n_ch, n_samples))
    data_uv[1, :] = 0.0  # flat/disconnected for the whole recording

    raw = _make_raw(data_uv, sfreq=sfreq)
    annotations, _, _ = _scan_cardiac_windows(raw)

    flagged_channels = {ch for chs in annotations.ch_names for ch in chs}
    assert "EEG001" not in flagged_channels
