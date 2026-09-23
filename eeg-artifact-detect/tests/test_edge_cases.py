"""Edge case tests: recordings that are missing something (an EOG channel,
enough samples, a working channel) or are outright corrupt. Each test
asserts the *specific* graceful-failure or documented-fallback behavior,
not just "no crash" -- see priority 6 in the project roadmap.
"""

from pathlib import Path

import mne
import numpy as np
import pytest

from eeg_autoclean.detector import detect_artifacts, load_eeg


def _make_raw(data_uv, sfreq=100.0, ch_types="eeg", ch_names=None):
    """Build a small in-memory Raw object from data given in microvolts."""
    n_channels = data_uv.shape[0]
    if ch_names is None:
        ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
    if isinstance(ch_types, str):
        ch_types = [ch_types] * n_channels

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data_uv * 1e-6, info, verbose=False)  # microvolts -> volts
    return raw


# --- No EOG channel -----------------------------------------------------


def test_no_eog_channel_skips_layer2_gracefully():
    # Already implicitly covered by the "clean signal" detector test, but
    # made explicit here as a dedicated edge case: Layer 2 must degrade to
    # a documented "skipped" state, not raise or silently return garbage.
    rng = np.random.default_rng(0)
    sfreq = 100.0
    data_uv = rng.normal(0.0, 5.0, size=(4, int(5 * sfreq)))
    raw = _make_raw(data_uv, sfreq=sfreq)

    result = detect_artifacts(raw)

    assert result["ica"] is None
    assert result["ica_components_flagged"] is None
    assert result["notes"] == ["ICA-based detection skipped: no EOG channel found in raw data."]
    # Layer 1 still runs normally even though Layer 2 was skipped.
    assert result["annotations"] is not None


# --- Very short recordings -----------------------------------------------


def test_recording_shorter_than_half_a_window_raises_clear_error():
    # 0.3s at 100 Hz is well under half of the default 1.0s window, so there
    # is no whole-or-mostly-whole window to score at all.
    rng = np.random.default_rng(1)
    sfreq = 100.0
    data_uv = rng.normal(0.0, 5.0, size=(2, int(0.3 * sfreq)))
    raw = _make_raw(data_uv, sfreq=sfreq)

    with pytest.raises(ValueError, match="too short for even one detection window"):
        detect_artifacts(raw)


def test_recording_with_one_partial_window_runs_without_crashing():
    # 0.6s is short but still over half a window, so the (documented) fallback
    # is to score it as a single partial window rather than raising.
    rng = np.random.default_rng(2)
    sfreq = 100.0
    data_uv = rng.normal(0.0, 5.0, size=(2, int(0.6 * sfreq)))
    raw = _make_raw(data_uv, sfreq=sfreq)

    result = detect_artifacts(raw)

    # One partial window per channel is scored; a mild-noise signal should
    # not trip either threshold.
    assert list(result["annotations"].description) == []


def test_short_recording_with_eog_still_fits_ica_without_crashing():
    # A few seconds is short for ICA/EOG-epoch machinery (short filters,
    # few/no blink events), but detect_artifacts should still complete and
    # return a well-formed (possibly empty) result instead of raising.
    rng = np.random.default_rng(3)
    sfreq = 100.0
    n_samples = int(3 * sfreq)
    n_eeg = 4
    data_uv = rng.normal(0.0, 5.0, size=(n_eeg + 1, n_samples))
    ch_names = [f"EEG{i:03d}" for i in range(n_eeg)] + ["EOG000"]
    raw = _make_raw(data_uv, sfreq=sfreq, ch_types=["eeg"] * n_eeg + ["eog"], ch_names=ch_names)

    result = detect_artifacts(raw, ica_n_components=2)

    assert result["ica"] is not None
    assert isinstance(result["ica_components_flagged"], list)


# --- Fully bad/disconnected channel ---------------------------------------


def test_fully_flat_channel_is_flagged_and_does_not_break_ica():
    # A channel that is flat for the *entire* recording (not just one
    # window) is the extreme case of "disconnected electrode": Layer 1 must
    # still flag every window for it, and its zero variance must not crash
    # ICA fitting in Layer 2.
    rng = np.random.default_rng(4)
    sfreq = 100.0
    n_samples = int(10 * sfreq)
    n_eeg = 5
    eeg_data = rng.normal(0.0, 5.0, size=(n_eeg, n_samples))
    eeg_data[2, :] = 0.0  # EEG002 disconnected for the whole recording
    eog_data = rng.normal(0.0, 5.0, size=(1, n_samples))
    data_uv = np.vstack([eeg_data, eog_data])
    ch_names = [f"EEG{i:03d}" for i in range(n_eeg)] + ["EOG000"]
    raw = _make_raw(data_uv, sfreq=sfreq, ch_types=["eeg"] * n_eeg + ["eog"], ch_names=ch_names)

    result = detect_artifacts(raw, ica_n_components=3)

    flat_channels = {
        ch
        for desc, chs in zip(result["annotations"].description, result["annotations"].ch_names)
        if desc == "BAD_flat"
        for ch in chs
    }
    assert "EEG002" in flat_channels
    # Every one of the 10 one-second windows for the flat channel must be flagged.
    n_flat_windows_for_ch = sum(
        1
        for desc, chs in zip(result["annotations"].description, result["annotations"].ch_names)
        if desc == "BAD_flat" and "EEG002" in chs
    )
    assert n_flat_windows_for_ch == 10
    # ICA still fits and returns a well-formed result despite the zero-variance channel.
    assert result["ica"] is not None
    assert isinstance(result["ica_components_flagged"], list)


def test_channel_pre_marked_bad_is_excluded_from_ica_fit():
    # Pre-marking a dead channel via raw.info["bads"] (standard MNE practice)
    # should exclude it from the ICA fit, while Layer 1 -- which scans all
    # EEG channels regardless of bads -- still flags it as flat.
    rng = np.random.default_rng(5)
    sfreq = 100.0
    n_samples = int(10 * sfreq)
    n_eeg = 5
    eeg_data = rng.normal(0.0, 5.0, size=(n_eeg, n_samples))
    eeg_data[2, :] = 0.0
    eog_data = rng.normal(0.0, 5.0, size=(1, n_samples))
    data_uv = np.vstack([eeg_data, eog_data])
    ch_names = [f"EEG{i:03d}" for i in range(n_eeg)] + ["EOG000"]
    raw = _make_raw(data_uv, sfreq=sfreq, ch_types=["eeg"] * n_eeg + ["eog"], ch_names=ch_names)
    raw.info["bads"] = ["EEG002"]

    result = detect_artifacts(raw, ica_n_components=3)

    assert result["ica"] is not None
    # ICA was fit on the 4 non-bad EEG channels, not 5.
    assert "EEG002" not in result["ica"].ch_names
    assert result["ica"].info["nchan"] == n_eeg - 1
    flat_channels = {
        ch
        for desc, chs in zip(result["annotations"].description, result["annotations"].ch_names)
        if desc == "BAD_flat"
        for ch in chs
    }
    assert "EEG002" in flat_channels


# --- Empty / corrupt input -------------------------------------------------


def test_load_eeg_nonexistent_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_eeg("Z:/definitely/does/not/exist.edf")


def test_load_eeg_empty_file_raises_clear_error(tmp_path):
    empty_path = tmp_path / "empty.edf"
    empty_path.write_bytes(b"")

    with pytest.raises(ValueError, match="Bad EDF file"):
        load_eeg(empty_path)


def test_load_eeg_corrupt_file_raises_clear_error(tmp_path):
    garbage_path = tmp_path / "garbage.edf"
    garbage_path.write_bytes(b"\x00\x01\x02not a real edf header" * 20)

    with pytest.raises(ValueError, match="Bad EDF file"):
        load_eeg(garbage_path)


def test_detect_artifacts_raises_clear_error_on_nan_data():
    # Simulates a corrupt/failed acquisition rather than a normal
    # disconnected channel (which is flat, not NaN). Without an explicit
    # check, NaN silently survives every threshold comparison (NaN > x is
    # always False), so a broken recording would otherwise be reported as
    # perfectly clean -- this must fail loudly instead.
    data_uv = np.full((2, 500), np.nan)
    raw = _make_raw(data_uv, sfreq=100.0)

    with pytest.raises(ValueError, match="NaN or Inf"):
        detect_artifacts(raw)


def test_detect_artifacts_raises_clear_error_on_nan_data_with_eog():
    n_eeg = 4
    data_uv = np.full((n_eeg + 1, 500), np.nan)
    ch_names = [f"EEG{i:03d}" for i in range(n_eeg)] + ["EOG000"]
    raw = _make_raw(data_uv, sfreq=100.0, ch_types=["eeg"] * n_eeg + ["eog"], ch_names=ch_names)

    with pytest.raises(ValueError, match="NaN or Inf"):
        detect_artifacts(raw, ica_n_components=2)
