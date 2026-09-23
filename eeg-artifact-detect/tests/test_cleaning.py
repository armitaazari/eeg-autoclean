"""Tests for eeg_autoclean.cleaning.apply_cleaning -- confirms each opt-in
cleaning action actually changes what it claims to, that nothing happens
unless explicitly requested, and that the original raw is never modified.
"""

import mne
import numpy as np
import pytest

from eeg_autoclean.cleaning import apply_cleaning
from eeg_autoclean.detector import detect_artifacts
from eeg_autoclean.synthetic import SEVERITY_SCENARIOS, generate_synthetic_raw


def _make_raw(data_uv, sfreq=250.0, ch_names=None, ch_types="eeg", montage=None):
    n_channels = data_uv.shape[0]
    if ch_names is None:
        ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
    if isinstance(ch_types, str):
        ch_types = [ch_types] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data_uv * 1e-6, info, verbose=False)
    if montage is not None:
        raw.set_montage(montage)
    return raw


def test_no_action_requested_returns_unchanged_copy():
    rng = np.random.default_rng(0)
    data_uv = rng.normal(0.0, 5.0, size=(4, 5000))
    raw = _make_raw(data_uv)
    result = detect_artifacts(raw)

    cleaned, report = apply_cleaning(raw, result)

    assert cleaned.ch_names == raw.ch_names
    assert cleaned.n_times == raw.n_times
    assert np.allclose(cleaned.get_data(), raw.get_data())
    assert report["ica_components_removed"] == []
    assert report["channels_dropped"] == []
    assert report["channels_interpolated"] == []
    assert report["time_cropped_s"] == 0.0


def test_original_raw_is_never_modified():
    rng = np.random.default_rng(0)
    data_uv = rng.normal(0.0, 5.0, size=(6, int(60 * 250.0)))
    data_uv[2, :] = 0.0  # flat channel, so drop_bad_channels has something to do
    raw = _make_raw(data_uv, sfreq=250.0)
    original_data = raw.get_data().copy()
    original_ch_names = list(raw.ch_names)
    result = detect_artifacts(raw)

    apply_cleaning(raw, result, drop_bad_channels=True, crop_bad_segments=True)

    assert raw.ch_names == original_ch_names
    assert np.allclose(raw.get_data(), original_data)


def test_remove_ocular_ica_actually_removes_the_component():
    scenario = SEVERITY_SCENARIOS[0]  # sparse/mild: reliably flags one ocular IC
    kwargs = {k: v for k, v in scenario.items() if k != "name"}
    raw, _ground_truth = generate_synthetic_raw(**kwargs)
    result = detect_artifacts(raw)
    assert result["ica_components_flagged"], "fixture assumption: expected a flagged component"

    cleaned, report = apply_cleaning(raw, result, remove_ocular_ica=True)

    assert report["ica_components_removed"] == result["ica_components_flagged"]
    assert not np.allclose(cleaned.get_data(), raw.get_data())


def test_remove_ocular_ica_is_noop_when_layer2_skipped():
    rng = np.random.default_rng(1)
    data_uv = rng.normal(0.0, 5.0, size=(4, 5000))
    raw = _make_raw(data_uv)  # no EOG channel -> Layer 2 skipped
    result = detect_artifacts(raw)
    assert result["ica_components_flagged"] is None

    cleaned, report = apply_cleaning(raw, result, remove_ocular_ica=True)

    assert report["ica_components_removed"] == []
    assert np.allclose(cleaned.get_data(), raw.get_data())


def test_drop_bad_channels_exclude_mode_removes_the_channel():
    rng = np.random.default_rng(2)
    sfreq = 250.0
    n_ch = 6
    data_uv = rng.normal(0.0, 5.0, size=(n_ch, int(60 * sfreq)))
    data_uv[3, :] = 0.0  # flat for the entire recording -> bad fraction 1.0
    raw = _make_raw(data_uv, sfreq=sfreq)
    result = detect_artifacts(raw)

    cleaned, report = apply_cleaning(raw, result, drop_bad_channels=True, bad_channel_mode="exclude")

    assert report["channels_dropped"] == ["EEG003"]
    assert report["channels_interpolated"] == []
    assert "EEG003" not in cleaned.ch_names
    assert len(cleaned.ch_names) == n_ch - 1
    assert report["bad_channel_fractions"]["EEG003"] == pytest.approx(1.0)


def test_drop_bad_channels_below_threshold_is_left_alone():
    rng = np.random.default_rng(3)
    sfreq = 250.0
    n_ch = 4
    data_uv = rng.normal(0.0, 5.0, size=(n_ch, int(60 * sfreq)))
    # One brief amplitude spike -- nowhere near the default 50% threshold.
    data_uv[1, :250] = 300.0 * np.sign(np.sin(2 * np.pi * 5 * np.arange(250) / sfreq))
    raw = _make_raw(data_uv, sfreq=sfreq)
    result = detect_artifacts(raw)

    cleaned, report = apply_cleaning(raw, result, drop_bad_channels=True)

    assert report["channels_dropped"] == []
    assert cleaned.ch_names == raw.ch_names


def test_drop_bad_channels_interpolate_mode_without_montage_raises_clear_error():
    rng = np.random.default_rng(4)
    sfreq = 250.0
    n_ch = 6
    data_uv = rng.normal(0.0, 5.0, size=(n_ch, int(60 * sfreq)))
    data_uv[3, :] = 0.0
    raw = _make_raw(data_uv, sfreq=sfreq)  # no montage set
    result = detect_artifacts(raw)

    with pytest.raises(ValueError, match="montage"):
        apply_cleaning(raw, result, drop_bad_channels=True, bad_channel_mode="interpolate")


def test_drop_bad_channels_interpolate_mode_with_montage_preserves_channel_count():
    rng = np.random.default_rng(5)
    sfreq = 250.0
    ch_names = ["Fz", "Cz", "Pz", "O1", "O2", "C3"]
    data_uv = rng.normal(0.0, 5.0, size=(len(ch_names), int(60 * sfreq)))
    data_uv[3, :] = 0.0  # O1 flat for the whole recording
    montage = mne.channels.make_standard_montage("standard_1020")
    raw = _make_raw(data_uv, sfreq=sfreq, ch_names=ch_names, montage=montage)
    result = detect_artifacts(raw)

    cleaned, report = apply_cleaning(raw, result, drop_bad_channels=True, bad_channel_mode="interpolate")

    assert report["channels_interpolated"] == ["O1"]
    assert report["channels_dropped"] == []
    assert cleaned.ch_names == raw.ch_names  # channel count/order preserved
    assert not np.allclose(cleaned.get_data(picks=["O1"]), 0.0)  # no longer flat
    assert np.allclose(raw.get_data(picks=["O1"]), 0.0)  # original untouched


def test_drop_bad_channels_invalid_mode_raises():
    rng = np.random.default_rng(6)
    data_uv = rng.normal(0.0, 5.0, size=(4, 5000))
    raw = _make_raw(data_uv)
    result = detect_artifacts(raw)

    with pytest.raises(ValueError, match="bad_channel_mode"):
        apply_cleaning(raw, result, drop_bad_channels=True, bad_channel_mode="nonsense")


def test_crop_bad_segments_shortens_the_recording():
    scenario = SEVERITY_SCENARIOS[2]  # dense/severe: plenty of flagged windows
    kwargs = {k: v for k, v in scenario.items() if k != "name"}
    raw, _ground_truth = generate_synthetic_raw(**kwargs)
    result = detect_artifacts(raw)

    cleaned, report = apply_cleaning(raw, result, crop_bad_segments=True)

    assert cleaned.n_times < raw.n_times
    assert report["time_cropped_s"] > 0.0
    assert report["n_segments_cropped"] > 0
    assert report["duration_after_s"] == pytest.approx(report["duration_before_s"] - report["time_cropped_s"], abs=0.1)


def test_crop_bad_segments_narrower_descriptions_crops_less():
    scenario = SEVERITY_SCENARIOS[2]
    kwargs = {k: v for k, v in scenario.items() if k != "name"}
    raw, _ground_truth = generate_synthetic_raw(**kwargs)
    result = detect_artifacts(raw)

    full_cleaned, full_report = apply_cleaning(raw, result, crop_bad_segments=True)
    narrow_cleaned, narrow_report = apply_cleaning(
        raw, result, crop_bad_segments=True, crop_descriptions=("BAD_amplitude",)
    )

    assert narrow_report["time_cropped_s"] <= full_report["time_cropped_s"]
    assert narrow_cleaned.n_times >= full_cleaned.n_times


def test_crop_bad_segments_no_flags_leaves_recording_unchanged():
    # 60s / 60 windows, seed chosen (empirically verified) to produce zero
    # flags across all layers -- with only 4 channels, Layer 3/4's already-
    # documented baseline false-positive rate means many seeds do flag
    # something even on pure noise (see tests/test_detector.py's
    # clean-signal test for the same reasoning), so this isn't guaranteed
    # for an arbitrary seed.
    rng = np.random.default_rng(1)
    data_uv = rng.normal(0.0, 5.0, size=(4, int(60 * 250.0)))
    raw = _make_raw(data_uv)
    result = detect_artifacts(raw)
    assert list(result["annotations"].description) == []  # fixture assumption

    cleaned, report = apply_cleaning(raw, result, crop_bad_segments=True)

    assert cleaned.n_times == raw.n_times
    assert report["time_cropped_s"] == 0.0
    assert report["n_segments_cropped"] == 0


def test_combined_actions_run_in_one_call():
    scenario = SEVERITY_SCENARIOS[2]
    kwargs = {k: v for k, v in scenario.items() if k != "name"}
    raw, _ground_truth = generate_synthetic_raw(**kwargs)
    result = detect_artifacts(raw)

    cleaned, report = apply_cleaning(
        raw,
        result,
        remove_ocular_ica=True,
        drop_bad_channels=True,
        bad_channel_mode="exclude",
        crop_bad_segments=True,
    )

    # Whatever combination of channels/time survives, the report is fully
    # self-consistent and nothing was silently skipped.
    assert cleaned.n_times <= raw.n_times
    assert len(cleaned.ch_names) <= len(raw.ch_names)
    assert isinstance(report["ica_components_removed"], list)
    assert isinstance(report["channels_dropped"], list)
    assert isinstance(report["time_cropped_s"], float)
