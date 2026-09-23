"""Tests for eeg_autoclean.consensus.run_consensus.

Uses small (6-channel, 20-30s) fixtures to keep AutoReject's cross-
validated threshold search fast -- these tests care about run_consensus's
own logic (grid alignment, categorization, error handling), not about
AutoReject's internal accuracy, which is out of this project's scope to
validate.
"""

import mne
import numpy as np
import pytest

autoreject = pytest.importorskip("autoreject", reason="consensus mode requires the optional autoreject dependency")

import eeg_autoclean.consensus as consensus
from eeg_autoclean.consensus import DISAGREE, run_consensus


def _make_raw(data_uv, sfreq=250.0, with_montage=True):
    n_ch = data_uv.shape[0]
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_names = montage.ch_names[:n_ch]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * n_ch)
    raw = mne.io.RawArray(data_uv * 1e-6, info, verbose=False)
    if with_montage:
        raw.set_montage(montage)
    return raw


def test_raises_without_montage():
    rng = np.random.default_rng(0)
    data_uv = rng.normal(0.0, 5.0, size=(6, int(20 * 250.0)))
    raw = _make_raw(data_uv, with_montage=False)

    with pytest.raises(ValueError, match="montage"):
        run_consensus(raw)


def test_raises_without_eeg_channels():
    rng = np.random.default_rng(0)
    data_uv = rng.normal(0.0, 5.0, size=(1, int(20 * 250.0)))
    info = mne.create_info(ch_names=["MISC000"], sfreq=250.0, ch_types=["misc"])
    raw = mne.io.RawArray(data_uv * 1e-6, info, verbose=False)

    with pytest.raises(ValueError, match="No EEG channels"):
        run_consensus(raw)


def test_invalid_methods_raises():
    rng = np.random.default_rng(0)
    data_uv = rng.normal(0.0, 5.0, size=(6, int(20 * 250.0)))
    raw = _make_raw(data_uv)

    with pytest.raises(ValueError, match="methods"):
        run_consensus(raw, methods=("ours", "pyprep"))


def test_result_structure_and_counts_are_self_consistent():
    rng = np.random.default_rng(1)
    data_uv = rng.normal(0.0, 5.0, size=(6, int(20 * 250.0)))
    raw = _make_raw(data_uv)

    result = run_consensus(raw)

    assert set(result["counts"].keys()) == {"AGREE_BAD", "AGREE_CLEAN", "DISAGREE", "total_cells"}
    assert result["counts"]["AGREE_BAD"] + result["counts"]["AGREE_CLEAN"] + result["counts"]["DISAGREE"] == (
        result["counts"]["total_cells"]
    )
    assert result["counts"]["total_cells"] == len(result["channels"]) * result["n_windows"]
    assert sum(result["fractions"].values()) == pytest.approx(1.0, abs=1e-9)
    assert len(result["disagreements"]) == result["counts"]["DISAGREE"]
    for entry in result["disagreements"]:
        assert entry["channel"] in result["channels"]
        assert 0 <= entry["window"] < result["n_windows"]
        assert entry["flagged_by"] in ("ours", "autoreject")


def test_obvious_amplitude_spike_is_flagged_by_at_least_our_method():
    rng = np.random.default_rng(2)
    sfreq = 250.0
    n_ch = 6
    n_samples = int(30 * sfreq)
    data_uv = rng.normal(0.0, 5.0, size=(n_ch, n_samples))
    spike_start, spike_stop = int(10 * sfreq), int(11 * sfreq)
    t = np.arange(spike_stop - spike_start) / sfreq
    data_uv[2, spike_start:spike_stop] = 300.0 * np.sign(np.sin(2 * np.pi * 5 * t))
    raw = _make_raw(data_uv, sfreq=sfreq)

    result = run_consensus(raw)
    ch_name = result["channels"][2]

    cell_flagged_by_ours = any(
        d["channel"] == ch_name and d["window"] == 10 and d["flagged_by"] == "ours" for d in result["disagreements"]
    )
    # Either both methods caught it (AGREE_BAD, not in the disagreements
    # list at all) or only ours did -- either way our own detector must not
    # have missed its own, already-validated bread-and-butter case.
    window_col = 10
    ch_idx = result["channels"].index(ch_name)
    our_result = result["our_result"]
    flagged_desc = {
        (onset, ch)
        for onset, desc, chs in zip(
            our_result["annotations"].onset, our_result["annotations"].description, our_result["annotations"].ch_names
        )
        for ch in chs
        if desc == "BAD_amplitude"
    }
    assert (raw.first_time + window_col * result["window_duration"], ch_name) in flagged_desc or cell_flagged_by_ours


def test_disagreement_onset_matches_window_duration_grid():
    rng = np.random.default_rng(3)
    data_uv = rng.normal(0.0, 5.0, size=(6, int(20 * 250.0)))
    raw = _make_raw(data_uv)

    result = run_consensus(raw)

    for entry in result["disagreements"]:
        expected_onset = raw.first_time + entry["window"] * result["window_duration"]
        assert entry["onset_s"] == pytest.approx(expected_onset)


def test_missing_autoreject_raises_clear_import_error(monkeypatch):
    monkeypatch.setattr(consensus, "AutoReject", None)
    rng = np.random.default_rng(4)
    data_uv = rng.normal(0.0, 5.0, size=(6, int(20 * 250.0)))
    raw = _make_raw(data_uv)

    with pytest.raises(ImportError, match="autoreject"):
        run_consensus(raw)
