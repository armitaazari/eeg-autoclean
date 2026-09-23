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


def test_load_eeg_rejects_unsupported_extension(tmp_path):
    bad_file = tmp_path / "recording.txt"
    bad_file.write_text("not an eeg file")

    with pytest.raises(ValueError):
        load_eeg(bad_file)


def test_detect_artifacts_clean_signal_has_no_flags():
    rng = np.random.default_rng(0)
    # 30s / 30 windows: enough windows for every layer's per-channel
    # median+MAD adaptive threshold to be statistically meaningful. All
    # three layers use the same window-based adaptive-threshold philosophy,
    # which (like any median/MAD estimate) needs a reasonable sample size to
    # be stable -- a handful of windows can make the MAD itself noisy enough
    # to spuriously flag an unremarkable window (this was observed directly
    # with Layer 3's ratio metric on a 5-window recording during development).
    duration_s = 30
    sfreq = 100.0
    n_samples = int(duration_s * sfreq)

    # Mild noise, well within the default amplitude/flat thresholds.
    data_uv = rng.normal(loc=0.0, scale=5.0, size=(2, n_samples))
    raw = _make_raw(data_uv, sfreq=sfreq)

    results = detect_artifacts(raw)

    assert list(results["annotations"].description) == []
    assert results["ica_components_flagged"] is None
    assert results["ica"] is None
    assert any("skipped" in note.lower() for note in results["notes"])


def test_detect_artifacts_flags_amplitude_spike():
    rng = np.random.default_rng(1)
    sfreq = 100.0
    n_samples = int(5 * sfreq)  # 5 one-second windows

    data_uv = rng.normal(loc=0.0, scale=5.0, size=(2, n_samples))
    # Inject a large-swing artifact (> default 150 uV p2p) into channel 0,
    # window 2 (seconds 2-3). A constant offset wouldn't change peak-to-peak,
    # so use an alternating square wave to actually widen the swing.
    spike_start = int(2 * sfreq)
    spike_stop = int(3 * sfreq)
    n_spike_samples = spike_stop - spike_start
    square_wave = 100.0 * np.sign(np.sin(2 * np.pi * 5 * np.arange(n_spike_samples) / sfreq))
    data_uv[0, spike_start:spike_stop] = square_wave  # +-100 uV -> 200 uV p2p

    raw = _make_raw(data_uv, sfreq=sfreq, ch_names=["EEG000", "EEG001"])
    results = detect_artifacts(raw)

    descriptions = list(results["annotations"].description)
    ch_names = list(results["annotations"].ch_names)
    assert "BAD_amplitude" in descriptions

    flagged_channels = {
        ch for desc, chs in zip(descriptions, ch_names) if desc == "BAD_amplitude" for ch in chs
    }
    assert "EEG000" in flagged_channels
    assert "EEG001" not in flagged_channels


def test_detect_artifacts_flags_flat_channel():
    rng = np.random.default_rng(2)
    sfreq = 100.0
    n_samples = int(5 * sfreq)

    data_uv = rng.normal(loc=0.0, scale=5.0, size=(2, n_samples))
    data_uv[1, :] = 0.0  # channel 1 is completely flat (disconnected electrode)

    raw = _make_raw(data_uv, sfreq=sfreq, ch_names=["EEG000", "EEG001"])
    results = detect_artifacts(raw)

    descriptions = list(results["annotations"].description)
    ch_names = list(results["annotations"].ch_names)
    assert "BAD_flat" in descriptions

    flagged_channels = {
        ch for desc, chs in zip(descriptions, ch_names) if desc == "BAD_flat" for ch in chs
    }
    assert "EEG001" in flagged_channels
    assert "EEG000" not in flagged_channels


def test_detect_artifacts_raises_without_eeg_channels():
    rng = np.random.default_rng(3)
    sfreq = 100.0
    n_samples = int(2 * sfreq)
    data_uv = rng.normal(scale=5.0, size=(1, n_samples))

    raw = _make_raw(data_uv, sfreq=sfreq, ch_types="misc", ch_names=["MISC000"])

    with pytest.raises(ValueError):
        detect_artifacts(raw)


def test_detect_artifacts_runs_ica_when_eog_present():
    # Smoke test for the Layer 2 code path: with an EOG channel present, ICA
    # should be fit and a (possibly empty) list of flagged components
    # returned, instead of the "skipped" path used when there's no EOG.
    rng = np.random.default_rng(4)
    sfreq = 250.0
    duration_s = 20
    n_samples = int(duration_s * sfreq)
    n_eeg = 4

    times = np.arange(n_samples) / sfreq
    eeg_data = rng.normal(loc=0.0, scale=5.0, size=(n_eeg, n_samples))

    # Simulate periodic blink-like pulses on the EOG channel, weakly leaking
    # into one EEG channel, so ICA has a real ocular source to find.
    eog_data = np.zeros(n_samples)
    for blink_time in np.arange(2, duration_s, 4):
        pulse = 150.0 * np.exp(-0.5 * ((times - blink_time) / 0.1) ** 2)
        eog_data += pulse
        eeg_data[0] += 0.3 * pulse

    data_uv = np.vstack([eeg_data, eog_data])
    ch_names = [f"EEG{i:03d}" for i in range(n_eeg)] + ["EOG000"]
    ch_types = ["eeg"] * n_eeg + ["eog"]
    raw = _make_raw(data_uv, sfreq=sfreq, ch_types=ch_types, ch_names=ch_names)

    results = detect_artifacts(raw, ica_n_components=2)

    assert results["ica"] is not None
    assert isinstance(results["ica_components_flagged"], list)
    assert all(isinstance(idx, (int, np.integer)) for idx in results["ica_components_flagged"])
