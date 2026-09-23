import matplotlib

matplotlib.use("Agg")  # headless backend; we only save PNGs in tests, never show()

import mne
import numpy as np

from eeg_autoclean.detector import detect_artifacts
from eeg_autoclean.visualize import plot_artifacts


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


def test_plot_artifacts_saves_png_without_ica_panel(tmp_path):
    rng = np.random.default_rng(0)
    sfreq = 100.0
    n_samples = int(10 * sfreq)
    n_eeg = 4

    data_uv = rng.normal(loc=0.0, scale=5.0, size=(n_eeg, n_samples))
    # Inject an amplitude artifact so there's an annotation to shade.
    data_uv[0, 200:300] = 100.0 * np.sign(np.sin(2 * np.pi * 5 * np.arange(100) / sfreq))

    raw = _make_raw(data_uv, sfreq=sfreq)
    result = detect_artifacts(raw)  # no EOG channel -> Layer 2 skipped

    output_path = tmp_path / "no_ica.png"
    fig = plot_artifacts(raw, result, output_path=output_path, n_channels=4, duration=10.0)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert fig is not None


def test_plot_artifacts_saves_png_with_ica_panel(tmp_path):
    rng = np.random.default_rng(4)
    sfreq = 250.0
    duration_s = 20
    n_samples = int(duration_s * sfreq)
    n_eeg = 4

    times = np.arange(n_samples) / sfreq
    eeg_data = rng.normal(loc=0.0, scale=5.0, size=(n_eeg, n_samples))

    eog_data = np.zeros(n_samples)
    for blink_time in np.arange(2, duration_s, 4):
        pulse = 150.0 * np.exp(-0.5 * ((times - blink_time) / 0.1) ** 2)
        eog_data += pulse
        eeg_data[0] += 0.3 * pulse

    data_uv = np.vstack([eeg_data, eog_data])
    ch_names = [f"EEG{i:03d}" for i in range(n_eeg)] + ["EOG000"]
    ch_types = ["eeg"] * n_eeg + ["eog"]
    raw = _make_raw(data_uv, sfreq=sfreq, ch_types=ch_types, ch_names=ch_names)

    result = detect_artifacts(raw, ica_n_components=2)

    output_path = tmp_path / "with_ica.png"
    fig = plot_artifacts(raw, result, output_path=output_path, n_channels=4, duration=10.0)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert fig is not None
