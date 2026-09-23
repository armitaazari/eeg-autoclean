"""Synthetic EEG generation with known ground-truth artifacts.

Used to compute quantitative precision/recall/F1 metrics for
eeg_autoclean.detector against artifacts we know we injected, rather than
relying only on qualitative agreement with MNE's own detectors.
"""

import mne
import numpy as np

from .cardiac import DEFAULT_CARDIAC_WINDOW_DURATION
from .detector import DEFAULT_WINDOW_DURATION
from .muscle import MUSCLE_BAND

# Scenarios spanning a range of artifact rate/intensity and background
# noise, used to check that detect_artifacts()'s default parameters hold up
# across more than one lucky case. Shared between scripts/generate_synthetic.py
# (full report) and tests/test_synthetic_evaluation.py (assertions).
#
# muscle_amplitude_uv and cardiac_amplitude_uv are each calibrated per
# scenario to sit clear of Layer 1's own amplitude threshold at that
# scenario's background_noise_uv, so Layer 1, 3, and 4 ground truth stay
# cleanly separated. n_cardiac_events is kept modest relative to
# n_amplitude_events/n_muscle_events since each cardiac event reserves
# cardiac_event_windows whole 6s cardiac windows (not just a 1s window), a
# much bigger footprint on the shared reservation grid. cardiac_event_windows
# defaults to 2 here (12s pulse trains): real cardiac/pulse contamination is
# sustained, periodic activity that persists across several windows rather
# than appearing for one and vanishing, and it's what makes
# eeg_autoclean.cardiac's persistence_n=2 default possible to validate at
# all (a single-window event can't survive a persistence requirement).
SEVERITY_SCENARIOS = [
    {
        "name": "sparse/mild",
        "seed": 0,
        "n_amplitude_events": 2,
        "n_flat_events": 1,
        "n_blink_events": 4,
        "n_muscle_events": 2,
        "n_cardiac_events": 1,
        "cardiac_event_windows": 2,
        "amplitude_uv": 250.0,
        "blink_amplitude_uv": 130.0,
        "muscle_amplitude_uv": 10.0,
        "cardiac_amplitude_uv": 15.0,
        "background_noise_uv": 5.0,
    },
    {
        "name": "moderate",
        "seed": 1,
        "n_amplitude_events": 4,
        "n_flat_events": 2,
        "n_blink_events": 6,
        "n_muscle_events": 4,
        "n_cardiac_events": 2,
        "cardiac_event_windows": 2,
        "amplitude_uv": 300.0,
        "blink_amplitude_uv": 150.0,
        "muscle_amplitude_uv": 15.0,
        "cardiac_amplitude_uv": 20.0,
        "background_noise_uv": 8.0,
    },
    {
        "name": "dense/severe",
        "seed": 2,
        "n_amplitude_events": 8,
        "n_flat_events": 4,
        "n_blink_events": 10,
        "n_muscle_events": 8,
        "n_cardiac_events": 2,
        "cardiac_event_windows": 2,
        "amplitude_uv": 400.0,
        "blink_amplitude_uv": 200.0,
        "muscle_amplitude_uv": 15.0,
        "cardiac_amplitude_uv": 20.0,
        "background_noise_uv": 12.0,
    },
    {
        "name": "noisy background",
        "seed": 3,
        "n_amplitude_events": 3,
        "n_flat_events": 2,
        "n_blink_events": 5,
        "n_muscle_events": 3,
        "n_cardiac_events": 1,
        "cardiac_event_windows": 2,
        "amplitude_uv": 200.0,
        "blink_amplitude_uv": 100.0,
        "muscle_amplitude_uv": 30.0,
        "cardiac_amplitude_uv": 35.0,
        "background_noise_uv": 20.0,
    },
]


def generate_synthetic_raw(
    seed=0,
    duration_s=60.0,
    sfreq=250.0,
    n_eeg=16,
    n_frontal=3,
    n_amplitude_events=3,
    n_flat_events=2,
    flat_event_windows=2,
    n_blink_events=6,
    n_muscle_events=0,
    muscle_event_windows=1,
    n_cardiac_events=0,
    cardiac_event_windows=1,
    background_noise_uv=5.0,
    amplitude_uv=300.0,
    blink_amplitude_uv=150.0,
    muscle_amplitude_uv=20.0,
    muscle_band=MUSCLE_BAND,
    muscle_n_tones=8,
    cardiac_amplitude_uv=20.0,
    cardiac_bpm=75.0,
    cardiac_pulse_width_s=0.025,
    cardiac_window_duration=DEFAULT_CARDIAC_WINDOW_DURATION,
    blink_width_s=0.1,
    blink_leakage=0.3,
):
    """Generate a synthetic multi-channel EEG Raw with known injected artifacts.

    Amplitude, flat-channel, muscle, and cardiac events are injected only
    into non-frontal channels (at non-overlapping windows from each other),
    and blink leakage only into frontal channels, so none of the
    ground-truth categories overlap in the same (channel, window) cell and
    confound each other's scoring. Amplitude/flat/muscle events are aligned
    to whole 1.0s detection windows (using the same window length
    detect_artifacts() uses by default); cardiac events are aligned to
    whole `cardiac_window_duration` windows (several times longer -- see
    eeg_autoclean.cardiac for why), each of which spans several of the 1.0s
    windows and reserves all of them.

    Parameters
    ----------
    seed : int
        Random seed, for reproducible generation.
    duration_s : float
        Recording length in seconds (truncated to a whole number of
        detection windows).
    sfreq : float
        Sampling rate in Hz.
    n_eeg : int
        Number of synthetic EEG channels.
    n_frontal : int
        Number of "frontal" channels (the first `n_frontal`) that receive
        blink leakage instead of amplitude/flat/muscle events.
    n_amplitude_events, n_flat_events, n_blink_events, n_muscle_events,
    n_cardiac_events : int
        Number of each artifact type to inject. muscle and cardiac default
        to 0 for backward compatibility with callers that don't want that
        layer's ground truth.
    flat_event_windows, muscle_event_windows, cardiac_event_windows : int
        Number of consecutive windows each flat/muscle/cardiac event spans
        (one continuous waveform across the span, not independent
        per-window events). flat and muscle windows are 1.0s;
        cardiac windows are cardiac_window_duration. muscle/cardiac default
        to 1 (single window). Raise either to build ground truth for
        testing a temporal-persistence requirement (see
        eeg_autoclean.muscle / eeg_autoclean.cardiac's persistence_n) --
        persistence_n can never exceed the span length and still confirm a
        real event.
    background_noise_uv : float
        Standard deviation of the background Gaussian noise, in microvolts.
    amplitude_uv, blink_amplitude_uv, muscle_amplitude_uv, cardiac_amplitude_uv : float
        Peak (or peak-to-peak, for amplitude_uv) amplitude of each
        injected artifact type, in microvolts. Keep muscle/cardiac well
        clear of that scenario's amplitude-threshold trigger point (see
        SEVERITY_SCENARIOS's comment) so their ground truth stays separate
        from Layer 1's.
    muscle_band : tuple of (float, float)
        Frequency range (Hz) the injected muscle burst's energy is confined
        to. Defaults to detect_artifacts()'s own muscle band.
    muscle_n_tones : int
        Number of equally-spaced sinusoids (random phase each) summed to
        build a burst confined to muscle_band -- a simple stand-in for
        genuine broadband EMG activity in that range.
    cardiac_bpm : float
        Beats per minute of the injected periodic pulse train.
    cardiac_pulse_width_s : float
        Gaussian width (sigma) of each individual pulse, in seconds -- much
        narrower than a blink (a QRS-like pulse is far briefer).
    cardiac_window_duration : float
        Length in seconds of one cardiac_event_windows unit. Defaults to
        detect_artifacts()'s own Layer 4 window.
    blink_width_s : float
        Gaussian width (sigma) of each blink deflection, in seconds.
    blink_leakage : float
        Fraction of the EOG blink amplitude that leaks into frontal EEG
        channels (simulates genuine ocular contamination).

    Returns
    -------
    raw : mne.io.Raw
    ground_truth : dict
        {
            "window_duration": float,
            "amplitude_events": list of (channel_name, window_index),
            "flat_events": list of (channel_name, window_index),
            "muscle_events": list of (channel_name, window_index),
            "cardiac_window_duration": float,
            "cardiac_events": list of (channel_name, cardiac_window_index),
            "blink_times": list of float (seconds),
            "blink_width_s": float,
            "blink_amplitude_uv": float,
            "blink_leakage": float,
            "frontal_channels": list of str,
            "eog_channel": str,
        }
    """
    if n_eeg <= n_frontal:
        raise ValueError("n_eeg must be greater than n_frontal.")

    rng = np.random.default_rng(seed)
    window_duration = DEFAULT_WINDOW_DURATION
    window_samples = int(round(window_duration * sfreq))
    n_windows = int(duration_s * sfreq) // window_samples
    if flat_event_windows > n_windows:
        raise ValueError("flat_event_windows cannot exceed the number of available windows.")
    if muscle_event_windows > n_windows:
        raise ValueError("muscle_event_windows cannot exceed the number of available windows.")
    n_samples = n_windows * window_samples  # whole windows only

    cardiac_sub_windows = max(1, int(round(cardiac_window_duration / window_duration)))
    n_cardiac_windows = n_windows // cardiac_sub_windows
    if n_cardiac_events > 0 and n_cardiac_windows < cardiac_event_windows:
        raise ValueError(
            f"Recording has only {n_cardiac_windows} cardiac_window_duration window(s), "
            f"fewer than cardiac_event_windows={cardiac_event_windows}."
        )

    ch_names = [f"EEG{i:03d}" for i in range(n_eeg)]
    frontal_channels = ch_names[:n_frontal]
    non_frontal_channels = list(range(n_frontal, n_eeg))
    eog_channel = "EOG000"

    eeg_data = rng.normal(0.0, background_noise_uv, size=(n_eeg, n_samples))

    # --- Amplitude spikes: whole-window square waves on non-frontal channels ---
    reserved = set()
    max_attempts = 10000

    def _random_free_window(channel):
        for _ in range(max_attempts):
            w = int(rng.integers(0, n_windows))
            if (channel, w) not in reserved:
                return w
        raise RuntimeError("Could not find a free window; reduce event counts or increase duration.")

    def _random_free_window_span(channel, span_length):
        """Like _random_free_window, generalized to a span of consecutive
        windows all free for `channel`. At span_length=1 this consumes
        random draws identically to _random_free_window (same fixed-channel,
        retry-only-the-window structure), so muscle event placement with
        the default muscle_event_windows=1 is unchanged from before this
        span support was added.
        """
        for _ in range(max_attempts):
            w0 = int(rng.integers(0, n_windows - span_length + 1))
            span = range(w0, w0 + span_length)
            if not any((channel, w) in reserved for w in span):
                return span
        raise RuntimeError("Could not find a free window span; reduce event counts or increase duration.")

    amplitude_events = []
    for _ in range(n_amplitude_events):
        ch = int(rng.choice(non_frontal_channels))
        w = _random_free_window(ch)
        reserved.add((ch, w))
        start, stop = w * window_samples, (w + 1) * window_samples
        t = np.arange(stop - start) / sfreq
        # Alternating square wave (not a constant offset): a constant shift
        # doesn't change peak-to-peak amplitude, so it wouldn't trip the
        # amplitude threshold.
        eeg_data[ch, start:stop] = (amplitude_uv / 2) * np.sign(np.sin(2 * np.pi * 5 * t))
        amplitude_events.append((ch_names[ch], w))

    # --- Flat/disconnected channel segments on non-frontal channels ---
    flat_events = []
    for _ in range(n_flat_events):
        for _ in range(max_attempts):
            ch = int(rng.choice(non_frontal_channels))
            w0 = int(rng.integers(0, n_windows - flat_event_windows + 1))
            span = range(w0, w0 + flat_event_windows)
            if not any((ch, w) in reserved for w in span):
                break
        else:
            raise RuntimeError("Could not find a free window span; reduce event counts or increase duration.")
        for w in span:
            reserved.add((ch, w))
            start, stop = w * window_samples, (w + 1) * window_samples
            eeg_data[ch, start:stop] = 0.0
            flat_events.append((ch_names[ch], w))

    # --- Muscle bursts: multi-tone signal confined to muscle_band, on
    # non-frontal channels, spanning muscle_event_windows consecutive
    # windows not already used by an amplitude/flat event ---
    muscle_events = []
    tone_freqs = np.linspace(
        muscle_band[0], muscle_band[1], muscle_n_tones, endpoint=False
    ) + (muscle_band[1] - muscle_band[0]) / (2 * muscle_n_tones)
    for _ in range(n_muscle_events):
        ch = int(rng.choice(non_frontal_channels))
        span = _random_free_window_span(ch, muscle_event_windows)
        w0 = span.start
        for w in span:
            reserved.add((ch, w))
        start, stop = w0 * window_samples, (w0 + muscle_event_windows) * window_samples
        t = np.arange(stop - start) / sfreq
        burst = np.zeros(stop - start)
        for f in tone_freqs:
            burst += np.sin(2 * np.pi * f * t + rng.uniform(0.0, 2 * np.pi))
        burst *= muscle_amplitude_uv / np.max(np.abs(burst))
        eeg_data[ch, start:stop] += burst
        for w in span:
            muscle_events.append((ch_names[ch], w))

    # --- Cardiac pulse trains: a periodic train of narrow Gaussian pulses
    # (a simple stand-in for QRS-like spikes) at cardiac_bpm, on non-frontal
    # channels, spanning cardiac_event_windows consecutive
    # cardiac_window_duration windows at a time. Each event reserves all of
    # its span's underlying 1.0s sub-windows, so it can't overlap an
    # amplitude/flat/muscle event either.
    cardiac_events = []
    cardiac_window_samples = cardiac_sub_windows * window_samples
    cardiac_period_s = 60.0 / cardiac_bpm
    for _ in range(n_cardiac_events):
        for _ in range(max_attempts):
            ch = int(rng.choice(non_frontal_channels))
            w0_c = int(rng.integers(0, n_cardiac_windows - cardiac_event_windows + 1))
            cardiac_span = range(w0_c, w0_c + cardiac_event_windows)
            sub_span = range(w0_c * cardiac_sub_windows, (w0_c + cardiac_event_windows) * cardiac_sub_windows)
            if not any((ch, w) in reserved for w in sub_span):
                break
        else:
            raise RuntimeError("Could not find a free cardiac window span; reduce event counts or increase duration.")
        for w in sub_span:
            reserved.add((ch, w))
        start = w0_c * cardiac_window_samples
        stop = (w0_c + cardiac_event_windows) * cardiac_window_samples
        t = np.arange(stop - start) / sfreq
        pulse_times = np.arange(0.0, (stop - start) / sfreq, cardiac_period_s)
        pulse_train = np.zeros(stop - start)
        for pt in pulse_times:
            pulse_train += cardiac_amplitude_uv * np.exp(-0.5 * ((t - pt) / cardiac_pulse_width_s) ** 2)
        eeg_data[ch, start:stop] += pulse_train
        for w_c in cardiac_span:
            cardiac_events.append((ch_names[ch], w_c))

    # --- Blink-like deflections on a synthetic EOG channel, leaking only
    # into frontal EEG channels ---
    times = np.arange(n_samples) / sfreq
    eog_data = np.zeros(n_samples)
    margin = 2.0  # keep blinks away from the very start/end of the recording
    blink_times = sorted(rng.uniform(margin, duration_s - margin, size=n_blink_events).tolist())
    for center in blink_times:
        pulse = blink_amplitude_uv * np.exp(-0.5 * ((times - center) / blink_width_s) ** 2)
        eog_data += pulse
        for ch in range(n_frontal):
            eeg_data[ch] += blink_leakage * pulse

    data_uv = np.vstack([eeg_data, eog_data])
    info = mne.create_info(
        ch_names=ch_names + [eog_channel],
        sfreq=sfreq,
        ch_types=["eeg"] * n_eeg + ["eog"],
    )
    raw = mne.io.RawArray(data_uv * 1e-6, info, verbose=False)  # microvolts -> volts

    ground_truth = {
        "window_duration": window_duration,
        "amplitude_events": amplitude_events,
        "flat_events": flat_events,
        "muscle_events": muscle_events,
        "cardiac_window_duration": cardiac_window_duration,
        "cardiac_events": cardiac_events,
        "blink_times": blink_times,
        "blink_width_s": blink_width_s,
        "blink_amplitude_uv": blink_amplitude_uv,
        "blink_leakage": blink_leakage,
        "frontal_channels": frontal_channels,
        "eog_channel": eog_channel,
    }
    return raw, ground_truth
