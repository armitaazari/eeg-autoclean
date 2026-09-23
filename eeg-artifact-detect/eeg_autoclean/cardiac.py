"""Layer 4: cardiac (ECG/pulse) artifact detection.

Cardiac (pulse/ECG) contamination shows up in EEG as a quasi-periodic
pulse at heart-rate frequency -- unlike ocular artifacts (low-frequency,
event-locked to blinks, Layer 2's territory) or muscle artifacts
(broadband, aperiodic high-frequency power, Layer 3's territory), the
defining signature here is PERIODICITY at a specific, narrow, physiologically
plausible rate, not a particular frequency band or amplitude level as such.

Method: per channel, per window, the strongest normalized autocorrelation
value at any lag corresponding to a plausible heart rate, then an adaptive
per-channel threshold (median + k * MAD of that value across the channel's
own windows) -- same philosophy as Layers 1 and 3. Autocorrelation (rather
than template-based QRS peak detection) is used because it doesn't need to
know the pulse's shape or amplitude in advance, only that it repeats -- a
genuinely periodic contamination source produces a strong, narrow
autocorrelation peak near its true period, while normal (aperiodic, or
periodic at a different, unrelated rate) EEG activity does not.

No ECG channel is required -- pure EEG montages essentially never have one,
so the primary method above works on EEG channels alone. If an ECG channel
IS present, its cross-correlation with each EEG channel is used as a second,
independent detection path (mirrors Layer 2's dual-pass EOG design): a
window is flagged if EITHER path catches it.

In practice, real cardiac contamination is typically strongest on channels
physically closer to the heart/neck (temporal, posterior, or channels near
a pulsating vessel) -- this isn't hardcoded via channel-name matching
(unlike eye_state's occipital-channel lookup), since that would only work
on montages using standard 10-20-style names. Instead, since detection and
thresholding are both per-channel, those channels simply end up with a
higher periodicity metric and threshold-relative flag rate on their own,
without needing to know their names or positions.

Temporal persistence (require a flag to persist across `persistence_n`
consecutive windows, see _windowing.apply_persistence_filter) is on by
default here, unlike Layer 3/muscle where the same idea didn't pay off
(see eeg_autoclean.muscle). Real cardiac contamination is sustained,
periodic activity that persists across several consecutive windows, not a
single 6s burst that vanishes, so the synthetic battery's cardiac ground
truth (eeg_autoclean.synthetic.SEVERITY_SCENARIOS) injects 2-window (12s)
pulse trains, and persistence_n=2 is validated directly against it
(scripts/generate_synthetic.py):

    persistence_n=1: mean P=0.29 (sd=0.17)  R=0.75 (sd=0.43)  F1=0.41 (sd=0.24)
    persistence_n=2: mean P=0.67 (sd=0.41)  R=0.75 (sd=0.43)  F1=0.70 (sd=0.41)

Precision more than doubles; recall is identical per scenario, not just on
average (1.00/1.00/1.00/0.00 either way) -- every real detection that
existed at persistence_n=1 survives, and false positives are suppressed
(fp per scenario: 3->0, 5->0, 9->2, 4->0).

The "noisy background" scenario's 0.0 recall is unchanged by persistence_n
(present at both 1 and 2): background noise (sigma=20uV) overwhelms the
pulse amplitude there, the same kind of SNR-limited soft spot Layer 2 has
on this exact scenario -- not something persistence filtering caused or
can fix.
"""

import warnings

import mne
import numpy as np

from ._windowing import (
    apply_persistence_filter as _apply_persistence_filter,
    compute_window_bounds as _compute_window_bounds,
)

# Plausible heart-rate range used to bound which autocorrelation lags count
# as "cardiac". Typical resting heart rate is 60-100 bpm; widened to 40-140
# bpm to also cover trained-athlete bradycardia and mild tachycardia/anxiety
# without drifting into genuine EEG rhythm territory -- even the upper bound
# (140 bpm = 2.33 Hz) stays safely below theta (4-8 Hz), so a strongly
# regular theta oscillation can't masquerade as a cardiac-range periodicity.
DEFAULT_CARDIAC_MIN_BPM = 40.0
DEFAULT_CARDIAC_MAX_BPM = 140.0

# Analysis window: needs several full pulse cycles to get a stable
# autocorrelation estimate at the lag of interest, unlike Layer 1/3's 1.0s
# window. At the slowest rate in range (40 bpm, 1.5s period), 6.0s contains
# 4 full cycles; at the fastest (140 bpm, 0.43s period) it contains 14 --
# comfortably enough at both ends for the peak to be distinguishable from
# autocorrelation noise.
DEFAULT_CARDIAC_WINDOW_DURATION = 6.0  # seconds

DEFAULT_CARDIAC_MAD_K = 5.0
DEFAULT_CARDIAC_HIGHPASS_HZ = 1.0
# Fixed correlation-coefficient threshold for the optional ECG cross-
# correlation path -- not adaptive, since a normalized correlation
# coefficient is already a bounded, directly interpretable quantity (unlike
# a raw power-ratio or amplitude value). 0.5 matches the same threshold
# eeg_autoclean.evaluation.evaluate_layer2 already uses to call a component
# "truly ocular" when correlated against a known EOG waveform -- the same
# standard for "this is clearly the same physiological source," reused here
# for consistency rather than picked arbitrarily.
DEFAULT_CARDIAC_ECG_CORRELATION_THRESHOLD = 0.5
# Small lag window (seconds) searched for the best-aligned cross-correlation
# against the ECG channel, to tolerate a modest propagation delay between
# the ECG channel and a given EEG electrode's pulse artifact.
DEFAULT_CARDIAC_ECG_MAX_LAG_S = 0.2
# Temporal persistence: a flagged window is only confirmed if the same
# channel is flagged in at least this many consecutive windows.
# persistence_n=1 is a no-op (every flagged window confirmed on its own).
# Default is 2: validated on the synthetic battery (see module docstring)
# to more than double precision (0.29 -> 0.67) with recall unchanged
# (0.75 -> 0.75, identical per scenario, not just on average) -- unlike
# Layer 3/muscle, where the matching investigation was a wash and its
# default stayed at 1 (see eeg_autoclean.muscle).
DEFAULT_CARDIAC_PERSISTENCE_N = 2


def _max_autocorrelation_in_bpm_range(segment_uv, sfreq, min_bpm, max_bpm):
    """Per-channel max normalized autocorrelation at lags within
    [60/max_bpm, 60/min_bpm] seconds, for one window's (channels x samples)
    segment. NaN for a channel with ~zero energy (flat/disconnected).
    """
    n = segment_uv.shape[1]
    lag_min = max(1, int(round(sfreq * 60.0 / max_bpm)))
    lag_max = min(n - 1, int(round(sfreq * 60.0 / min_bpm)))
    if lag_min > lag_max:
        return np.full(segment_uv.shape[0], np.nan)

    x = segment_uv - segment_uv.mean(axis=1, keepdims=True)
    energy = np.sum(x**2, axis=1)  # equals the lag-0 autocorrelation

    # Linear (not circular) autocorrelation via zero-padded FFT: circular
    # wraparound would let energy from one end of the window spuriously
    # inflate the correlation at large lags near the window length.
    n_fft = 2 * n
    spectrum = np.fft.rfft(x, n=n_fft, axis=1)
    acf = np.fft.irfft(spectrum * np.conj(spectrum), n=n_fft, axis=1)[:, :n]

    with np.errstate(divide="ignore", invalid="ignore"):
        acf_norm = np.where(energy[:, None] > 0, acf / energy[:, None], np.nan)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        return np.nanmax(acf_norm[:, lag_min : lag_max + 1], axis=1)


def _best_lag_correlation(eeg_segment_uv, ecg_segment_uv, sfreq, max_lag_s):
    """Per-channel Pearson correlation between each EEG channel and the ECG
    channel, maximized over a small range of lags (+-max_lag_s), to tolerate
    a propagation delay between the ECG channel and a given electrode.

    Each lagged slice is re-centered on its OWN mean (not the whole
    segment's), since a slice's local mean can drift from the full
    segment's mean -- using the global mean would bias the covariance
    estimate at nonzero lags.
    """
    max_lag = max(1, int(round(max_lag_s * sfreq)))
    ecg = ecg_segment_uv[0]
    n = eeg_segment_uv.shape[1]
    best = np.zeros(eeg_segment_uv.shape[0])
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            eeg_slice = eeg_segment_uv[:, : n - lag] if lag > 0 else eeg_segment_uv
            ecg_slice = ecg[lag:]
        else:
            eeg_slice = eeg_segment_uv[:, -lag:]
            ecg_slice = ecg[: n + lag]
        if eeg_slice.shape[1] < 2:
            continue
        eeg_c = eeg_slice - eeg_slice.mean(axis=1, keepdims=True)
        ecg_c = ecg_slice - ecg_slice.mean()
        eeg_std = eeg_c.std(axis=1)
        ecg_std = ecg_c.std()
        if ecg_std == 0:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(eeg_std > 0, (eeg_c * ecg_c[None, :]).mean(axis=1) / (eeg_std * ecg_std), 0.0)
        best = np.maximum(best, corr)
    return best


def _scan_cardiac_windows(
    raw,
    cardiac_mad_k=DEFAULT_CARDIAC_MAD_K,
    min_bpm=DEFAULT_CARDIAC_MIN_BPM,
    max_bpm=DEFAULT_CARDIAC_MAX_BPM,
    window_duration=DEFAULT_CARDIAC_WINDOW_DURATION,
    highpass_hz=DEFAULT_CARDIAC_HIGHPASS_HZ,
    ecg_correlation_threshold=DEFAULT_CARDIAC_ECG_CORRELATION_THRESHOLD,
    ecg_max_lag_s=DEFAULT_CARDIAC_ECG_MAX_LAG_S,
    persistence_n=DEFAULT_CARDIAC_PERSISTENCE_N,
):
    """Scan EEG channels in fixed-length windows for a quasi-periodic
    cardiac/pulse pattern.

    Returns
    -------
    annotations : mne.Annotations
        Annotations with description "BAD_cardiac". Empty if skipped.
    used_ecg_channel : bool
        Whether an ECG channel was found and used as a second detection path.
    note : str or None
        Explanation if Layer 4 was skipped (recording too short for even one
        of its longer windows), else None. Unlike a too-short recording for
        Layer 1/3 (a hard error there, since no detection at all could run),
        this is a graceful skip: window_duration is deliberately several
        times longer than Layer 1/3's, so a recording that's plenty long
        enough for the rest of the tool can still be too short for Layer 4
        alone -- that shouldn't abort the whole detect_artifacts() call.
    """
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(picks) == 0:
        raise ValueError("No EEG channels found in raw data; cannot run cardiac detection.")
    ch_names = [raw.ch_names[p] for p in picks]

    ecg_picks = mne.pick_types(raw.info, ecg=True, exclude=[])
    used_ecg_channel = len(ecg_picks) > 0

    if highpass_hz is not None:
        scan_raw = raw.copy().filter(l_freq=highpass_hz, h_freq=None, picks=list(picks) + list(ecg_picks), verbose=False)
    else:
        scan_raw = raw

    data_uv = scan_raw.get_data(picks=picks) * 1e6  # volts -> microvolts
    if not np.all(np.isfinite(data_uv)):
        raise ValueError(
            "EEG data contains NaN or Inf values; cannot run cardiac detection "
            "(check for a corrupt recording or a failed/disconnected channel)."
        )
    ecg_data_uv = scan_raw.get_data(picks=ecg_picks) * 1e6 if used_ecg_channel else None

    sfreq = raw.info["sfreq"]
    window_samples = max(1, int(round(window_duration * sfreq)))
    n_samples = data_uv.shape[1]

    window_bounds = _compute_window_bounds(n_samples, window_samples)
    if not window_bounds:
        empty = mne.Annotations(onset=[], duration=[], description=[], orig_time=raw.info["meas_date"])
        note = (
            f"Cardiac detection skipped: recording ({n_samples / sfreq:.1f}s) is shorter than "
            f"one Layer 4 window ({window_duration:.1f}s)."
        )
        return empty, used_ecg_channel, note

    periodicity_matrix = np.full((len(ch_names), len(window_bounds)), np.nan)
    ecg_corr_matrix = np.full((len(ch_names), len(window_bounds)), np.nan) if used_ecg_channel else None
    for w, (start, stop) in enumerate(window_bounds):
        segment = data_uv[:, start:stop]
        periodicity_matrix[:, w] = _max_autocorrelation_in_bpm_range(segment, sfreq, min_bpm, max_bpm)
        if used_ecg_channel:
            ecg_segment = ecg_data_uv[:, start:stop]
            ecg_corr_matrix[:, w] = _best_lag_correlation(segment, ecg_segment, sfreq, ecg_max_lag_s)

    # Per-channel adaptive threshold on the autocorrelation-based metric --
    # same median + k * MAD philosophy as Layers 1 and 3.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        median_periodicity = np.nanmedian(periodicity_matrix, axis=1)
        mad_periodicity = np.nanmedian(np.abs(periodicity_matrix - median_periodicity[:, None]), axis=1)
    robust_std = 1.4826 * mad_periodicity
    adaptive_threshold = median_periodicity + cardiac_mad_k * robust_std

    with np.errstate(invalid="ignore"):
        flagged_by_autocorr = periodicity_matrix > adaptive_threshold[:, None]
    if used_ecg_channel:
        flagged_by_ecg = ecg_corr_matrix > ecg_correlation_threshold
        flagged = flagged_by_autocorr | flagged_by_ecg
    else:
        flagged = flagged_by_autocorr
    flagged = _apply_persistence_filter(flagged, persistence_n)

    onsets, durations, descriptions, annot_ch_names = [], [], [], []
    for w, (start, stop) in enumerate(window_bounds):
        onset = raw.first_time + start / sfreq
        duration = (stop - start) / sfreq
        for ch_idx, ch_name in enumerate(ch_names):
            if flagged[ch_idx, w]:
                onsets.append(onset)
                durations.append(duration)
                descriptions.append("BAD_cardiac")
                annot_ch_names.append([ch_name])

    annotations = mne.Annotations(
        onset=onsets,
        duration=durations,
        description=descriptions,
        ch_names=annot_ch_names,
        orig_time=raw.info["meas_date"],
    )
    return annotations, used_ecg_channel, None
