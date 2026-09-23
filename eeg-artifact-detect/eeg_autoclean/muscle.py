"""Layer 3: muscle (EMG) artifact detection.

Muscle contraction artifacts show up as a broadband increase in
high-frequency power, unlike ocular artifacts (low-frequency, Layer 2's
territory) or the gross amplitude/flat issues Layer 1 catches. Reuses
Layer 1's window scheme (see _windowing.compute_window_bounds) so all
annotation-producing layers line up at the same (channel, window)
granularity.

Method: per channel, per window, the ratio of power in a high-frequency
"muscle band" to power in the channel's own broader working band, then an
adaptive per-channel threshold (median + k * MAD of that ratio across the
channel's own windows) -- same philosophy as Layer 1's amplitude
threshold, and for the same reason: a fixed ratio cutoff over- or
under-flags depending on how much genuine high-frequency activity a given
channel or subject normally has.

Like any median/MAD threshold, this needs a reasonable number of windows
per channel to be stable -- with only a handful of windows, the MAD itself
gets noisy enough to spuriously flag an unremarkable one. The ratio metric
here has more window-to-window variance than Layer 1's peak-to-peak
amplitude, so it shows up here first, but in practice this only matters
for recordings a few seconds long, well under any real EEG recording.

Precision (mean 0.50 across the synthetic battery) is noticeably lower
than Layer 1's (1.00) -- a structural property of the PSD-ratio metric
(more inherent estimator variance than a time-domain peak-to-peak
measurement), confirmed rather than assumed: halving the window to 0.5s
made things worse (mean F1 0.60 -> ~0.19 -- more, noisier windows outweigh
better localization); the shipped Welch nperseg = n // 4 heuristic
(~7 averaged sub-segments) already beats both coarser (n // 2: F1 0.31)
and finer (whole-window periodogram: F1 0.54) alternatives; and
muscle_mad_k plateaus at F1 0.56-0.60 across k=4-7 with a real
precision/recall tradeoff inside that range (k=4: P=0.43/R=0.97; k=5
shipped: P=0.50/R=0.84; k=7: P=0.92/R=0.46), collapsing outside it. k=5
stays the default -- it matches Layer 1's k and sits at the highest-recall
end of the plateau, and for a detector flagging candidates for review,
missing a real artifact costs more than an extra one to dismiss. The
"dense/severe" scenario's recall dip (~0.38) is insensitive to k in this
range -- several muscle/amplitude/flat events sharing few channels raises
those channels' own adaptive baseline, which isn't something a threshold
tweak fixes.

Temporal persistence (require a flag to survive `persistence_n`
consecutive windows, see _windowing.apply_persistence_filter) was also
tried, on the theory that a real muscle burst spans multiple windows.
Against ground truth built with muscle_event_windows=3, persistence_n=2
gives mean P=0.75/R=0.52/F1=0.60 vs persistence_n=1's P=0.66/R=0.62/F1=0.60
-- precision up, recall down by about as much, F1 unchanged. Not a clear
win, so the default stays at persistence_n=1 (no requirement); the
parameter is implemented (DEFAULT_MUSCLE_PERSISTENCE_N) for callers who
want it anyway. Layer 4/cardiac responded differently to the same
experiment -- a real improvement, see eeg_autoclean.cardiac -- so the two
layers were decided independently rather than as one blanket policy.
"""

import warnings

import mne
import numpy as np
from scipy.signal import welch

from ._windowing import (
    DEFAULT_WINDOW_DURATION,
    apply_persistence_filter as _apply_persistence_filter,
    compute_window_bounds as _compute_window_bounds,
)

# Muscle band: EMG contamination is broadband but strongest in the
# higher end of what's still "typical EEG bandwidth" for this project (Layer
# 1/2 both default to a 1 Hz high-pass; nothing here assumes filtering above
# 45 Hz). 20-40 Hz sits above the alpha/beta rhythms Layer 2 already has to
# reason about (occipital alpha near 10 Hz, beta up to ~30 Hz can overlap
# the low end, which is why we don't start lower than 20 Hz), and stays
# below line-noise territory (50/60 Hz) and typical hardware/anti-aliasing
# roll-off near 45 Hz, so it doesn't need to special-case a specific mains
# frequency.
MUSCLE_BAND = (20.0, 40.0)
# Reference band the muscle band is compared against: 1-20 Hz, i.e.
# everything below the muscle band down to Layer 1/2's shared 1 Hz
# high-pass floor (delta/theta/alpha/low-beta -- where genuine EEG signal
# normally dominates). This is deliberately EXCLUSIVE of the muscle band,
# not the full 1-40 Hz working range: an earlier version used an inclusive
# 1-40 Hz reference, but since that band average partly consists of the
# muscle band's own power, a strong muscle burst inflates its own
# denominator along with the numerator, capping the achievable ratio at
# (reference bandwidth / muscle bandwidth) regardless of how strong the
# burst is -- close to 2.0 here, leaving almost no headroom above the
# adaptive threshold's typical noise margin. An exclusive reference band
# has no such ceiling: a genuine high-frequency burst can push the ratio
# arbitrarily high without also lifting its own baseline.
MUSCLE_REFERENCE_BAND = (1.0, 20.0)

DEFAULT_MUSCLE_MAD_K = 5.0
DEFAULT_MUSCLE_HIGHPASS_HZ = 1.0
# Temporal persistence: a flagged window is only confirmed if the same
# channel is flagged in at least this many consecutive windows.
# persistence_n=1 (the default) is a no-op -- every flagged window is
# confirmed on its own, i.e. the original, pre-persistence behavior.
# See scripts/generate_synthetic.py / README for the measured
# precision/recall effect of raising this.
DEFAULT_MUSCLE_PERSISTENCE_N = 1


def _window_band_ratio(segment_uv, sfreq, muscle_band, reference_band):
    """Per-channel ratio of mean power in muscle_band to mean power in
    reference_band, for one window's (channels x samples) segment.

    Uses Welch's method (averaged over several overlapping sub-segments)
    rather than a single periodogram over the whole window: a single-segment
    estimate is noisy enough that, with only a handful of windows in a
    recording, the per-channel median/MAD statistics below can themselves be
    unstable and produce spurious flags on pure noise. No PSD normalization
    is applied -- normalization constants are identical for both bands and
    cancel out in the ratio, so relative power from welch()'s own scaling is
    enough.
    """
    n = segment_uv.shape[1]
    nperseg = min(n, max(8, n // 4))
    freqs, power = welch(segment_uv, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2, axis=1)

    def _band_mean(band):
        mask = (freqs >= band[0]) & (freqs < band[1])
        if not mask.any():
            return np.zeros(segment_uv.shape[0])
        return power[:, mask].mean(axis=1)

    muscle_power = _band_mean(muscle_band)
    reference_power = _band_mean(reference_band)
    # A channel with ~zero reference-band power (flat/disconnected -- Layer
    # 1's territory) gives an undefined ratio; mark it NaN rather than
    # dividing by zero, and exclude it from thresholding/flagging below.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(reference_power > 0, muscle_power / reference_power, np.nan)
    return ratio


def _scan_muscle_windows(
    raw,
    muscle_mad_k=DEFAULT_MUSCLE_MAD_K,
    muscle_band=MUSCLE_BAND,
    muscle_reference_band=MUSCLE_REFERENCE_BAND,
    window_duration=DEFAULT_WINDOW_DURATION,
    highpass_hz=DEFAULT_MUSCLE_HIGHPASS_HZ,
    persistence_n=DEFAULT_MUSCLE_PERSISTENCE_N,
):
    """Scan EEG channels in fixed-length windows for broadband high-frequency
    (muscle/EMG) power increases.

    Returns
    -------
    annotations : mne.Annotations
        Annotations with description "BAD_muscle".
    """
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(picks) == 0:
        raise ValueError("No EEG channels found in raw data; cannot run muscle detection.")
    ch_names = [raw.ch_names[p] for p in picks]

    if highpass_hz is not None:
        scan_raw = raw.copy().filter(l_freq=highpass_hz, h_freq=None, picks=picks, verbose=False)
    else:
        scan_raw = raw

    data_uv = scan_raw.get_data(picks=picks) * 1e6  # volts -> microvolts
    if not np.all(np.isfinite(data_uv)):
        raise ValueError(
            "EEG data contains NaN or Inf values; cannot run muscle detection "
            "(check for a corrupt recording or a failed/disconnected channel)."
        )

    sfreq = raw.info["sfreq"]
    window_samples = max(1, int(round(window_duration * sfreq)))
    n_samples = data_uv.shape[1]

    window_bounds = _compute_window_bounds(n_samples, window_samples)
    if not window_bounds:
        raise ValueError("Recording is too short for even one detection window; cannot run muscle detection.")

    ratio_matrix = np.full((len(ch_names), len(window_bounds)), np.nan)
    for w, (start, stop) in enumerate(window_bounds):
        segment = data_uv[:, start:stop]
        ratio_matrix[:, w] = _window_band_ratio(segment, sfreq, muscle_band, muscle_reference_band)

    # Per-channel adaptive threshold: median + k * robust_std (1.4826 * MAD)
    # of that channel's own muscle-band ratio across all its windows -- same
    # philosophy as Layer 1's amplitude threshold, and for the same reason: a
    # fixed ratio cutoff was already shown to fail across recordings with
    # different noise floors (see detector.py's amplitude threshold notes).
    # NaN entries (flat/disconnected channels, see _window_band_ratio) are
    # excluded from both the statistics and any flagging.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        median_ratio = np.nanmedian(ratio_matrix, axis=1)
        mad_ratio = np.nanmedian(np.abs(ratio_matrix - median_ratio[:, None]), axis=1)
    robust_std = 1.4826 * mad_ratio
    adaptive_threshold = median_ratio + muscle_mad_k * robust_std

    # NaN ratio or threshold (flat/disconnected channel) compares as False
    # automatically, so those cells are never flagged here.
    with np.errstate(invalid="ignore"):
        flagged = ratio_matrix > adaptive_threshold[:, None]
    flagged = _apply_persistence_filter(flagged, persistence_n)

    onsets, durations, descriptions, annot_ch_names = [], [], [], []
    for w, (start, stop) in enumerate(window_bounds):
        onset = raw.first_time + start / sfreq
        duration = (stop - start) / sfreq
        for ch_idx, ch_name in enumerate(ch_names):
            if flagged[ch_idx, w]:
                onsets.append(onset)
                durations.append(duration)
                descriptions.append("BAD_muscle")
                annot_ch_names.append([ch_name])

    return mne.Annotations(
        onset=onsets,
        duration=durations,
        description=descriptions,
        ch_names=annot_ch_names,
        orig_time=raw.info["meas_date"],
    )
