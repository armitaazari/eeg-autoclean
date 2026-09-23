"""Core detection routines for eeg_autoclean."""

from pathlib import Path

import mne
import numpy as np

from ._windowing import DEFAULT_WINDOW_DURATION, compute_window_bounds as _compute_window_bounds
from .cardiac import (
    DEFAULT_CARDIAC_ECG_CORRELATION_THRESHOLD,
    DEFAULT_CARDIAC_ECG_MAX_LAG_S,
    DEFAULT_CARDIAC_HIGHPASS_HZ,
    DEFAULT_CARDIAC_MAD_K,
    DEFAULT_CARDIAC_MAX_BPM,
    DEFAULT_CARDIAC_MIN_BPM,
    DEFAULT_CARDIAC_PERSISTENCE_N,
    DEFAULT_CARDIAC_WINDOW_DURATION,
    _scan_cardiac_windows,
)
from .muscle import (
    DEFAULT_MUSCLE_HIGHPASS_HZ,
    DEFAULT_MUSCLE_MAD_K,
    DEFAULT_MUSCLE_PERSISTENCE_N,
    MUSCLE_BAND,
    MUSCLE_REFERENCE_BAND,
    _scan_muscle_windows,
)

# Layer 1 defaults
# Amplitude flagging is adaptive by default (see _scan_amplitude_and_flat_windows):
# a per-channel threshold of median + amplitude_mad_k * robust_std(MAD) of that
# channel's own window-to-window peak-to-peak amplitude. amplitude_threshold_uv
# is None by default (no fixed ceiling); set it to a number to cap the adaptive
# threshold from climbing arbitrarily high on very noisy channels.
DEFAULT_AMPLITUDE_THRESHOLD_UV = None
DEFAULT_AMPLITUDE_MAD_K = 5.0
DEFAULT_FLAT_THRESHOLD_UV = 1.0
DEFAULT_AMPLITUDE_HIGHPASS_HZ = 1.0  # set to None to disable pre-filtering

# Layer 2 defaults
DEFAULT_ICA_N_COMPONENTS = 15
DEFAULT_ICA_RANDOM_STATE = 97
DEFAULT_ICA_HIGHPASS_HZ = 1.0  # set to None to fit ICA on unfiltered data
# z-score threshold passed to ICA.find_bads_eog() for each correlation pass
# (see _detect_ocular_ica_components); 3.0 matches MNE's own default. Lower
# = more sensitive. This is the knob to reach for if a genuinely-ocular
# component scores just under threshold and goes unflagged -- e.g. we saw
# this on an 8-channel synthetic recording, where too few components diluted
# the z-score statistics enough that a component with 0.86 correlation to
# the true ocular source still scored under 3.0.
DEFAULT_ICA_BLINK_EPOCH_THRESHOLD = 3.0
DEFAULT_ICA_WHOLE_RECORDING_THRESHOLD = 3.0


def load_eeg(file_path):
    """Load an EDF or BDF EEG recording into an MNE Raw object.

    Parameters
    ----------
    file_path : str or Path
        Path to a .edf or .bdf file.

    Returns
    -------
    raw : mne.io.Raw
        The loaded raw EEG data.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".edf":
        raw = mne.io.read_raw_edf(file_path, preload=True)
    elif suffix == ".bdf":
        raw = mne.io.read_raw_bdf(file_path, preload=True)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}. Expected .edf or .bdf.")

    return raw


def _scan_amplitude_and_flat_windows(
    raw,
    amplitude_threshold_uv=DEFAULT_AMPLITUDE_THRESHOLD_UV,
    amplitude_mad_k=DEFAULT_AMPLITUDE_MAD_K,
    flat_threshold_uv=DEFAULT_FLAT_THRESHOLD_UV,
    window_duration=DEFAULT_WINDOW_DURATION,
    highpass_hz=DEFAULT_AMPLITUDE_HIGHPASS_HZ,
):
    """Scan EEG channels in fixed-length windows for gross/flat amplitude issues.

    Data is never modified or dropped; issues are reported as annotations
    (onset, duration, description, and the specific channel involved), matching
    standard EEG preprocessing practice of preserving the raw trace.

    Amplitude flagging uses a per-channel adaptive threshold: median + k * robust_std
    of that channel's own peak-to-peak amplitude across all windows, where robust_std
    is 1.4826 * median absolute deviation (MAD) -- a standard consistent estimator of
    standard deviation that, unlike mean/std, stays reliable even when a handful of
    the channel's own windows are themselves artifacts (median/MAD only break down
    once more than half the windows are outliers). This lets the threshold scale with
    each channel's actual noise floor instead of a single fixed microvolt value, which
    otherwise causes false positives on noisier channels/recordings and false
    negatives on very quiet ones. If `amplitude_threshold_uv` is given, it acts as a
    hard ceiling on the adaptive threshold (never let it climb above that value),
    rather than replacing the adaptive computation.

    Returns
    -------
    annotations : mne.Annotations
        Annotations with description "BAD_amplitude" or "BAD_flat".
    """
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(picks) == 0:
        raise ValueError("No EEG channels found in raw data; cannot run amplitude/flat detection.")

    ch_names = [raw.ch_names[p] for p in picks]

    # Scan a high-pass-filtered copy: slow drift/DC offset inflates
    # peak-to-peak amplitude without being a gross/blink/muscle artifact
    # itself. The caller's raw is left untouched.
    if highpass_hz is not None:
        scan_raw = raw.copy().filter(l_freq=highpass_hz, h_freq=None, picks=picks, verbose=False)
    else:
        scan_raw = raw

    data_uv = scan_raw.get_data(picks=picks) * 1e6  # volts -> microvolts
    if not np.all(np.isfinite(data_uv)):
        # NaN/Inf silently survive both the peak-to-peak computation and every
        # threshold comparison below (comparisons against NaN are always
        # False), which would otherwise report a corrupt recording as
        # perfectly clean instead of failing loudly.
        raise ValueError(
            "EEG data contains NaN or Inf values; cannot run amplitude/flat detection "
            "(check for a corrupt recording or a failed/disconnected channel)."
        )
    sfreq = raw.info["sfreq"]
    window_samples = max(1, int(round(window_duration * sfreq)))
    n_samples = data_uv.shape[1]

    # A trailing partial window under half the intended length is dropped:
    # peak-to-peak of just a handful of samples (in the extreme, a single
    # leftover sample) is trivially ~0 regardless of the true signal, which
    # would otherwise falsely look "flat" no matter what the data actually
    # looks like.
    window_bounds = _compute_window_bounds(n_samples, window_samples)

    if not window_bounds:
        raise ValueError(
            "Recording is too short for even one detection window; cannot run amplitude/flat detection."
        )

    # Peak-to-peak amplitude per (channel, window), computed up front so the
    # per-channel adaptive threshold can be derived before any flagging
    # decision is made.
    ptp_matrix = np.empty((len(ch_names), len(window_bounds)))
    for w, (start, stop) in enumerate(window_bounds):
        window = data_uv[:, start:stop]
        ptp_matrix[:, w] = window.max(axis=1) - window.min(axis=1)

    median_ptp = np.median(ptp_matrix, axis=1)
    mad_ptp = np.median(np.abs(ptp_matrix - median_ptp[:, None]), axis=1)
    robust_std = 1.4826 * mad_ptp
    adaptive_threshold = median_ptp + amplitude_mad_k * robust_std
    # Never let the adaptive threshold drop below the flat-detection cutoff:
    # on a near-silent channel a tiny MAD would otherwise make ordinary,
    # non-artifact windows look like amplitude outliers relative to that
    # channel's own (near-zero) baseline.
    adaptive_threshold = np.maximum(adaptive_threshold, flat_threshold_uv)

    if amplitude_threshold_uv is not None:
        effective_threshold = np.minimum(adaptive_threshold, amplitude_threshold_uv)
    else:
        effective_threshold = adaptive_threshold

    onsets, durations, descriptions, annot_ch_names = [], [], [], []

    for w, (start, stop) in enumerate(window_bounds):
        onset = raw.first_time + start / sfreq
        duration = (stop - start) / sfreq

        for ch_idx, ch_name in enumerate(ch_names):
            ptp = ptp_matrix[ch_idx, w]
            if ptp > effective_threshold[ch_idx]:
                onsets.append(onset)
                durations.append(duration)
                descriptions.append("BAD_amplitude")
                annot_ch_names.append([ch_name])
            elif ptp < flat_threshold_uv:
                onsets.append(onset)
                durations.append(duration)
                descriptions.append("BAD_flat")
                annot_ch_names.append([ch_name])

    return mne.Annotations(
        onset=onsets,
        duration=durations,
        description=descriptions,
        ch_names=annot_ch_names,
        orig_time=raw.info["meas_date"],
    )


def _detect_ocular_ica_components(
    raw,
    ica_n_components=DEFAULT_ICA_N_COMPONENTS,
    ica_random_state=DEFAULT_ICA_RANDOM_STATE,
    ica_highpass_hz=DEFAULT_ICA_HIGHPASS_HZ,
    ica_blink_epoch_threshold=DEFAULT_ICA_BLINK_EPOCH_THRESHOLD,
    ica_whole_recording_threshold=DEFAULT_ICA_WHOLE_RECORDING_THRESHOLD,
):
    """Fit ICA and, if an EOG channel is present, flag EOG-correlated components.

    If no EOG channel exists we don't guess which components are ocular;
    Layer 2 is skipped entirely and the caller is told why.

    Two independent correlation passes are run, and a component is flagged if
    EITHER catches it:
    - "blink_epochs": ICA sources correlated against EOG-locked blink epochs
      (mne.preprocessing.create_eog_epochs + ICA.find_bads_eog), tuned for
      blink-shaped deflections. Misses continuous ocular activity that isn't
      blink-shaped, e.g. saccades.
    - "whole_recording": ICA sources correlated against the continuous EOG
      channel(s) over the entire recording (ICA.find_bads_eog(raw) directly),
      which also catches that non-blink ocular activity but is a coarser,
      less targeted correlation.
    The two signals are kept separate (which path caught which component)
    rather than silently merged, since they can disagree and that's useful
    diagnostic information (e.g. observed on task recordings with prominent
    saccades: the blink-epoch pass alone missed the ocular component that the
    whole-recording pass found).

    Returns
    -------
    ica : mne.preprocessing.ICA or None
        The fitted ICA object, or None if Layer 2 was skipped.
    flagged : list of int or None
        Union of components flagged by either detection path, or None if
        Layer 2 was skipped.
    detection_paths : dict[int, list[str]] or None
        For each flagged component index, which path(s) ("blink_epochs",
        "whole_recording") flagged it. None if Layer 2 was skipped.
    note : str or None
        Explanation when Layer 2 was skipped, or diagnostic notes from either
        pass (e.g. no blink events found).
    """
    eog_picks = mne.pick_types(raw.info, eog=True, exclude=[])
    if len(eog_picks) == 0:
        return None, None, None, "ICA-based detection skipped: no EOG channel found in raw data."

    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if len(eeg_picks) == 0:
        return None, None, None, "ICA-based detection skipped: no EEG channels available to fit ICA."

    n_components = min(ica_n_components, len(eeg_picks))

    # Fit on a high-pass-filtered copy: fastica converges poorly in the
    # presence of slow drifts. We keep the caller's raw untouched since we
    # still want to inspect/annotate the original, unfiltered trace.
    if ica_highpass_hz is not None:
        raw_for_ica = raw.copy().filter(l_freq=ica_highpass_hz, h_freq=None, picks=eeg_picks, verbose=False)
    else:
        raw_for_ica = raw

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method="fastica",
        random_state=ica_random_state,
        max_iter="auto",
    )
    ica.fit(raw_for_ica, picks=eeg_picks)

    detection_paths = {}  # component index -> list of path names that flagged it
    notes = []

    # Pass 1: blink-locked epochs.
    try:
        # No `picks` restriction here: find_bads_eog() needs the EOG channel
        # itself to still be present in eog_epochs to correlate components
        # against. (ICA itself was already fit on eeg_picks only, above.)
        eog_epochs = mne.preprocessing.create_eog_epochs(raw, verbose=False)
        if len(eog_epochs) == 0:
            notes.append("No EOG/blink events detected for the blink-epoch pass.")
        else:
            blink_indices, _scores = ica.find_bads_eog(
                eog_epochs, threshold=ica_blink_epoch_threshold, verbose=False
            )
            for idx in blink_indices:
                detection_paths.setdefault(int(idx), []).append("blink_epochs")
    except (RuntimeError, ValueError) as exc:
        notes.append(f"Blink-epoch EOG detection failed: {exc}")

    # Pass 2: whole continuous recording -- catches non-blink ocular activity
    # (e.g. saccades) that never produces a detectable "event" for the
    # blink-epoch pass above.
    try:
        whole_indices, _scores = ica.find_bads_eog(raw, threshold=ica_whole_recording_threshold, verbose=False)
        for idx in whole_indices:
            detection_paths.setdefault(int(idx), []).append("whole_recording")
    except (RuntimeError, ValueError) as exc:
        notes.append(f"Whole-recording EOG detection failed: {exc}")

    flagged = sorted(detection_paths.keys())
    note = "; ".join(notes) if notes else None
    return ica, flagged, detection_paths, note


def detect_artifacts(
    raw,
    amplitude_threshold_uv=DEFAULT_AMPLITUDE_THRESHOLD_UV,
    amplitude_mad_k=DEFAULT_AMPLITUDE_MAD_K,
    flat_threshold_uv=DEFAULT_FLAT_THRESHOLD_UV,
    window_duration=DEFAULT_WINDOW_DURATION,
    amplitude_highpass_hz=DEFAULT_AMPLITUDE_HIGHPASS_HZ,
    ica_n_components=DEFAULT_ICA_N_COMPONENTS,
    ica_random_state=DEFAULT_ICA_RANDOM_STATE,
    ica_highpass_hz=DEFAULT_ICA_HIGHPASS_HZ,
    ica_blink_epoch_threshold=DEFAULT_ICA_BLINK_EPOCH_THRESHOLD,
    ica_whole_recording_threshold=DEFAULT_ICA_WHOLE_RECORDING_THRESHOLD,
    muscle_mad_k=DEFAULT_MUSCLE_MAD_K,
    muscle_band=MUSCLE_BAND,
    muscle_reference_band=MUSCLE_REFERENCE_BAND,
    muscle_highpass_hz=DEFAULT_MUSCLE_HIGHPASS_HZ,
    muscle_persistence_n=DEFAULT_MUSCLE_PERSISTENCE_N,
    cardiac_mad_k=DEFAULT_CARDIAC_MAD_K,
    cardiac_min_bpm=DEFAULT_CARDIAC_MIN_BPM,
    cardiac_max_bpm=DEFAULT_CARDIAC_MAX_BPM,
    cardiac_window_duration=DEFAULT_CARDIAC_WINDOW_DURATION,
    cardiac_highpass_hz=DEFAULT_CARDIAC_HIGHPASS_HZ,
    cardiac_ecg_correlation_threshold=DEFAULT_CARDIAC_ECG_CORRELATION_THRESHOLD,
    cardiac_ecg_max_lag_s=DEFAULT_CARDIAC_ECG_MAX_LAG_S,
    cardiac_persistence_n=DEFAULT_CARDIAC_PERSISTENCE_N,
):
    """Detect eye-blink, muscle, cardiac, and gross amplitude/flat artifacts in an EEG recording.

    Four layers, each producing its own kind of flag:
    - Layer 1: adaptive per-channel amplitude threshold (BAD_amplitude) and
      near-zero-variance detection (BAD_flat).
    - Layer 2: ICA fit on the EEG channels; if an EOG channel is present,
      components correlated with it are flagged by two passes -- against
      blink-locked epochs, and against the whole continuous recording
      (catches non-blink activity like saccades) -- unioned together.
      Skipped gracefully if no EOG channel exists.
    - Layer 3: per-channel muscle-band/reference-band power ratio
      (BAD_muscle). See eeg_autoclean.muscle for the method.
    - Layer 4: per-channel autocorrelation in a plausible heart-rate range,
      plus ECG cross-correlation as a second detection path if an ECG
      channel is present (BAD_cardiac). See eeg_autoclean.cardiac.

    Most parameters below are per-layer variations on the same two knobs: a
    `*_mad_k` adaptive-threshold sensitivity (median + k * MAD of that
    layer's metric across its own windows; higher = less sensitive) and a
    `*_highpass_hz` pre-filter cutoff. The reasoning behind each layer's
    defaults lives in that layer's own module rather than repeated here.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data to analyze.
    amplitude_threshold_uv : float or None
        Hard ceiling (uV) on Layer 1's adaptive amplitude threshold. None
        (default) leaves it purely adaptive, with no ceiling.
    amplitude_mad_k, muscle_mad_k, cardiac_mad_k : float
        Adaptive-threshold sensitivity for Layer 1, 3, and 4 respectively.
    flat_threshold_uv : float
        Peak-to-peak amplitude (uV) below which a window is flagged
        BAD_flat; also floors Layer 1's adaptive threshold.
    window_duration : float
        Layer 1/3 window length, in seconds.
    amplitude_highpass_hz, ica_highpass_hz, muscle_highpass_hz, cardiac_highpass_hz : float or None
        Pre-filter cutoff for each layer's own scan; None skips filtering.
    ica_n_components : int
        ICA components to fit, clamped to the available channel count.
    ica_random_state : int
        Random state for reproducible ICA fits.
    ica_blink_epoch_threshold, ica_whole_recording_threshold : float
        z-score threshold for the two EOG correlation passes; lower = more
        sensitive. Reach for these if a component you can see is clearly
        ocular still isn't flagged -- usually too few ICA components
        diluting the z-score statistics.
    muscle_band, muscle_reference_band : tuple of (float, float)
        Muscle-band and reference-band frequency ranges (Hz). See
        eeg_autoclean.muscle for why they're 20-40 and 1-20 by default.
    muscle_persistence_n, cardiac_persistence_n : int
        Confirm a flag only if the same channel is flagged in at least this
        many consecutive windows; isolated single-window flags are dropped.
        Default 1 (no requirement) for muscle, 2 for cardiac -- see
        eeg_autoclean.muscle / eeg_autoclean.cardiac for the validation
        behind that difference.
    cardiac_min_bpm, cardiac_max_bpm : float
        Heart-rate range (bpm) searched for a periodicity peak.
    cardiac_window_duration : float
        Layer 4 window length, in seconds -- longer than window_duration,
        since detecting periodicity needs several full pulse cycles.
    cardiac_ecg_correlation_threshold : float
        Correlation threshold for the optional ECG cross-correlation path.
    cardiac_ecg_max_lag_s : float
        Lag range (seconds) searched for that cross-correlation.

    Returns
    -------
    results : dict
        {
            "annotations": mne.Annotations, Layer 1/3/4 combined,
            "ica_components_flagged": list of int, or None if Layer 2 skipped,
            "ica_detection_paths": dict[int, list[str]] mapping each flagged
                component to which pass(es) ("blink_epochs", "whole_recording")
                caught it, or None if Layer 2 was skipped,
            "ica": fitted mne.preprocessing.ICA, or None if Layer 2 was skipped,
            "cardiac_used_ecg_channel": bool, whether Layer 4 had an ECG
                channel available for its second detection path,
            "notes": list of str with any skip/warning messages,
        }
    """
    if not raw.preload:
        raw.load_data()

    annotations = _scan_amplitude_and_flat_windows(
        raw,
        amplitude_threshold_uv=amplitude_threshold_uv,
        amplitude_mad_k=amplitude_mad_k,
        flat_threshold_uv=flat_threshold_uv,
        window_duration=window_duration,
        highpass_hz=amplitude_highpass_hz,
    )

    muscle_annotations = _scan_muscle_windows(
        raw,
        muscle_mad_k=muscle_mad_k,
        muscle_band=muscle_band,
        muscle_reference_band=muscle_reference_band,
        window_duration=window_duration,
        highpass_hz=muscle_highpass_hz,
        persistence_n=muscle_persistence_n,
    )
    annotations = annotations + muscle_annotations

    cardiac_annotations, cardiac_used_ecg_channel, cardiac_note = _scan_cardiac_windows(
        raw,
        cardiac_mad_k=cardiac_mad_k,
        min_bpm=cardiac_min_bpm,
        max_bpm=cardiac_max_bpm,
        window_duration=cardiac_window_duration,
        highpass_hz=cardiac_highpass_hz,
        ecg_correlation_threshold=cardiac_ecg_correlation_threshold,
        ecg_max_lag_s=cardiac_ecg_max_lag_s,
        persistence_n=cardiac_persistence_n,
    )
    annotations = annotations + cardiac_annotations

    ica, ica_components_flagged, ica_detection_paths, ica_note = _detect_ocular_ica_components(
        raw,
        ica_n_components=ica_n_components,
        ica_random_state=ica_random_state,
        ica_highpass_hz=ica_highpass_hz,
        ica_blink_epoch_threshold=ica_blink_epoch_threshold,
        ica_whole_recording_threshold=ica_whole_recording_threshold,
    )

    notes = []
    if ica_note is not None:
        notes.append(ica_note)
    if cardiac_note is not None:
        notes.append(cardiac_note)

    return {
        "annotations": annotations,
        "ica_components_flagged": ica_components_flagged,
        "ica_detection_paths": ica_detection_paths,
        "ica": ica,
        "cardiac_used_ecg_channel": cardiac_used_ecg_channel,
        "notes": notes,
    }
