"""Precision/recall/F1 scoring of detect_artifacts() output against known
ground truth (currently: synthetic ground truth from eeg_autoclean.synthetic).
"""

import mne
import numpy as np


def _prf1(tp, fp, fn):
    """Precision/recall/F1 with the standard edge-case conventions.

    - No predictions and nothing to find -> precision 1.0 (correctly abstained).
    - No predictions but something to find -> precision 0.0 (missed everything).
    - Nothing to find -> recall 1.0 (vacuously perfect).
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_layer1(raw, result, ground_truth):
    """Score Layer 1 (amplitude/flat window detection) against known injections.

    Ground truth and predictions are both compared at (channel, window)
    granularity, using the same window length detect_artifacts() used.

    Frontal channels (ground_truth["frontal_channels"]) are excluded from
    the scored grid: by construction (see synthetic.generate_synthetic_raw),
    those channels carry genuine blink-leakage deflections that Layer 1's
    amplitude scan may legitimately pick up on, but which were never part of
    the Layer 1 (amplitude/flat) ground truth -- that's Layer 2's target.
    Scoring them here would count correct blink-related detections as false
    positives against a ground truth that was never meant to cover them.

    Returns
    -------
    metrics : dict
        {
            "BAD_amplitude": {"tp", "fp", "fn", "precision", "recall", "f1"},
            "BAD_flat": {...},
        }
    """
    sfreq = raw.info["sfreq"]
    window_duration = ground_truth["window_duration"]
    window_samples = int(round(window_duration * sfreq))
    n_windows = raw.n_times // window_samples
    eeg_ch_names = [raw.ch_names[p] for p in mne.pick_types(raw.info, eeg=True, exclude=[])]
    frontal_channels = set(ground_truth.get("frontal_channels", []))
    scored_ch_names = [ch for ch in eeg_ch_names if ch not in frontal_channels]

    predicted = {"BAD_amplitude": set(), "BAD_flat": set()}
    annotations = result["annotations"]
    for onset, desc, chs in zip(annotations.onset, annotations.description, annotations.ch_names):
        if desc not in predicted:
            continue
        w = int(round((onset - raw.first_time) / window_duration))
        for ch in chs:
            predicted[desc].add((ch, w))

    truth = {
        "BAD_amplitude": set(ground_truth["amplitude_events"]),
        "BAD_flat": set(ground_truth["flat_events"]),
    }

    # Sanity bound: every (channel, window) cell not in truth is a true negative.
    _all_cells = {(ch, w) for ch in scored_ch_names for w in range(n_windows)}

    metrics = {}
    for key in ("BAD_amplitude", "BAD_flat"):
        pred = predicted[key] & _all_cells  # ignore any predictions outside the scored grid
        tp = len(pred & truth[key])
        fp = len(pred - truth[key])
        fn = len(truth[key] - pred)
        precision, recall, f1 = _prf1(tp, fp, fn)
        metrics[key] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
    return metrics


def evaluate_layer3(raw, result, ground_truth):
    """Score Layer 3 (muscle window detection) against known injections.

    Same (channel, window) grid and frontal-channel exclusion as
    evaluate_layer1, for the same reason: muscle bursts are only ever
    injected on non-frontal channels (see synthetic.generate_synthetic_raw),
    so frontal channels were never part of this ground truth either.

    Returns
    -------
    metrics : dict
        {"tp", "fp", "fn", "precision", "recall", "f1"}
    """
    sfreq = raw.info["sfreq"]
    window_duration = ground_truth["window_duration"]
    window_samples = int(round(window_duration * sfreq))
    n_windows = raw.n_times // window_samples
    eeg_ch_names = [raw.ch_names[p] for p in mne.pick_types(raw.info, eeg=True, exclude=[])]
    frontal_channels = set(ground_truth.get("frontal_channels", []))
    scored_ch_names = [ch for ch in eeg_ch_names if ch not in frontal_channels]

    predicted = set()
    annotations = result["annotations"]
    for onset, desc, chs in zip(annotations.onset, annotations.description, annotations.ch_names):
        if desc != "BAD_muscle":
            continue
        w = int(round((onset - raw.first_time) / window_duration))
        for ch in chs:
            predicted.add((ch, w))

    truth = set(ground_truth.get("muscle_events", []))
    _all_cells = {(ch, w) for ch in scored_ch_names for w in range(n_windows)}

    pred = predicted & _all_cells
    tp = len(pred & truth)
    fp = len(pred - truth)
    fn = len(truth - pred)
    precision, recall, f1 = _prf1(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate_layer4(raw, result, ground_truth):
    """Score Layer 4 (cardiac window detection) against known injections.

    Same frontal-channel exclusion as evaluate_layer1/3, for the same
    reason: cardiac pulse trains are only ever injected on non-frontal
    channels. Unlike Layer 1/3, the (channel, window) grid here uses
    ground_truth["cardiac_window_duration"] (several times longer than the
    1.0s grid) since that's the granularity Layer 4 actually scans and
    annotates at -- see eeg_autoclean.cardiac for why.

    Returns
    -------
    metrics : dict
        {"tp", "fp", "fn", "precision", "recall", "f1"}
    """
    sfreq = raw.info["sfreq"]
    window_duration = ground_truth["cardiac_window_duration"]
    window_samples = int(round(window_duration * sfreq))
    n_windows = raw.n_times // window_samples
    eeg_ch_names = [raw.ch_names[p] for p in mne.pick_types(raw.info, eeg=True, exclude=[])]
    frontal_channels = set(ground_truth.get("frontal_channels", []))
    scored_ch_names = [ch for ch in eeg_ch_names if ch not in frontal_channels]

    predicted = set()
    annotations = result["annotations"]
    for onset, desc, chs in zip(annotations.onset, annotations.description, annotations.ch_names):
        if desc != "BAD_cardiac":
            continue
        w = int(round((onset - raw.first_time) / window_duration))
        for ch in chs:
            predicted.add((ch, w))

    truth = set(ground_truth.get("cardiac_events", []))
    _all_cells = {(ch, w) for ch in scored_ch_names for w in range(n_windows)}

    pred = predicted & _all_cells
    tp = len(pred & truth)
    fp = len(pred - truth)
    fn = len(truth - pred)
    precision, recall, f1 = _prf1(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate_layer2(raw, result, ground_truth, correlation_threshold=0.5):
    """Score Layer 2 (ICA ocular component detection) against a known blink source.

    ICA component ordering/identity is data-driven, so we can't check "is
    component index X flagged" against a fixed ground-truth index. Instead we
    reconstruct the true injected blink waveform and correlate it against
    each flagged component's source time course; a flagged component counts
    as a true positive if the absolute Pearson correlation is at least
    `correlation_threshold`.

    Precision is scored at component granularity (what fraction of flagged
    components are truly ocular); recall is scored at the run/event level
    (was the true ocular source recovered by at least one flagged component).

    Returns
    -------
    metrics : dict
        {
            "n_flagged": int,
            "n_true_positive_components": int,
            "precision": float,
            "recall": float,
            "f1": float,
            "best_correlation": float,
        }
    """
    ica = result.get("ica")
    flagged = result.get("ica_components_flagged") or []

    times = raw.times
    true_waveform = np.zeros(len(times))
    for center in ground_truth["blink_times"]:
        true_waveform += ground_truth["blink_amplitude_uv"] * np.exp(
            -0.5 * ((times - center) / ground_truth["blink_width_s"]) ** 2
        )

    if ica is None or len(flagged) == 0:
        return {
            "n_flagged": 0,
            "n_true_positive_components": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "best_correlation": 0.0,
        }

    sources = ica.get_sources(raw).get_data(picks=flagged)
    correlations = [abs(np.corrcoef(source, true_waveform)[0, 1]) for source in sources]

    n_tp = sum(1 for r in correlations if r >= correlation_threshold)
    precision = n_tp / len(flagged)
    recall = 1.0 if n_tp > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "n_flagged": len(flagged),
        "n_true_positive_components": n_tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "best_correlation": max(correlations),
    }
