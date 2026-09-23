"""Shared summary helpers for real-data validation scripts.

Kept separate from eeg_autoclean.evaluation (which scores against known
synthetic ground truth): there is no ground truth for real recordings, so
these functions only summarize detect_artifacts() output and check
agreement with MNE's own find_bads_eog, they don't compute precision/recall.
"""

import numpy as np


def summarize_layer1(raw, result, window_duration=1.0):
    """Summarize Layer 1 (amplitude/flat) annotations for a real recording.

    Layer 1 emits one annotation per (window, channel) pair, so this reports
    both unique windows flagged (time-based) and the raw per-channel-window
    instance count, to avoid the count of one being mistaken for the other.
    """
    annotations = result["annotations"]
    amplitude_onsets = {o for o, d in zip(annotations.onset, annotations.description) if d == "BAD_amplitude"}
    flat_onsets = {o for o, d in zip(annotations.onset, annotations.description) if d == "BAD_flat"}
    n_amplitude_instances = sum(1 for d in annotations.description if d == "BAD_amplitude")
    n_flat_instances = sum(1 for d in annotations.description if d == "BAD_flat")
    window_samples = round(window_duration * raw.info["sfreq"])
    total_windows = int(np.ceil(raw.n_times / window_samples))
    return {
        "n_amplitude_windows": len(amplitude_onsets),
        "n_flat_windows": len(flat_onsets),
        "n_amplitude_instances": n_amplitude_instances,
        "n_flat_instances": n_flat_instances,
        "total_windows": total_windows,
    }


def summarize_layer2_agreement(raw, result):
    """Compare our flagged ICA components against MNE's own find_bads_eog on raw.

    This is a qualitative sanity/agreement check, not a ground-truth
    accuracy measure: both sides use the same underlying MNE machinery, just
    invoked slightly differently (ours via create_eog_epochs, this one
    directly on the continuous raw).

    Returns None if Layer 2 was skipped (no EOG channel).
    """
    if result["ica"] is None or result["ica_components_flagged"] is None:
        return None

    our_flagged = set(result["ica_components_flagged"])
    eog_indices, _eog_scores = result["ica"].find_bads_eog(raw, verbose=False)
    mne_flagged = set(eog_indices)
    overlap = our_flagged & mne_flagged

    return {
        "our_flagged": sorted(our_flagged),
        "mne_flagged": sorted(mne_flagged),
        "overlap": sorted(overlap),
        "agreement": (len(overlap) / len(mne_flagged)) if mne_flagged else None,
    }
