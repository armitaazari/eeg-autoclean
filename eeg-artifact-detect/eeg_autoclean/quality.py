"""Overall recording quality score: a single 0-100 summary combining all
four detect_artifacts() layers, plus a transparent per-layer breakdown so
the single number is never the only thing available.
"""

import mne
import numpy as np

from ._windowing import DEFAULT_WINDOW_DURATION
from .cardiac import DEFAULT_CARDIAC_WINDOW_DURATION

# Equal weight per layer (25% each), not per flag-type -- Layer 1 has two
# flag types (amplitude, flat) but counts as ONE category here (scored as
# the fraction of (channel, window) cells flagged by EITHER), so it isn't
# implicitly double-weighted relative to Layers 2-4.
#
# Equal weighting is the transparent default: each layer targets a
# genuinely distinct artifact mechanism (gross amplitude/disconnection,
# ocular, muscle, cardiac), and without a specific downstream analysis in
# mind there's no principled basis in this project for calling one
# mechanism worse than another. A real analysis often does have such a
# basis -- an ERP researcher may care far more about ocular contamination
# near stimulus onset, a sleep researcher far more about muscle/EMG -- so
# `weights` is exposed for the caller to override with their own domain
# judgment rather than this module silently baking in an unstated one.
DEFAULT_LAYER_WEIGHTS = {
    "amplitude_flat": 0.25,
    "ocular": 0.25,
    "muscle": 0.25,
    "cardiac": 0.25,
}


def _badness_fraction(annotations, descriptions, ch_names, window_duration, first_time, n_times, sfreq):
    """Fraction of (channel, window) cells flagged with any description in
    `descriptions`, out of all cells on that window grid.

    The actual window duration is read directly off a matching annotation's
    own `duration` field when one exists (every Layer 1/3/4 annotation's
    duration equals the window size that layer scanned with), rather than
    trusting the caller-supplied `window_duration` to match what
    detect_artifacts() was actually called with -- self-consistent even if
    a caller passes a mismatched default. The parameter is only a fallback
    for when nothing of this description was ever flagged, so there's
    nothing to infer from.
    """
    for onset, desc, duration in zip(annotations.onset, annotations.description, annotations.duration):
        if desc in descriptions:
            window_duration = float(duration)
            break

    window_samples = int(round(window_duration * sfreq))
    n_windows = int(np.ceil(n_times / window_samples))
    total_cells = len(ch_names) * n_windows
    if total_cells == 0:
        return 0.0

    flagged = set()
    for onset, desc, chs in zip(annotations.onset, annotations.description, annotations.ch_names):
        if desc not in descriptions:
            continue
        w = int(round((onset - first_time) / window_duration))
        for ch in chs:
            flagged.add((ch, w))
    return len(flagged) / total_cells


def compute_quality_score(
    raw,
    result,
    weights=None,
    window_duration=DEFAULT_WINDOW_DURATION,
    cardiac_window_duration=DEFAULT_CARDIAC_WINDOW_DURATION,
):
    """Combine all four detect_artifacts() layers into a single 0-100 score.

    Each layer's contribution is its own "badness fraction" in [0, 1] -- the
    fraction of (channel, window) cells it flagged (Layers 1, 3, 4), or the
    fraction of fitted ICA components it flagged as ocular (Layer 2) -- so
    every layer is measured on the same scale regardless of how differently
    each one detects things. The score is:

        100 * (1 - weighted average badness fraction)

    i.e. 100 = nothing flagged anywhere, 0 = everything flagged everywhere.

    A layer that was skipped (Layer 2: no EOG channel; Layer 4: recording
    too short for even one of its windows) is left out of the weighted
    average entirely -- its weight is redistributed proportionally across
    the layers that did run, rather than either counting it as 0% badness
    (which would silently reward a recording for having less measurable)
    or keeping its weight allocated to a measurement that never happened.

    Parameters
    ----------
    raw : mne.io.Raw
        The same raw passed to detect_artifacts().
    result : dict
        The dict returned by detect_artifacts(raw).
    weights : dict or None
        Per-layer weights with keys "amplitude_flat", "ocular", "muscle",
        "cardiac". Defaults to DEFAULT_LAYER_WEIGHTS (25% each). Need not
        sum to 1 -- normalized internally over whichever layers actually ran.
    window_duration : float
        Must match the window_duration detect_artifacts() was called with
        (Layer 1/3's grid).
    cardiac_window_duration : float
        Must match the cardiac_window_duration detect_artifacts() was
        called with (Layer 4's grid).

    Returns
    -------
    summary : dict
        {
            "score": float in [0, 100],
            "breakdown": {
                "amplitude_flat": {"badness_fraction": float, "weight": float, "skipped": False},
                "ocular": {"badness_fraction": float or None, "weight": float, "skipped": bool},
                "muscle": {"badness_fraction": float, "weight": float, "skipped": False},
                "cardiac": {"badness_fraction": float or None, "weight": float, "skipped": bool},
            },
        }
    """
    weights = dict(DEFAULT_LAYER_WEIGHTS if weights is None else weights)

    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    ch_names = [raw.ch_names[p] for p in eeg_picks]
    if len(ch_names) == 0:
        raise ValueError("No EEG channels found in raw data; cannot compute a quality score.")

    sfreq = raw.info["sfreq"]
    annotations = result["annotations"]

    breakdown = {
        "amplitude_flat": {
            "badness_fraction": _badness_fraction(
                annotations, {"BAD_amplitude", "BAD_flat"}, ch_names, window_duration, raw.first_time, raw.n_times, sfreq
            ),
            "skipped": False,
        },
        "muscle": {
            "badness_fraction": _badness_fraction(
                annotations, {"BAD_muscle"}, ch_names, window_duration, raw.first_time, raw.n_times, sfreq
            ),
            "skipped": False,
        },
    }

    ica = result.get("ica")
    ica_flagged = result.get("ica_components_flagged")
    if ica is None or ica_flagged is None:
        breakdown["ocular"] = {"badness_fraction": None, "skipped": True}
    else:
        n_components = getattr(ica, "n_components_", None) or len(ica_flagged)
        badness = (len(ica_flagged) / n_components) if n_components else 0.0
        breakdown["ocular"] = {"badness_fraction": badness, "skipped": False}

    cardiac_skipped = any("cardiac detection skipped" in note.lower() for note in result.get("notes", []))
    if cardiac_skipped:
        breakdown["cardiac"] = {"badness_fraction": None, "skipped": True}
    else:
        breakdown["cardiac"] = {
            "badness_fraction": _badness_fraction(
                annotations, {"BAD_cardiac"}, ch_names, cardiac_window_duration, raw.first_time, raw.n_times, sfreq
            ),
            "skipped": False,
        }

    active = {k: v for k, v in breakdown.items() if not v["skipped"]}
    total_weight = sum(weights[k] for k in active)
    weighted_badness = (
        sum(weights[k] * active[k]["badness_fraction"] for k in active) / total_weight if total_weight > 0 else 0.0
    )

    for key, entry in breakdown.items():
        entry["weight"] = weights[key]

    score = max(0.0, min(100.0, 100.0 * (1.0 - weighted_badness)))

    return {"score": score, "breakdown": breakdown}
