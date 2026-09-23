"""Consensus mode: run our own detect_artifacts() alongside a second,
independently-built artifact detector (autoreject's AutoReject) to surface
artifacts our detector catches that AutoReject's method structurally
cannot see.

That is a deliberately narrower claim than "general-purpose agreement/
disagreement review tool," and the scope is narrow on purpose -- see
"Validated" below for exactly what is and is not backed by evidence.
Ours and AutoReject are built on different assumptions (ours:
amplitude/spectral/periodicity signatures per layer; AutoReject:
statistical, cross-validated per-epoch/per-channel thresholds) and so have
different blind spots. A cell WE flag that AutoReject doesn't is a
meaningful, validated signal: something spectral/periodicity-shaped
(muscle, cardiac) that an amplitude/covariance-based method is structurally
unlikely to catch. The reverse direction -- a cell AutoReject flags that we
don't -- is NOT validated as meaningful; see below before reading anything
into it.

This module is a separate, OPT-IN add-on. detect_artifacts() remains the
project's core, primary API and is unaffected by any of this -- consensus
mode uses it as one of the two methods being cross-checked, not a
replacement for it.

Method:
1. Run our own detect_artifacts() to get Layer 1/3/4's time/channel
   annotations (BAD_amplitude, BAD_flat, BAD_muscle, BAD_cardiac). Layer 2
   is deliberately excluded here: it flags ICA COMPONENTS, not a
   (channel, window) cell, so it has no direct per-channel-window "bad"
   flag to compare against AutoReject's per-channel-epoch judgment without
   a separate projection step -- out of scope for this first version,
   documented rather than silently dropped.
2. Epoch the raw into fixed-length windows (see DEFAULT_CONSENSUS_WINDOW_DURATION
   for why the length is what it is) and run AutoReject on them, using its
   RejectLog (per-epoch bad-epoch flag, per-(epoch, channel) good/bad/
   interpolated labels) as the second method's per-cell judgment.
3. Align both methods onto the same (channel, window) grid -- reusing
   eeg_autoclean._windowing's window-bounds convention -- and classify
   every cell as AGREE_BAD (both flag it), AGREE_CLEAN (neither does), or
   DISAGREE (exactly one does).

AutoReject requires real channel positions (a montage) to do its spatial
interpolation-based reasoning; run_consensus() checks for one up front and
raises a clear error if missing, rather than surfacing AutoReject's own
internal RuntimeError.

Validated against the synthetic battery (known ground truth) and MNE's
sample dataset (real, no ground truth) via scripts/validate_consensus.py:

VALIDATED -- ours-only disagreements are reliably real artifacts: 21-29 of
each synthetic scenario's ours-only disagreements land on a known
ground-truth cell (e.g. 29/31 in "moderate", 21/24 in "dense/severe"),
vs. only 1-3 that don't, in every scenario. This is the module's actual,
evidence-backed value: our detector catches spectral/periodicity-based
artifact types (muscle, cardiac contamination) that AutoReject's
amplitude/covariance-based method structurally cannot see, and this cell
category reliably reflects that rather than noise.

NOT VALIDATED -- autoreject-only disagreements, which numerically DOMINATE
the DISAGREE category, are not shown to mean anything in particular.
Across all 4 synthetic scenarios, cells AutoReject flagged that we didn't
matched a known ground-truth artifact ZERO times (out of 280-350 such
cells per scenario). A small visual spot-check of 3 such cells on the real
sample dataset found no visually obvious artifact in any of them either
(see README's Consensus mode section for specifics). This does NOT prove
AutoReject is wrong -- it may be flagging genuine statistical outliers
outside this project's specific artifact taxonomy, or simply doing its
designed job (protecting a downstream ERP average against any atypical
epoch, a broader and more conservative goal than matching a specific
artifact category). It also does NOT mean these cells are the "actionable,
ambiguous" cases a reviewer should prioritize -- that claim has no
evidence behind it here. AutoReject's much higher overall flag rate on
this data (roughly 5-10x ours, checked at both 1.0s and 2.0s epochs, so
not just an artifact of epoch length) is unexplained territory: not
confirmed as noise, not confirmed as signal, genuinely open. Treat
autoreject-only DISAGREE cells as exactly that -- a disagreement -- and
nothing more specific than that until this is investigated further.
"""

import mne
import numpy as np

from ._windowing import DEFAULT_WINDOW_DURATION
from .detector import detect_artifacts

try:
    from autoreject import AutoReject

    _AUTOREJECT_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover -- exercised only when autoreject isn't installed
    AutoReject = None
    _AUTOREJECT_IMPORT_ERROR = exc

AGREE_BAD = "AGREE_BAD"
AGREE_CLEAN = "AGREE_CLEAN"
DISAGREE = "DISAGREE"

# Matches Layer 1/3's own window length, for two reasons: it's the
# project's existing convention (see _windowing.py), and it lets our own
# per-window flags line up with AutoReject's per-epoch flags one-to-one
# without resampling either grid. AutoReject is more commonly run on
# longer, event-locked epochs in ERP/cognitive-task contexts, but for
# continuous-recording quality control -- what both this project and
# consensus mode are for -- a short fixed-length epoch is standard practice
# (it's exactly what mne.make_fixed_length_epochs() is for), and 1.0s
# still gives AutoReject enough samples per epoch for its per-channel
# variance-based statistics to be meaningful.
DEFAULT_CONSENSUS_WINDOW_DURATION = DEFAULT_WINDOW_DURATION

# Our own annotation descriptions counted as "bad" for consensus purposes.
# See module docstring for why Layer 2 (ocular, component-based) isn't here.
OUR_CONSENSUS_DESCRIPTIONS = ("BAD_amplitude", "BAD_flat", "BAD_muscle", "BAD_cardiac")


def _require_autoreject():
    if AutoReject is None:
        raise ImportError(
            "Consensus mode requires the 'autoreject' package, which is not installed. "
            "Install it with: pip install autoreject  (or: pip install eeg_autoclean[consensus])"
        ) from _AUTOREJECT_IMPORT_ERROR


def _our_flagged_grid(raw, window_duration, ch_names, detect_kwargs):
    """Run detect_artifacts() and return a boolean (n_channels, n_windows)
    grid on the given window_duration grid, plus the full result dict.

    An annotation whose own duration spans more than one grid window (e.g.
    Layer 4's 6.0s cardiac window on a 1.0s grid) marks every grid window
    it covers, read directly from the annotation's actual duration rather
    than assuming a fixed window size -- this is what lets Layer 1/3/4's
    different native window lengths all land correctly on one grid.
    """
    result = detect_artifacts(raw, window_duration=window_duration, **detect_kwargs)

    sfreq = raw.info["sfreq"]
    window_samples = int(round(window_duration * sfreq))
    n_windows = int(np.ceil(raw.n_times / window_samples))
    ch_index = {ch: i for i, ch in enumerate(ch_names)}

    grid = np.zeros((len(ch_names), n_windows), dtype=bool)
    annotations = result["annotations"]
    for onset, duration, desc, chs in zip(
        annotations.onset, annotations.duration, annotations.description, annotations.ch_names
    ):
        if desc not in OUR_CONSENSUS_DESCRIPTIONS:
            continue
        w0 = int(round((onset - raw.first_time) / window_duration))
        n_span = max(1, int(round(duration / window_duration)))
        for offset in range(n_span):
            w = w0 + offset
            if not (0 <= w < n_windows):
                continue
            for ch in chs:
                if ch in ch_index:
                    grid[ch_index[ch], w] = True

    return grid, result


def _autoreject_flagged_grid(raw, window_duration, ch_names, ar_kwargs):
    """Epoch raw into fixed-length windows and run AutoReject, returning a
    boolean (n_channels, n_windows) grid aligned to `ch_names`'s order.

    reject_by_annotation=False on the epoching step: AutoReject is meant to
    be an INDEPENDENT second opinion, so it must see the whole raw signal,
    not one that's already had our own tool's bad segments silently
    excluded (which would happen if the caller had attached our
    annotations to `raw` before calling this).
    """
    _require_autoreject()

    epochs = mne.make_fixed_length_epochs(
        raw, duration=window_duration, preload=True, reject_by_annotation=False, verbose=False
    )
    epochs.pick(ch_names)

    ar = AutoReject(picks=mne.pick_types(epochs.info, eeg=True, exclude=[]), random_state=97, verbose=False, **ar_kwargs)
    _epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)

    n_windows = len(reject_log.bad_epochs)
    ch_index = {ch: i for i, ch in enumerate(reject_log.ch_names)}
    grid = np.zeros((len(ch_names), n_windows), dtype=bool)
    for w in range(n_windows):
        if reject_log.bad_epochs[w]:
            grid[:, w] = True  # whole epoch judged unsalvageable -> every channel counts as bad here
            continue
        for ch_name, out_idx in ((ch, i) for i, ch in enumerate(ch_names) if ch in ch_index):
            if reject_log.labels[w, ch_index[ch_name]] in (1, 2):
                grid[out_idx, w] = True

    return grid, reject_log


def run_consensus(raw, methods=("ours", "autoreject"), window_duration=DEFAULT_CONSENSUS_WINDOW_DURATION, detect_kwargs=None, ar_kwargs=None):
    """Cross-check detect_artifacts() against AutoReject, per (channel,
    window) cell, to surface artifacts our detector catches that
    AutoReject's method structurally cannot see.

    See this module's docstring for what's actually validated: cells this
    result's `disagreements` list marks `flagged_by="ours"` are reliably
    real artifacts (validated against synthetic ground truth). Cells marked
    `flagged_by="autoreject"` are NOT validated as meaningful -- treat them
    as a disagreement and nothing more specific than that.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data to analyze. Must have channel positions set (a
        montage) -- AutoReject needs these for its spatial reasoning; a
        clear error is raised up front if none is found.
    methods : tuple of str
        Which methods to cross-check. Currently only ("ours", "autoreject")
        (the default, and the only supported value) -- kept as a parameter
        rather than hardcoded so a third method has an obvious place to
        plug in later, but this version doesn't implement anything else.
    window_duration : float
        Grid granularity in seconds. See DEFAULT_CONSENSUS_WINDOW_DURATION
        for why 1.0s is the default.
    detect_kwargs : dict or None
        Extra keyword arguments forwarded to our own detect_artifacts().
    ar_kwargs : dict or None
        Extra keyword arguments forwarded to autoreject.AutoReject's
        constructor (e.g. {"cv": 5} to trade accuracy for speed).

    Returns
    -------
    result : dict
        {
            "counts": {"AGREE_BAD": int, "AGREE_CLEAN": int, "DISAGREE": int, "total_cells": int},
            "fractions": {"AGREE_BAD": float, "AGREE_CLEAN": float, "DISAGREE": float},
            "disagreements": list of dict, each
                {"channel": str, "window": int, "onset_s": float,
                 "flagged_by": "ours" or "autoreject"},
            "channels": list of str, the channels compared (in grid row order),
            "n_windows": int, the number of windows compared,
            "window_duration": float,
            "our_result": dict, the raw detect_artifacts() return value,
            "autoreject_reject_log": autoreject.RejectLog,
        }
    """
    if tuple(methods) != ("ours", "autoreject"):
        raise ValueError(
            f"run_consensus currently only supports methods=('ours', 'autoreject'), got {methods!r}."
        )

    detect_kwargs = detect_kwargs or {}
    ar_kwargs = ar_kwargs or {}

    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(eeg_picks) == 0:
        raise ValueError("No EEG channels found in raw data; cannot run consensus mode.")
    ch_names = [raw.ch_names[p] for p in eeg_picks]

    if raw.get_montage() is None:
        raise ValueError(
            "Consensus mode requires channel positions (a montage) on `raw` -- AutoReject "
            "needs these for its spatial interpolation-based reasoning. Set one with "
            "raw.set_montage(...) before calling run_consensus() (e.g. "
            "mne.channels.make_standard_montage('standard_1020') for a standard 10-20 layout)."
        )

    if not raw.preload:
        raw.load_data()

    grid_ours, our_result = _our_flagged_grid(raw, window_duration, ch_names, detect_kwargs)
    grid_ar, reject_log = _autoreject_flagged_grid(raw, window_duration, ch_names, ar_kwargs)

    # mne.make_fixed_length_epochs() drops a trailing partial window
    # outright (unlike our own _windowing.compute_window_bounds, which
    # keeps one down to half-length), so the two grids can differ by one
    # window at the very end -- compare only the overlap, which is what
    # both methods actually got to judge.
    n_windows = min(grid_ours.shape[1], grid_ar.shape[1])
    ours = grid_ours[:, :n_windows]
    ar = grid_ar[:, :n_windows]

    agree_bad = ours & ar
    agree_clean = (~ours) & (~ar)
    disagree = ours ^ ar

    disagreements = []
    for ch_idx, ch in enumerate(ch_names):
        for w in range(n_windows):
            if disagree[ch_idx, w]:
                disagreements.append(
                    {
                        "channel": ch,
                        "window": w,
                        "onset_s": raw.first_time + w * window_duration,
                        "flagged_by": "ours" if ours[ch_idx, w] else "autoreject",
                    }
                )

    total_cells = int(ours.size)
    counts = {
        AGREE_BAD: int(agree_bad.sum()),
        AGREE_CLEAN: int(agree_clean.sum()),
        DISAGREE: int(disagree.sum()),
        "total_cells": total_cells,
    }
    fractions = {k: (v / total_cells if total_cells else 0.0) for k, v in counts.items() if k != "total_cells"}

    return {
        "counts": counts,
        "fractions": fractions,
        "disagreements": disagreements,
        "channels": ch_names,
        "n_windows": n_windows,
        "window_duration": window_duration,
        "our_result": our_result,
        "autoreject_reject_log": reject_log,
    }
