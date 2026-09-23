"""Fixed-length windowing shared by Layer 1 (amplitude/flat), Layer 3
(muscle), and Layer 4 (cardiac) scans, so annotation-producing layers line
up at consistent, well-defined granularity for downstream evaluation/
reporting code that reads them.

Kept in its own module (rather than living inside detector.py, which Layer
3/4 would then have to import from) purely to avoid a circular import
between detector.py and muscle.py/cardiac.py, all of which need it.
"""

import numpy as np

DEFAULT_WINDOW_DURATION = 1.0  # seconds


def compute_window_bounds(n_samples, window_samples):
    """Fixed-length (start, stop) sample bounds spanning n_samples, dropping
    a trailing partial window if it's under half the intended length (too
    few samples to characterize either a peak-to-peak amplitude or a
    frequency-band power estimate reliably).
    """
    window_bounds = []
    for start in range(0, n_samples, window_samples):
        stop = min(start + window_samples, n_samples)
        if stop - start < window_samples // 2:
            continue
        window_bounds.append((start, stop))
    return window_bounds


def apply_persistence_filter(flagged, persistence_n):
    """Drop isolated flags: for each channel (row of `flagged`, a boolean
    (n_channels, n_windows) array), only keep a run of consecutive flagged
    windows if the run is at least `persistence_n` windows long.

    Used by Layer 3/muscle and Layer 4/cardiac to require a flag to persist
    across consecutive windows before it's confirmed, rather than trusting
    a single window in isolation -- a genuine muscle/cardiac artifact
    should span multiple windows anyway, so an isolated single-window flag
    is more likely to be the kind of sporadic statistical noise already
    documented for both layers (see their module docstrings) than a real
    artifact.

    persistence_n <= 1 is a no-op (every flagged window is confirmed,
    matching the original, non-persistence behavior) so this can be safely
    left at its default without changing anything.
    """
    if persistence_n <= 1:
        return flagged

    confirmed = np.zeros_like(flagged)
    n_ch, n_windows = flagged.shape
    for ch in range(n_ch):
        run_start = None
        for w in range(n_windows):
            if flagged[ch, w]:
                if run_start is None:
                    run_start = w
            else:
                if run_start is not None and (w - run_start) >= persistence_n:
                    confirmed[ch, run_start:w] = True
                run_start = None
        if run_start is not None and (n_windows - run_start) >= persistence_n:
            confirmed[ch, run_start:n_windows] = True
    return confirmed
