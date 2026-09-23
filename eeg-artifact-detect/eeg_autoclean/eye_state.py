"""Occipital alpha-band eye-state detection (eyes open vs. eyes closed).

Resting-state EEG shows a well-known "alpha blocking" phenomenon: a
posterior dominant rhythm around 8-12 Hz appears over occipital/parietal
cortex when the eyes are closed and relaxed, and largely disappears when the
eyes open. This module detects that alpha "bump" relative to its own
flanking frequency bands (rather than against some fixed absolute power
value), so it needs no per-subject or per-recording amplitude calibration.
"""

import mne
import numpy as np

# Occipital channel names this module looks for, matched case-insensitively
# after stripping trailing dots/whitespace -- e.g. PhysioNet's eegbci
# dataset ships names like "O1..", "Oz..", "O2.." (padded to 4 characters).
OCCIPITAL_CHANNELS = ("O1", "O2", "OZ")

# Alpha band and its flanking baseline bands. Both flanks are the same width
# (4 Hz) as the alpha band and immediately adjacent to it on either side, so
# the "bump ratio" below reflects a genuine spectral peak rather than the
# broadband 1/f trend -- a wider or more distant flank would mostly pick up
# unrelated slope instead of the alpha peak itself.
ALPHA_BAND = (8.0, 12.0)
LOWER_FLANK_BAND = (4.0, 8.0)  # theta, immediately below alpha
UPPER_FLANK_BAND = (12.0, 16.0)  # low beta, immediately above alpha

# Default classification threshold for the bump ratio (alpha-band power /
# mean flanking-band power). Rationale: with narrow, immediately-adjacent
# flanking bands, the 1/f background trend alone contributes only a mild
# multiplicative difference across this frequency range, so a flat
# spectrum with no real peak sits close to a ratio of 1.0-1.3. A genuine
# posterior alpha rhythm during eyes-closed rest is well documented to
# produce a sharp, large-amplitude peak -- typically reported as roughly a
# 2x-5x power increase over neighboring bands for subjects with a normal
# (non-suppressed) alpha rhythm. 1.5 sits with margin above the flat-
# spectrum baseline and comfortably below the typical genuine-peak range,
# favoring not calling eyes_closed unless there's a real, unambiguous bump.
DEFAULT_BUMP_RATIO_THRESHOLD = 1.5


def _find_occipital_picks(raw):
    """EEG channel indices whose (normalized) name is O1/O2/Oz-equivalent."""
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    return [p for p in eeg_picks if raw.ch_names[p].strip(". \t").upper() in OCCIPITAL_CHANNELS]


def detect_eye_state(
    raw,
    bump_ratio_threshold=DEFAULT_BUMP_RATIO_THRESHOLD,
    alpha_band=ALPHA_BAND,
    lower_flank_band=LOWER_FLANK_BAND,
    upper_flank_band=UPPER_FLANK_BAND,
):
    """Classify a recording as eyes_closed or eyes_open via occipital alpha power.

    For each available O1/O2/Oz-equivalent channel, computes a Welch PSD and
    a "bump ratio": mean power in `alpha_band` divided by the mean of the
    mean power in `lower_flank_band` and `upper_flank_band`. The recording
    is classified as eyes_closed if the ratio averaged across available
    occipital channels exceeds `bump_ratio_threshold`, else eyes_open.

    Missing occipital channels are handled gracefully: channels are matched
    case-insensitively with trailing dots/whitespace stripped (some montages,
    e.g. PhysioNet's eegbci, pad channel names with dots), any subset of
    O1/O2/Oz that is present is used, and if none are present eye-state
    detection is skipped rather than raising.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data to analyze. Loaded into memory if not already.
    bump_ratio_threshold : float
        Bump ratio above which the recording is classified eyes_closed. See
        DEFAULT_BUMP_RATIO_THRESHOLD's docstring comment for the reasoning
        behind the default value.
    alpha_band, lower_flank_band, upper_flank_band : tuple of (float, float)
        Frequency bands in Hz, as (low, high) with low inclusive and high
        exclusive.

    Returns
    -------
    result : dict
        {
            "state": "eyes_closed", "eyes_open", or None if no occipital
                channels were found,
            "bump_ratio": float, the bump ratio averaged across occipital
                channels used, or None if none were found/usable,
            "channel_ratios": dict[str, float], per-channel bump ratio,
            "channels_used": list of str, occipital channels the ratio was
                averaged over,
            "notes": list of str, e.g. explaining a skip or an excluded
                channel.
        }
    """
    if not raw.preload:
        raw.load_data()

    picks = _find_occipital_picks(raw)
    if len(picks) == 0:
        return {
            "state": None,
            "bump_ratio": None,
            "channel_ratios": {},
            "channels_used": [],
            "notes": ["No occipital channels (O1/O2/Oz-equivalent) found; eye-state detection skipped."],
        }

    ch_names = [raw.ch_names[p] for p in picks]
    fmin, fmax = lower_flank_band[0], upper_flank_band[1]
    spectrum = raw.compute_psd(method="welch", picks=picks, fmin=fmin, fmax=fmax, verbose=False)
    psd = spectrum.get_data()  # (n_channels, n_freqs)
    freqs = spectrum.freqs

    def _band_power(band):
        mask = (freqs >= band[0]) & (freqs < band[1])
        return psd[:, mask].mean(axis=1)

    alpha_power = _band_power(alpha_band)
    flank_power = (_band_power(lower_flank_band) + _band_power(upper_flank_band)) / 2.0

    channel_ratios = {}
    notes = []
    usable_ratios = []
    for ch_name, alpha_p, flank_p in zip(ch_names, alpha_power, flank_power):
        if flank_p <= 0:
            # Zero flanking-band power means a flat/disconnected channel
            # (see detector.py's Layer 1 for the same concept); the ratio is
            # undefined, not informative, so exclude it rather than dividing
            # by zero.
            notes.append(f"{ch_name}: flanking-band power is zero (flat/disconnected channel?); excluded.")
            continue
        ratio = float(alpha_p / flank_p)
        channel_ratios[ch_name] = ratio
        usable_ratios.append(ratio)

    if not usable_ratios:
        return {
            "state": None,
            "bump_ratio": None,
            "channel_ratios": channel_ratios,
            "channels_used": [],
            "notes": notes + ["No occipital channel had usable (non-zero) flanking-band power."],
        }

    mean_ratio = float(np.mean(usable_ratios))
    state = "eyes_closed" if mean_ratio > bump_ratio_threshold else "eyes_open"

    return {
        "state": state,
        "bump_ratio": mean_ratio,
        "channel_ratios": channel_ratios,
        "channels_used": list(channel_ratios.keys()),
        "notes": notes,
    }
