"""Opt-in cleaning/removal, built on top of detect_artifacts()'s
non-destructive annotations.

detect_artifacts() itself never modifies or removes anything -- that stays
the default behavior of this project. apply_cleaning() is a separate,
explicit step a caller chooses to run when they actually want a cleaned
Raw object (e.g. before downstream analysis that can't tolerate annotated
bad segments/channels/components), not something detect_artifacts() ever
does on its own.

Three independently-selectable actions, all off by default:
- remove_ocular_ica: actually apply Layer 2's flagged ICA components'
  removal (ica.apply(raw, exclude=...)), instead of just leaving them
  flagged.
- drop_bad_channels: for channels flagged bad (Layer 1/3/4 annotations)
  more than `bad_channel_fraction_threshold` of the recording, either
  exclude them or interpolate them -- see DEFAULT_BAD_CHANNEL_FRACTION_THRESHOLD
  and bad_channel_mode's docstring for the exclude-vs-interpolate tradeoff.
- crop_bad_segments: physically remove time segments covered by Layer
  1/3/4 annotations, by extracting and concatenating the complementary
  clean segments.

Order matters and is fixed (not caller-configurable): ICA removal runs
first, while the channel set still matches what ICA was fit on; channel
dropping/interpolation runs next; segment cropping runs last, and only
considers bad segments tied to channels still present after any channel
dropping (a segment that was only bad because of an already-dropped
channel has nothing left to crop for).

Caveat: BAD_cardiac spans are 6.0s each (Layer 4's window, vs Layer 1/3's
1.0s), and Layer 4 has a meaningfully lower precision than the other
layers (see eeg_autoclean.cardiac and README), so a handful of Layer 4
false positives can crop away a disproportionate amount of the recording
(one dense/severe test run: 6 of 10 possible cardiac windows flagged, 47
of 60s cropped). That's not a bug -- Layer 4's per-window behavior is
unchanged and exactly as documented -- but it is a real consequence of
including a lower-precision, larger-window layer in crop_bad_segments's
defaults. Pass a narrower `crop_descriptions` (e.g. leaving out
"BAD_cardiac") if that's too aggressive.
"""

import mne

DEFAULT_CLEANING_DESCRIPTIONS = ("BAD_amplitude", "BAD_flat", "BAD_muscle", "BAD_cardiac")

# A channel bad for more than half the recording is treated as
# fundamentally unreliable rather than something worth salvaging piecemeal.
# Below this threshold, crop_bad_segments already removes that channel's
# specific bad intervals while keeping it around for the rest of the
# recording; above it, so little of the channel is trustworthy that
# per-segment rejection would be doing nearly all the work anyway, so
# dropping/interpolating the whole channel is simpler and no less honest
# about what's usable.
DEFAULT_BAD_CHANNEL_FRACTION_THRESHOLD = 0.5


def _merged_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for start, stop in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return [(start, stop) for start, stop in merged]


def _channel_bad_fractions(annotations, ch_names, duration_s, descriptions):
    """Per-channel fraction of `duration_s` covered by merged annotation
    spans matching `descriptions` for that channel.
    """
    per_channel_spans = {ch: [] for ch in ch_names}
    for onset, duration, desc, chs in zip(
        annotations.onset, annotations.duration, annotations.description, annotations.ch_names
    ):
        if desc not in descriptions:
            continue
        for ch in chs:
            if ch in per_channel_spans:
                per_channel_spans[ch].append((onset, onset + duration))

    fractions = {}
    for ch, spans in per_channel_spans.items():
        merged = _merged_intervals(spans)
        bad_time = sum(stop - start for start, stop in merged)
        fractions[ch] = (bad_time / duration_s) if duration_s > 0 else 0.0
    return fractions


def _crop_out_bad_segments(raw, bad_spans_absolute, first_time):
    """Physically remove `bad_spans_absolute` (onset, offset) time spans
    (in the same absolute reference as annotation onsets, i.e. including
    `first_time`) from `raw`, by extracting and concatenating the
    complementary clean intervals.

    mne's `reject_by_annotation` parameter (available on get_data(),
    epoching, ICA fitting, etc.) is a *soft* skip -- it excludes annotated
    spans from that one operation without changing the Raw object itself,
    so a later call without it sees the bad data again. That doesn't match
    what's being asked for here: an actually shorter, physically cleaned
    Raw object. mne has no single built-in "delete annotated spans" method
    for Raw, so the standard pattern (crop out each clean interval, then
    mne.concatenate_raws) is used instead.

    Returns
    -------
    cleaned : mne.io.Raw
    total_cropped_s : float
    n_segments_cropped : int
    """
    sfreq = raw.info["sfreq"]
    duration_s = raw.n_times / sfreq
    # raw.crop()'s tmax can't reach duration_s exactly -- the last valid
    # sample sits at (n_times - 1) / sfreq, half a sample short of the
    # nominal duration -- so this, not duration_s, bounds any crop() call.
    max_crop_time = (raw.n_times - 1) / sfreq

    bad_relative = []
    for start, stop in bad_spans_absolute:
        start = max(0.0, start - first_time)
        stop = min(duration_s, stop - first_time)
        if stop > start:
            bad_relative.append((start, stop))
    bad_relative = _merged_intervals(bad_relative)

    if not bad_relative:
        return raw.copy(), 0.0, 0

    good_intervals = []
    cursor = 0.0
    for start, stop in bad_relative:
        if start > cursor:
            good_intervals.append((cursor, min(start, max_crop_time)))
        cursor = max(cursor, stop)
    if cursor < max_crop_time:
        good_intervals.append((cursor, max_crop_time))
    good_intervals = [(start, stop) for start, stop in good_intervals if stop > start]

    if not good_intervals:
        raise ValueError(
            "Cropping every annotated bad segment would remove the entire recording; "
            "nothing clean is left to keep."
        )

    segments = [raw.copy().crop(tmin=start, tmax=stop) for start, stop in good_intervals]
    cleaned = segments[0] if len(segments) == 1 else mne.concatenate_raws(segments, verbose=False)
    total_cropped_s = sum(stop - start for start, stop in bad_relative)
    return cleaned, total_cropped_s, len(bad_relative)


def apply_cleaning(
    raw,
    result,
    remove_ocular_ica=False,
    drop_bad_channels=False,
    bad_channel_mode="exclude",
    bad_channel_fraction_threshold=DEFAULT_BAD_CHANNEL_FRACTION_THRESHOLD,
    bad_channel_descriptions=DEFAULT_CLEANING_DESCRIPTIONS,
    crop_bad_segments=False,
    crop_descriptions=DEFAULT_CLEANING_DESCRIPTIONS,
):
    """Produce an actually-cleaned copy of `raw` from a detect_artifacts()
    result. Every action is off by default -- the caller opts in to each one
    independently. `raw` itself is never modified; a cleaned copy is
    returned.

    Parameters
    ----------
    raw : mne.io.Raw
        The same raw passed to detect_artifacts().
    result : dict
        The dict returned by detect_artifacts(raw).
    remove_ocular_ica : bool
        If True and Layer 2 flagged one or more components, actually apply
        their removal (ica.apply(raw, exclude=...)) rather than leaving
        them flagged only.
    drop_bad_channels : bool
        If True, act on channels flagged bad (by bad_channel_descriptions)
        for more than bad_channel_fraction_threshold of the recording.
    bad_channel_mode : "exclude" or "interpolate"
        "exclude": drop the channel entirely (raw.drop_channels()). Never
        fabricates data; reduces channel count.
        "interpolate": spherical-spline interpolate the channel from its
        neighbors (mne's own raw.interpolate_bads()). Preserves channel
        count/montage consistency for downstream analysis that needs it,
        at the cost of that channel's data now being a model-based
        estimate rather than a real recording. Requires channel position
        info (a montage) -- raises a clear error if none is set, rather
        than a cryptic one from deep inside mne.
        Which to prefer depends on the downstream use: exclude when you'd
        rather have fewer, all-real channels; interpolate when a
        consistent channel count/montage across recordings matters more
        than every channel being unmodified.
    bad_channel_fraction_threshold : float
        Fraction of the recording (by merged annotation time, not raw
        instance count) a channel must be flagged bad for before
        drop_bad_channels acts on it. See module docstring for reasoning
        behind the 0.5 default.
    bad_channel_descriptions : tuple of str
        Annotation descriptions counted toward a channel's bad fraction.
    crop_bad_segments : bool
        If True, physically remove time segments covered by
        crop_descriptions annotations (see _crop_out_bad_segments for why
        this isn't done via reject_by_annotation). Only considers segments
        tied to channels still present after any drop_bad_channels step.
    crop_descriptions : tuple of str
        Annotation descriptions to crop out.

    Returns
    -------
    cleaned : mne.io.Raw
        A cleaned copy. `raw` itself is untouched.
    report : dict
        {
            "ica_components_removed": list of int,
            "channels_dropped": list of str,
            "channels_interpolated": list of str,
            "bad_channel_fractions": dict[str, float] (all channels considered,
                only computed if drop_bad_channels was True),
            "time_cropped_s": float,
            "n_segments_cropped": int,
            "duration_before_s": float,
            "duration_after_s": float,
        }
    """
    if bad_channel_mode not in ("exclude", "interpolate"):
        raise ValueError(f"bad_channel_mode must be 'exclude' or 'interpolate', got {bad_channel_mode!r}.")

    cleaned = raw.copy()
    if not cleaned.preload:
        cleaned.load_data()

    report = {
        "ica_components_removed": [],
        "channels_dropped": [],
        "channels_interpolated": [],
        "bad_channel_fractions": {},
        "time_cropped_s": 0.0,
        "n_segments_cropped": 0,
        "duration_before_s": raw.n_times / raw.info["sfreq"],
    }

    # 1. Ocular ICA removal -- must run before any channel dropping, while
    # the channel set still matches what ICA was fit on.
    if remove_ocular_ica:
        ica = result.get("ica")
        components = result.get("ica_components_flagged")
        if ica is not None and components:
            ica.apply(cleaned, exclude=components, verbose=False)
            report["ica_components_removed"] = list(components)

    # 2. Bad channel handling (uses the ORIGINAL result's annotations, not
    # anything from the already-modified `cleaned`, so the decision doesn't
    # depend on cleaning step order).
    if drop_bad_channels:
        duration_s = raw.n_times / raw.info["sfreq"]
        fractions = _channel_bad_fractions(result["annotations"], cleaned.ch_names, duration_s, bad_channel_descriptions)
        report["bad_channel_fractions"] = fractions
        bad_channels = [ch for ch, frac in fractions.items() if frac > bad_channel_fraction_threshold]

        if bad_channels:
            if bad_channel_mode == "exclude":
                cleaned.drop_channels(bad_channels)
                report["channels_dropped"] = bad_channels
            else:
                cleaned.info["bads"] = bad_channels
                try:
                    cleaned.interpolate_bads(reset_bads=True, verbose=False)
                except Exception as exc:
                    raise ValueError(
                        f"Could not interpolate bad channels {bad_channels}: {exc}. "
                        "interpolate_bads() needs channel position info (a montage) set on "
                        "raw.info -- for data without one (e.g. synthetic test data), use "
                        "bad_channel_mode='exclude' instead."
                    ) from exc
                report["channels_interpolated"] = bad_channels

    # 3. Crop bad segments -- only for annotations tied to a channel still
    # present, so a segment that was only bad because of an already-dropped
    # channel isn't cropped for no remaining reason.
    if crop_bad_segments:
        remaining = set(cleaned.ch_names)
        spans = []
        for onset, duration, desc, chs in zip(
            result["annotations"].onset,
            result["annotations"].duration,
            result["annotations"].description,
            result["annotations"].ch_names,
        ):
            if desc not in crop_descriptions:
                continue
            if not any(ch in remaining for ch in chs):
                continue
            spans.append((onset, onset + duration))

        cleaned, total_cropped_s, n_segments = _crop_out_bad_segments(cleaned, spans, raw.first_time)
        report["time_cropped_s"] = total_cropped_s
        report["n_segments_cropped"] = n_segments

    report["duration_after_s"] = cleaned.n_times / cleaned.info["sfreq"]
    return cleaned, report
