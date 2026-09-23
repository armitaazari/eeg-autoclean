"""Plotting helpers for inspecting eeg_autoclean.detector results."""

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.patches import Patch

# Distinct colors per annotation type, used for both shading and the legend.
ANNOTATION_COLORS = {
    "BAD_amplitude": "tab:red",
    "BAD_flat": "tab:blue",
    "BAD_muscle": "tab:orange",
    "BAD_cardiac": "tab:purple",
}


def _select_representative_picks(raw, n_channels):
    """Pick an evenly-spaced subset of EEG channel indices across the montage."""
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(eeg_picks) == 0:
        raise ValueError("No EEG channels found in raw data; cannot plot.")

    n_channels = min(n_channels, len(eeg_picks))
    positions = np.linspace(0, len(eeg_picks) - 1, n_channels).round().astype(int)
    positions = sorted(set(positions.tolist()))
    return [eeg_picks[i] for i in positions]


def _merged_annotation_spans(annotations, description):
    """Collapse per-channel annotation entries into unique (start, stop) time spans.

    Layer 1 emits one annotation per (window, channel) pair, so many channels
    flagged in the same window would otherwise produce overlapping duplicate
    shading.
    """
    spans = set()
    for onset, duration, desc in zip(annotations.onset, annotations.duration, annotations.description):
        if desc == description:
            spans.add((float(onset), float(onset) + float(duration)))
    return sorted(spans)


def plot_artifacts(raw, result, output_path=None, n_channels=8, duration=15.0, start=0.0, eye_state_result=None):
    """Plot a short EEG segment with detected artifacts highlighted.

    Shows a representative subset of EEG channels over a short time window,
    with BAD_amplitude, BAD_flat (Layer 1), BAD_muscle (Layer 3), and
    BAD_cardiac (Layer 4) annotation regions shaded in distinct colors. If
    Layer 2 flagged one or more ICA components as ocular, a second panel
    below shows their time courses over the same window for visual
    confirmation against the EEG trace. If eye-state detection was run, its
    classification and bump ratio are shown as a compact subtitle rather
    than a separate panel.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data (the same object passed to detect_artifacts).
    result : dict
        The dict returned by detect_artifacts(raw).
    output_path : str or Path or None
        If given, save the figure as a PNG here (dpi=150) instead of
        displaying it interactively.
    n_channels : int
        Number of representative EEG channels to plot.
    duration : float
        Length in seconds of the segment to plot.
    start : float
        Start time in seconds (relative to the recording) of the segment.
    eye_state_result : dict or None
        The dict returned by detect_eye_state(raw), if eye-state detection
        was run. When given and a state was actually classified (not None,
        e.g. skipped for lack of occipital channels), its state and bump
        ratio are added to the plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    picks = _select_representative_picks(raw, n_channels)
    ch_names = [raw.ch_names[p] for p in picks]

    sfreq = raw.info["sfreq"]
    start_sample = max(0, int(round(start * sfreq)))
    stop_sample = min(start_sample + int(round(duration * sfreq)), len(raw.times))
    start_sample = min(start_sample, max(stop_sample - 1, 0))

    data_uv = raw.get_data(picks=picks, start=start_sample, stop=stop_sample) * 1e6
    times = raw.times[start_sample:stop_sample] + raw.first_time

    ica = result.get("ica")
    flagged = result.get("ica_components_flagged") or []
    show_ica_panel = ica is not None and len(flagged) > 0

    if show_ica_panel:
        fig, (ax_eeg, ax_ica) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
    else:
        fig, ax_eeg = plt.subplots(1, 1, figsize=(12, 6))
        ax_ica = None

    # --- EEG traces, stacked top-to-bottom with a fixed vertical offset ---
    centered = data_uv - data_uv.mean(axis=1, keepdims=True)
    spacing = max(float(np.percentile(np.abs(centered), 99)) * 2.2, 20.0)

    for i, ch_name in enumerate(ch_names):
        offset = (len(ch_names) - 1 - i) * spacing
        ax_eeg.plot(times, centered[i] + offset, color="black", linewidth=0.6)

    ax_eeg.set_yticks([(len(ch_names) - 1 - i) * spacing for i in range(len(ch_names))])
    ax_eeg.set_yticklabels(ch_names)
    ax_eeg.set_ylabel("Channel")
    ax_eeg.set_xlim(times[0], times[-1])

    title = f"EEG segment ({times[0]:.1f}-{times[-1]:.1f}s)"
    if eye_state_result is not None and eye_state_result.get("state") is not None:
        state_label = eye_state_result["state"].replace("_", " ").capitalize()
        bump_ratio = eye_state_result["bump_ratio"]
        title += f"  |  {state_label}, alpha bump ratio: {bump_ratio:.2f}"
    ax_eeg.set_title(title)

    # --- Shade Layer 1 annotation spans ---
    annotations = result["annotations"]
    legend_handles = []
    for description, color in ANNOTATION_COLORS.items():
        spans = _merged_annotation_spans(annotations, description)
        added_label = False
        for span_start, span_stop in spans:
            if span_stop < times[0] or span_start > times[-1]:
                continue
            ax_eeg.axvspan(
                max(span_start, times[0]),
                min(span_stop, times[-1]),
                color=color,
                alpha=0.18,
                linewidth=0,
            )
            added_label = True
        if added_label:
            legend_handles.append(Patch(facecolor=color, alpha=0.3, label=description))

    if legend_handles:
        ax_eeg.legend(handles=legend_handles, loc="upper right", fontsize=9)

    # --- Flagged ICA component time courses (Layer 2) ---
    if show_ica_panel:
        sources = ica.get_sources(raw)
        ic_data = sources.get_data(picks=flagged, start=start_sample, stop=stop_sample)
        ic_centered = ic_data - ic_data.mean(axis=1, keepdims=True)
        ic_spacing = max(float(np.percentile(np.abs(ic_centered), 99)) * 2.2, 1.0)

        for i, comp_idx in enumerate(flagged):
            offset = (len(flagged) - 1 - i) * ic_spacing
            ax_ica.plot(times, ic_centered[i] + offset, color="tab:red", linewidth=0.7)

        ax_ica.set_yticks([(len(flagged) - 1 - i) * ic_spacing for i in range(len(flagged))])
        ax_ica.set_yticklabels([f"IC{idx}" for idx in flagged])
        ax_ica.set_ylabel("Flagged ICA\ncomponent(s)")
        ax_ica.set_xlabel("Time (s)")
    else:
        ax_eeg.set_xlabel("Time (s)")

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

    return fig
