"""Validate eeg_autoclean.detector against MNE's built-in sample dataset.

Loads the sample recording (downloaded automatically on first run), runs
detect_artifacts() plus eye-state detection on it, and compares our
ICA-flagged ocular components against MNE's own find_bads_eog result on
the real EOG channel as a rough accuracy/sanity check. Prints a summary
for Layers 1-3 and eye state; Layer 4 (cardiac) isn't broken out here,
though its flags are still included in the annotations passed to
plot_artifacts. This is also what generates docs/images/sample_validation.png,
the figure used in the README.

See scripts/validate_real_datasets.py for the same checks run across
multiple public datasets, to see whether results generalize.

Usage:
    python scripts/validate_on_sample.py
"""

from pathlib import Path

import mne

from eeg_autoclean.detector import detect_artifacts
from eeg_autoclean.eye_state import detect_eye_state
from eeg_autoclean.reporting import summarize_layer1, summarize_layer2_agreement
from eeg_autoclean.visualize import plot_artifacts

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def main():
    print("Downloading/locating MNE sample dataset (this may take a while on first run)...")
    sample_data_folder = mne.datasets.sample.data_path()
    raw_fname = sample_data_folder / "MEG" / "sample" / "sample_audvis_raw.fif"

    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
    raw.pick(["eeg", "eog"])  # keep this demo focused and fast
    raw.crop(tmax=60.0)  # first 60s is plenty to see a few blinks

    print(f"Loaded raw: {len(raw.ch_names)} channels, {raw.times[-1]:.1f}s duration")

    # Use the default adaptive amplitude threshold (per-channel, scaled to
    # that channel's own noise floor) rather than pinning a fixed microvolt
    # value -- that's the shipped default behavior we want this demo to show.
    results = detect_artifacts(raw)

    layer1 = summarize_layer1(raw, results)
    print("\n--- Layer 1: amplitude/flat window detection ---")
    print(f"Unique windows with an amplitude flag: {layer1['n_amplitude_windows']} / {layer1['total_windows']}")
    print(f"Unique windows with a flat flag: {layer1['n_flat_windows']} / {layer1['total_windows']}")
    print(
        f"(Per-channel-window instances: {layer1['n_amplitude_instances']} amplitude, "
        f"{layer1['n_flat_instances']} flat)"
    )

    print("\n--- Layer 2: ICA ocular component detection ---")
    if results["ica_components_flagged"] is None:
        print("Skipped:", "; ".join(results["notes"]))
    else:
        agreement = summarize_layer2_agreement(raw, results)
        print(f"Our flagged ICA components: {agreement['our_flagged']}")
        if results["notes"]:
            print("Notes:", "; ".join(results["notes"]))
        print(f"mne.find_bads_eog components (reference): {agreement['mne_flagged']}")
        print(f"Overlap: {agreement['overlap']}")
        if agreement["agreement"] is not None:
            print(
                f"Agreement: {len(agreement['overlap'])}/{len(agreement['mne_flagged'])} "
                "reference components matched"
            )

    # No ground truth on this real recording, so this is the same kind of
    # qualitative sanity check as validate_real_datasets.py's Layer 3
    # section, not a precision/recall claim (see README/muscle.py for the
    # actual synthetic-battery validation numbers).
    muscle_instances = sum(1 for d in results["annotations"].description if d == "BAD_muscle")
    muscle_windows = len(
        {o for o, d in zip(results["annotations"].onset, results["annotations"].description) if d == "BAD_muscle"}
    )
    print("\n--- Layer 3: muscle window detection (sanity check, no ground truth) ---")
    print(f"Windows with a muscle flag: {muscle_windows} / {layer1['total_windows']}")
    print(f"(Channel-window instances: {muscle_instances})")

    eye_state_result = detect_eye_state(raw)
    print("\n--- Eye-state detection (occipital alpha bump ratio) ---")
    if eye_state_result["state"] is None:
        print("Skipped:", "; ".join(eye_state_result["notes"]))
    else:
        print(f"Classified: {eye_state_result['state']} (bump ratio {eye_state_result['bump_ratio']:.2f})")
        print(f"Channels used: {eye_state_result['channels_used']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = OUTPUT_DIR / "sample_validation.png"
    # start/duration chosen (rather than the plot defaults) to land on a
    # segment where both an amplitude flag and a muscle flag are visible
    # together, so the published figure actually shows all three layers
    # doing something, not just Layer 1.
    plot_artifacts(
        raw, results, output_path=figure_path, start=38.0, duration=20.0, eye_state_result=eye_state_result
    )
    print(f"\nSaved figure: {figure_path}")


if __name__ == "__main__":
    main()
