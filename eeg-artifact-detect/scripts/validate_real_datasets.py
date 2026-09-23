"""Validate eeg_autoclean.detector across several public EEG datasets.

scripts/validate_on_sample.py checks a single dataset (MNE's `sample`); this
script runs the same checks across additional public datasets fetched via
mne.datasets, to see whether the "1/1 agreement with find_bads_eog" result
generalizes or was a lucky case for that one recording. Results are printed
per dataset and then compared side by side -- if a dataset diverges badly,
that's reported explicitly, not hidden.

Datasets used (all include a genuine EOG channel, which many mne.datasets
fetchers -- e.g. eegbci's 64-channel motor imagery set -- do not):
- sample: MNE's own sample recording (Elekta system, 60 EEG + 1 EOG channel).
- erp_core: ERP CORE Flankers task, subject 1 (BioSemi-style montage, 30 EEG
  + 3 EOG channels -- HEOG_left, HEOG_right, VEOG_lower).
- sleep_physionet: Sleep-EDF PSG recording (only 2 EEG channels + 1 EOG
  channel) -- included specifically because it stress-tests the tool on a
  very different, low-channel-count recording type.

Usage:
    python scripts/validate_real_datasets.py
"""

from pathlib import Path

import mne

from eeg_autoclean.detector import detect_artifacts
from eeg_autoclean.reporting import summarize_layer1, summarize_layer2_agreement
from eeg_autoclean.visualize import plot_artifacts

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def _load_sample():
    data_folder = mne.datasets.sample.data_path()
    raw_fname = data_folder / "MEG" / "sample" / "sample_audvis_raw.fif"
    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
    raw.pick(["eeg", "eog"])
    raw.crop(tmax=60.0)
    return raw


def _load_erp_core():
    data_folder = mne.datasets.erp_core.data_path()
    raw_fname = data_folder / "ERP-CORE_Subject-001_Task-Flankers_eeg.fif"
    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
    raw.crop(tmax=60.0)  # first 60s, consistent with the other datasets
    return raw


def _load_sleep_physionet():
    from mne.datasets.sleep_physionet.age import fetch_data

    paths = fetch_data(subjects=[0], recording=[1])
    psg_path, _hypnogram_path = paths[0]
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    # This EDF's channel names aren't auto-recognized by MNE, so everything
    # (EOG, EMG, respiration, temperature, event marker) loads as generic
    # "eeg" type by default. Fix that up explicitly -- otherwise Layer 1
    # would run its amplitude scan on a temperature channel.
    raw.set_channel_types(
        {
            "EOG horizontal": "eog",
            "Resp oro-nasal": "misc",
            "EMG submental": "misc",
            "Temp rectal": "misc",
            "Event marker": "misc",
        }
    )
    # Full-night PSG recording; the first few minutes (subject still awake,
    # settling in) is where eye blinks/movements actually show up.
    raw.crop(tmax=300.0)
    return raw


DATASETS = {
    "sample": _load_sample,
    "erp_core (Flankers, subj. 1)": _load_erp_core,
    "sleep_physionet (night 1)": _load_sleep_physionet,
}


def _fmt_agreement(agreement):
    if agreement is None:
        return "n/a (Layer 2 skipped)"
    if agreement["agreement"] is None:
        return f"our={agreement['our_flagged']} mne={agreement['mne_flagged']} (mne flagged nothing)"
    return (
        f"our={agreement['our_flagged']} mne={agreement['mne_flagged']} "
        f"overlap={agreement['overlap']} ({len(agreement['overlap'])}/{len(agreement['mne_flagged'])})"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for name, loader in DATASETS.items():
        print(f"\n=== Dataset: {name} ===")
        raw = loader()
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        eog_picks = mne.pick_types(raw.info, eog=True, exclude=[])
        print(
            f"{len(eeg_picks)} EEG + {len(eog_picks)} EOG channels, "
            f"sfreq={raw.info['sfreq']:.1f} Hz, {raw.times[-1]:.1f}s analyzed"
        )

        results = detect_artifacts(raw)

        layer1 = summarize_layer1(raw, results)
        print(
            f"Layer 1: {layer1['n_amplitude_windows']}/{layer1['total_windows']} windows amplitude, "
            f"{layer1['n_flat_windows']}/{layer1['total_windows']} windows flat "
            f"({layer1['n_amplitude_instances']} amplitude / {layer1['n_flat_instances']} flat "
            "channel-window instances)"
        )

        # No dataset here ships documented muscle-artifact ground truth (see
        # README's Design notes), so this is a qualitative sanity check only
        # -- does Layer 3 fire at a plausible rate, not a precision/recall
        # claim -- not a validation.
        muscle_instances = sum(1 for d in results["annotations"].description if d == "BAD_muscle")
        muscle_windows = len(
            {o for o, d in zip(results["annotations"].onset, results["annotations"].description) if d == "BAD_muscle"}
        )
        print(
            f"Layer 3 (sanity check, no ground truth): {muscle_windows}/{layer1['total_windows']} windows muscle "
            f"({muscle_instances} channel-window instances)"
        )

        agreement = summarize_layer2_agreement(raw, results)
        if results["ica_components_flagged"] is None:
            print("Layer 2: skipped --", "; ".join(results["notes"]))
        else:
            print(f"Layer 2: {_fmt_agreement(agreement)}")
            # Which pass(es) caught each flagged component -- kept visible
            # rather than merged, since a component caught only by the
            # whole-recording pass (not blink_epochs) is itself a finding.
            for idx, paths in results["ica_detection_paths"].items():
                print(f"  IC{idx} flagged via: {', '.join(paths)}")
            if results["notes"]:
                print("Notes:", "; ".join(results["notes"]))

        slug = "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()
        figure_path = OUTPUT_DIR / f"{slug}_validation.png"
        try:
            plot_artifacts(raw, results, output_path=figure_path)
            print(f"Saved figure: {figure_path}")
        except ValueError as exc:
            # E.g. plot_artifacts needs at least one EEG channel to pick from;
            # don't let a plotting edge case abort the whole comparison.
            print(f"Plotting skipped: {exc}")

        rows.append({"name": name, "layer1": layer1, "agreement": agreement, "notes": results["notes"]})

    print("\n=== Cross-dataset comparison ===")
    for row in rows:
        agreement = row["agreement"]
        if agreement is None:
            agreement_str = "n/a"
        elif agreement["agreement"] is None:
            agreement_str = "mne flagged nothing to compare against"
        else:
            agreement_str = f"{len(agreement['overlap'])}/{len(agreement['mne_flagged'])}"
        print(
            f"{row['name']:30s} amplitude={row['layer1']['n_amplitude_windows']}/{row['layer1']['total_windows']}  "
            f"flat={row['layer1']['n_flat_windows']}/{row['layer1']['total_windows']}  "
            f"L2 agreement={agreement_str}"
        )

    agreements = [r["agreement"]["agreement"] for r in rows if r["agreement"] and r["agreement"]["agreement"] is not None]
    if agreements and (max(agreements) - min(agreements) > 0.01 or any(a < 1.0 for a in agreements)):
        print(
            "\nNote: Layer 2 agreement with MNE's find_bads_eog is NOT consistent across "
            "datasets -- see per-dataset notes above for likely reasons (e.g. channel count)."
        )


if __name__ == "__main__":
    main()
