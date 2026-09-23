"""Validate eeg_autoclean.consensus.run_consensus across the synthetic
battery and a real dataset: reports the AGREE_BAD / AGREE_CLEAN / DISAGREE
fraction per case, and spot-checks a handful of DISAGREE cells against
ground truth (synthetic) or by direct inspection (real) to see whether
disagreements look like genuinely ambiguous/borderline cases or just one
method's known weak points.

The synthetic battery's channels (generic "EEG000".."EEGNNN" names, no
montage) are given real 10-20 names and a standard montage here --
AutoReject needs real channel positions for its spatial reasoning (see
consensus.py), which synthetic data has no natural equivalent of. This is
a validation-harness concern only; eeg_autoclean.synthetic itself is
unchanged.

Usage:
    python scripts/validate_consensus.py
"""

import mne

from eeg_autoclean.consensus import run_consensus
from eeg_autoclean.synthetic import SEVERITY_SCENARIOS, generate_synthetic_raw


def _attach_standard_montage(raw):
    """Rename raw's EEG channels to real 10-20 names and attach a standard
    montage, returning a {old_name: new_name} map so ground truth (which
    refers to the old names) can be translated too.
    """
    montage = mne.channels.make_standard_montage("standard_1020")
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(eeg_picks) > len(montage.ch_names):
        raise ValueError("More EEG channels than the standard_1020 montage has positions for.")
    rename_map = {raw.ch_names[p]: montage.ch_names[i] for i, p in enumerate(eeg_picks)}
    raw.rename_channels(rename_map)
    raw.set_montage(montage)
    return rename_map


def _translate_ground_truth(ground_truth, rename_map):
    translated = dict(ground_truth)
    for key in ("amplitude_events", "flat_events", "muscle_events", "cardiac_events"):
        if key in translated:
            translated[key] = [(rename_map.get(ch, ch), w) for ch, w in translated[key]]
    if "frontal_channels" in translated:
        translated["frontal_channels"] = [rename_map.get(ch, ch) for ch in translated["frontal_channels"]]
    return translated


def _ground_truth_bad_cells(ground_truth):
    """All (channel, 1.0s-window) cells any of our own ground-truth
    categories consider bad, for spot-checking DISAGREE cells against.
    """
    cells = set()
    for key in ("amplitude_events", "flat_events", "muscle_events"):
        cells.update(ground_truth.get(key, []))
    # cardiac_events are on the cardiac_window_duration grid (multi-window);
    # expand each to its underlying 1.0s windows.
    cardiac_window_duration = ground_truth.get("cardiac_window_duration")
    if cardiac_window_duration:
        sub_windows = int(round(cardiac_window_duration / ground_truth["window_duration"]))
        for ch, w_c in ground_truth.get("cardiac_events", []):
            for sub_w in range(w_c * sub_windows, (w_c + 1) * sub_windows):
                cells.add((ch, sub_w))
    return cells


def _run_synthetic():
    print("\n" + "=" * 70)
    print("SYNTHETIC BATTERY")
    print("=" * 70)

    rows = []
    for scenario in SEVERITY_SCENARIOS:
        name = scenario["name"]
        kwargs = {k: v for k, v in scenario.items() if k != "name"}
        raw, ground_truth = generate_synthetic_raw(**kwargs)
        rename_map = _attach_standard_montage(raw)
        ground_truth = _translate_ground_truth(ground_truth, rename_map)
        truth_cells = _ground_truth_bad_cells(ground_truth)

        result = run_consensus(raw)
        counts, fractions = result["counts"], result["fractions"]

        print(f"\n--- {name} ---")
        print(
            f"  AGREE_BAD={counts['AGREE_BAD']} ({fractions['AGREE_BAD']:.1%})  "
            f"AGREE_CLEAN={counts['AGREE_CLEAN']} ({fractions['AGREE_CLEAN']:.1%})  "
            f"DISAGREE={counts['DISAGREE']} ({fractions['DISAGREE']:.1%})  "
            f"(total_cells={counts['total_cells']})"
        )

        # Spot-check: for each DISAGREE cell, is it actually one of our
        # known-injected ground-truth artifacts? "ours"-flagged disagreements
        # should mostly be real (that's what Layer 1/3/4 were validated
        # against); "autoreject"-flagged disagreements have no such
        # ground-truth backing to check against directly, so instead we
        # report how many land on a real injected artifact BY COINCIDENCE
        # (would suggest AutoReject catching something ours structurally
        # can't, e.g. spatial covariance) vs how many land on plain
        # background noise (would suggest AutoReject's own false positives,
        # not genuine ambiguity).
        ours_flagged = [d for d in result["disagreements"] if d["flagged_by"] == "ours"]
        ar_flagged = [d for d in result["disagreements"] if d["flagged_by"] == "autoreject"]
        ours_on_truth = sum(1 for d in ours_flagged if (d["channel"], d["window"]) in truth_cells)
        ar_on_truth = sum(1 for d in ar_flagged if (d["channel"], d["window"]) in truth_cells)
        print(
            f"  Disagreements flagged by ours only: {len(ours_flagged)} "
            f"({ours_on_truth} land on a real injected artifact, {len(ours_flagged) - ours_on_truth} don't)"
        )
        print(
            f"  Disagreements flagged by autoreject only: {len(ar_flagged)} "
            f"({ar_on_truth} land on a real injected artifact, {len(ar_flagged) - ar_on_truth} don't)"
        )

        rows.append({"name": name, "counts": counts, "fractions": fractions})
    return rows


def _load_sample():
    data_folder = mne.datasets.sample.data_path()
    raw_fname = data_folder / "MEG" / "sample" / "sample_audvis_raw.fif"
    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
    raw.pick(["eeg", "eog"])
    raw.crop(tmax=60.0)
    return raw


def _run_real():
    print("\n" + "=" * 70)
    print("REAL DATASET: sample (no ground truth -- qualitative spot-check only)")
    print("=" * 70)

    raw = _load_sample()
    result = run_consensus(raw)
    counts, fractions = result["counts"], result["fractions"]
    print(
        f"  AGREE_BAD={counts['AGREE_BAD']} ({fractions['AGREE_BAD']:.1%})  "
        f"AGREE_CLEAN={counts['AGREE_CLEAN']} ({fractions['AGREE_CLEAN']:.1%})  "
        f"DISAGREE={counts['DISAGREE']} ({fractions['DISAGREE']:.1%})  "
        f"(total_cells={counts['total_cells']})"
    )

    print("\n  First 10 DISAGREE cells (for manual/visual spot-checking):")
    for d in result["disagreements"][:10]:
        print(f"    {d['channel']:>10s}  window={d['window']:>3d}  t={d['onset_s']:.1f}s  flagged_by={d['flagged_by']}")

    return result


if __name__ == "__main__":
    _run_synthetic()
    _run_real()
