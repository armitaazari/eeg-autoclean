"""Compare eeg_autoclean's detector against a bare ICA+EOG baseline.

Does Layer 1 (amplitude/flat scan) and Layer 2's dual-pass ICA/EOG
correlation actually earn their place over just running MNE's own
automated ICA + EOG detection alone -- fit ICA, call
`ica.find_bads_eog(raw)` once on the continuous data, done -- with no Layer 1
scan at all and no separate blink-epoch correlation pass?

Doesn't compare Layer 3 (muscle) against anything -- out of scope here.
detect_artifacts() merges all layers' annotations into one object, so
instance counts below are split out by layer (Layer 1 vs Layer 3) rather
than lumped together, to avoid silently mixing them.

The baseline here is NOT a separately re-fit ICA. It reuses the exact same
fitted ICA and the exact same `ica.find_bads_eog(raw, threshold=...)` call
that our own whole-recording pass already makes internally (see
detector._detect_ocular_ica_components, "Pass 2"). A component ends up in
the baseline's flagged set if and only if that call would have flagged it.
This is deliberate: re-fitting ICA separately would let ICA-fit randomness
or subtly different preprocessing leak into the comparison and blur the two
questions this script actually wants answered:

  (1) Does Layer 1 catch anything meaningful that no ICA+EOG approach could,
      even in principle, because it isn't ocular? (gross amplitude spikes,
      flat/disconnected channels)
  (2) Does the added blink-epoch correlation pass (Layer 2's "Pass 1") ever
      flag a genuinely-ocular component that the whole-recording pass alone
      (Layer 2's "Pass 2" -- the baseline here) would have missed? Or has it
      become redundant now that the whole-recording pass is included?

Run across the same synthetic battery (4 severity scenarios, known ground
truth) and real datasets (sample, erp_core, sleep_physionet) used by
scripts/generate_synthetic.py and scripts/validate_real_datasets.py.

Usage:
    python scripts/compare_baseline.py
"""

import mne
import numpy as np

from eeg_autoclean.detector import detect_artifacts
from eeg_autoclean.evaluation import evaluate_layer1, evaluate_layer2
from eeg_autoclean.synthetic import SEVERITY_SCENARIOS, generate_synthetic_raw
from validate_real_datasets import DATASETS as REAL_DATASETS


def _baseline_result(result):
    """Build the result a bare 'fit ICA, call find_bads_eog(raw) once' baseline
    would have produced, from our tool's already-computed output.

    No Layer 1 annotations at all (the baseline never scans for them), and
    Layer 2 restricted to components the whole-recording pass alone flagged
    -- see module docstring for why this is reused rather than recomputed.
    """
    ica = result["ica"]
    detection_paths = result["ica_detection_paths"] or {}
    baseline_flagged = sorted(idx for idx, paths in detection_paths.items() if "whole_recording" in paths)
    return {
        "annotations": mne.Annotations(onset=[], duration=[], description=[]),
        "ica_components_flagged": baseline_flagged if ica is not None else None,
        "ica": ica,
        "ica_detection_paths": None,
        "notes": [],
    }


def _blink_epoch_only_components(result):
    """Components flagged ONLY by the blink-epoch pass -- i.e. that a bare
    whole-recording find_bads_eog(raw) call (the baseline) would have missed.
    """
    detection_paths = result.get("ica_detection_paths") or {}
    return sorted(idx for idx, paths in detection_paths.items() if paths == ["blink_epochs"])


def _gross_artifact_indicator(raw, ground_truth):
    """Binary waveform: 1 during any injected amplitude/flat window, 0
    elsewhere. Used to concretely check whether ocular-flagged ICA
    components happen to also carry gross-artifact energy, rather than just
    assuming they can't because the two are "different kinds of signal".
    """
    sfreq = raw.info["sfreq"]
    window_samples = int(round(ground_truth["window_duration"] * sfreq))
    indicator = np.zeros(raw.n_times)
    for _ch, w in ground_truth["amplitude_events"] + ground_truth["flat_events"]:
        start, stop = w * window_samples, (w + 1) * window_samples
        indicator[start:stop] = 1.0
    return indicator


def _max_abs_correlation(ica, raw, component_indices, waveform):
    if ica is None or not component_indices:
        return None
    sources = ica.get_sources(raw).get_data(picks=component_indices)
    return max(abs(np.corrcoef(source, waveform)[0, 1]) for source in sources)


def _count_annotations(annotations):
    """Split a combined annotations object into Layer 1 (amplitude/flat) and
    Layer 3 (muscle) instance counts -- detect_artifacts() merges all three
    layers into one mne.Annotations object, so a bare len() silently mixes
    Layer 1 and Layer 3 together.
    """
    layer1 = sum(1 for d in annotations.description if d in ("BAD_amplitude", "BAD_flat"))
    layer3 = sum(1 for d in annotations.description if d == "BAD_muscle")
    return layer1, layer3


def _run_synthetic():
    print("\n" + "=" * 70)
    print("SYNTHETIC BATTERY (known ground truth)")
    print("=" * 70)

    rows = []
    for scenario in SEVERITY_SCENARIOS:
        name = scenario["name"]
        kwargs = {k: v for k, v in scenario.items() if k != "name"}
        raw, ground_truth = generate_synthetic_raw(**kwargs)

        result = detect_artifacts(raw)
        baseline = _baseline_result(result)

        our_layer1 = evaluate_layer1(raw, result, ground_truth)
        our_layer2 = evaluate_layer2(raw, result, ground_truth)
        baseline_layer1 = evaluate_layer1(raw, baseline, ground_truth)
        baseline_layer2 = evaluate_layer2(raw, baseline, ground_truth)

        blink_only = _blink_epoch_only_components(result)
        indicator = _gross_artifact_indicator(raw, ground_truth)
        gross_corr = _max_abs_correlation(result["ica"], raw, result["ica_components_flagged"] or [], indicator)

        n_gross_truth = len(ground_truth["amplitude_events"]) + len(ground_truth["flat_events"])
        our_gross_tp = our_layer1["BAD_amplitude"]["tp"] + our_layer1["BAD_flat"]["tp"]
        baseline_gross_tp = baseline_layer1["BAD_amplitude"]["tp"] + baseline_layer1["BAD_flat"]["tp"]

        print(f"\n--- {name} ---")
        print(f"  Ground truth gross (amplitude+flat) events: {n_gross_truth}")
        print(f"  Baseline (ICA+EOG alone, no Layer 1) caught: {baseline_gross_tp}/{n_gross_truth}")
        print(f"  Our tool (Layer 1) caught:                   {our_gross_tp}/{n_gross_truth}")
        if gross_corr is not None:
            print(
                f"  Max |correlation(flagged ocular component, gross-artifact indicator)| = {gross_corr:.3f} "
                "(near 0 = components aren't accidentally capturing the gross artifacts either)"
            )
        print(
            f"  Layer 2 dual-pass (ours):        P={our_layer2['precision']:.2f} R={our_layer2['recall']:.2f} "
            f"F1={our_layer2['f1']:.2f} (flagged={our_layer2['n_flagged']})"
        )
        print(
            f"  Layer 2 whole_recording-only (baseline): P={baseline_layer2['precision']:.2f} "
            f"R={baseline_layer2['recall']:.2f} F1={baseline_layer2['f1']:.2f} (flagged={baseline_layer2['n_flagged']})"
        )
        if blink_only:
            print(f"  Components flagged ONLY by the blink-epoch pass (missed by whole_recording): {blink_only}")
        else:
            print("  No components were flagged only by the blink-epoch pass in this scenario.")

        our_layer1_instances, our_layer3_instances = _count_annotations(result["annotations"])
        rows.append(
            {
                "name": name,
                "our_layer1_instances": our_layer1_instances,
                "our_layer3_instances": our_layer3_instances,
                "our_layer2_flagged": our_layer2["n_flagged"],
                "baseline_layer2_flagged": baseline_layer2["n_flagged"],
                "blink_only_components": blink_only,
            }
        )
    return rows


def _run_real():
    print("\n" + "=" * 70)
    print("REAL DATASETS (no ground truth -- structural/agreement comparison only)")
    print("=" * 70)

    rows = []
    for name, loader in REAL_DATASETS.items():
        raw = loader()
        result = detect_artifacts(raw)
        baseline = _baseline_result(result)
        blink_only = _blink_epoch_only_components(result)

        layer1_instances, layer3_instances = _count_annotations(result["annotations"])
        our_layer2_flagged = len(result["ica_components_flagged"] or [])
        baseline_layer2_flagged = len(baseline["ica_components_flagged"] or [])

        print(f"\n--- {name} ---")
        print(
            f"  Baseline (ICA+EOG alone, no Layer 1/3) flags 0 amplitude/flat/muscle artifacts by "
            f"construction (it never scans for them); our tool flagged {layer1_instances} Layer 1 "
            f"(amplitude/flat) and {layer3_instances} Layer 3 (muscle) channel-window instances."
        )
        print(f"  Layer 2 dual-pass: {our_layer2_flagged} flagged   Layer 2 whole_recording-only (baseline): {baseline_layer2_flagged} flagged")
        if blink_only:
            print(f"  Components flagged ONLY by the blink-epoch pass: {blink_only}")
        else:
            print("  No components were flagged only by the blink-epoch pass.")

        rows.append(
            {
                "name": name,
                "our_layer1_instances": layer1_instances,
                "our_layer3_instances": layer3_instances,
                "our_layer2_flagged": our_layer2_flagged,
                "baseline_layer2_flagged": baseline_layer2_flagged,
                "blink_only_components": blink_only,
            }
        )
    return rows


def main():
    synthetic_rows = _run_synthetic()
    real_rows = _run_real()
    all_rows = synthetic_rows + real_rows

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_layer1 = sum(r["our_layer1_instances"] for r in all_rows)
    total_layer3 = sum(r["our_layer3_instances"] for r in all_rows)
    total_layer2 = sum(r["our_layer2_flagged"] for r in all_rows)
    total = total_layer1 + total_layer3 + total_layer2
    frac_non_ocular = (total_layer1 + total_layer3) / total if total else 0.0
    print(f"Across all {len(all_rows)} test cases (4 synthetic scenarios + 3 real datasets):")
    print(f"  Total Layer 1 (amplitude/flat) channel-window instances flagged: {total_layer1}")
    print(f"  Total Layer 3 (muscle) channel-window instances flagged:         {total_layer3}")
    print(f"  Total Layer 2 ocular components flagged:                        {total_layer2}")
    print(f"  Non-ocular (Layer 1 + Layer 3) share of everything our tool flags: {frac_non_ocular:.1%}")
    print(
        "  (These are different units -- one channel-window annotation (Layer 1/3) vs one ICA component\n"
        "   (Layer 2) -- but all three are literally what detect_artifacts() emits. A baseline running\n"
        "   ICA+EOG alone produces zero of the Layer 1 or Layer 3 kind, by construction: it has no\n"
        "   mechanism to look for either.)"
    )

    blink_only_cases = [r for r in all_rows if r["blink_only_components"]]
    print(
        f"\nCases where the blink-epoch pass caught something whole_recording alone missed: "
        f"{len(blink_only_cases)}/{len(all_rows)}"
    )
    if not blink_only_cases:
        print(
            "  -> In this test battery, the blink-epoch pass never flagged a component that the\n"
            "     whole-recording pass didn't already flag. Now that whole_recording is part of the\n"
            "     default, the blink-epoch pass looks redundant here -- it isn't earning its keep on\n"
            "     these 7 cases. Worth stating plainly rather than assuming the earlier dual-pass fix\n"
            "     is still adding value just because it once did (on erp_core, before whole_recording\n"
            "     was added)."
        )
    else:
        for r in blink_only_cases:
            print(f"  - {r['name']}: components {r['blink_only_components']}")


if __name__ == "__main__":
    main()
