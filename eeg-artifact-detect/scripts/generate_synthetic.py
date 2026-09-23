"""Quantitative validation of eeg_autoclean.detector against synthetic
ground truth (known injected artifacts), across several severity scenarios.

This is the counterpart to scripts/validate_on_sample.py: that script checks
qualitative agreement with MNE's own EOG detector on real data, this one
computes actual precision/recall/F1 against artifacts we know we injected.

Usage:
    python scripts/generate_synthetic.py
"""

import numpy as np

from eeg_autoclean.detector import detect_artifacts
from eeg_autoclean.evaluation import evaluate_layer1, evaluate_layer2, evaluate_layer3, evaluate_layer4
from eeg_autoclean.synthetic import SEVERITY_SCENARIOS, generate_synthetic_raw


def _fmt(metrics):
    return f"P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1']:.2f}"


def main():
    rows = []

    for scenario in SEVERITY_SCENARIOS:
        name = scenario["name"]
        kwargs = {k: v for k, v in scenario.items() if k != "name"}

        raw, ground_truth = generate_synthetic_raw(**kwargs)
        result = detect_artifacts(raw)

        layer1 = evaluate_layer1(raw, result, ground_truth)
        layer2 = evaluate_layer2(raw, result, ground_truth)
        layer3 = evaluate_layer3(raw, result, ground_truth)
        layer4 = evaluate_layer4(raw, result, ground_truth)

        print(f"\n=== Scenario: {name} ===")
        print(f"  Layer 1 amplitude: {_fmt(layer1['BAD_amplitude'])} "
              f"(tp={layer1['BAD_amplitude']['tp']} fp={layer1['BAD_amplitude']['fp']} fn={layer1['BAD_amplitude']['fn']})")
        print(f"  Layer 1 flat:      {_fmt(layer1['BAD_flat'])} "
              f"(tp={layer1['BAD_flat']['tp']} fp={layer1['BAD_flat']['fp']} fn={layer1['BAD_flat']['fn']})")
        print(f"  Layer 2 ocular:    {_fmt(layer2)} "
              f"(flagged={layer2['n_flagged']}, true_positive={layer2['n_true_positive_components']}, "
              f"best_corr={layer2['best_correlation']:.2f})")
        print(f"  Layer 3 muscle:    {_fmt(layer3)} "
              f"(tp={layer3['tp']} fp={layer3['fp']} fn={layer3['fn']})")
        print(f"  Layer 4 cardiac:   {_fmt(layer4)} "
              f"(tp={layer4['tp']} fp={layer4['fp']} fn={layer4['fn']})")
        if result["notes"]:
            print("  Notes:", "; ".join(result["notes"]))

        rows.append({"name": name, "layer1": layer1, "layer2": layer2, "layer3": layer3, "layer4": layer4})

    print("\n=== Summary across scenarios ===")
    for task, label in [("BAD_amplitude", "Layer 1 amplitude"), ("BAD_flat", "Layer 1 flat")]:
        precisions = [r["layer1"][task]["precision"] for r in rows]
        recalls = [r["layer1"][task]["recall"] for r in rows]
        f1s = [r["layer1"][task]["f1"] for r in rows]
        print(
            f"{label:20s} mean P={np.mean(precisions):.2f} (sd={np.std(precisions):.2f})  "
            f"mean R={np.mean(recalls):.2f} (sd={np.std(recalls):.2f})  "
            f"mean F1={np.mean(f1s):.2f} (sd={np.std(f1s):.2f})"
        )

    precisions = [r["layer2"]["precision"] for r in rows]
    recalls = [r["layer2"]["recall"] for r in rows]
    f1s = [r["layer2"]["f1"] for r in rows]
    print(
        f"{'Layer 2 ocular':20s} mean P={np.mean(precisions):.2f} (sd={np.std(precisions):.2f})  "
        f"mean R={np.mean(recalls):.2f} (sd={np.std(recalls):.2f})  "
        f"mean F1={np.mean(f1s):.2f} (sd={np.std(f1s):.2f})"
    )

    precisions = [r["layer3"]["precision"] for r in rows]
    recalls = [r["layer3"]["recall"] for r in rows]
    f1s = [r["layer3"]["f1"] for r in rows]
    print(
        f"{'Layer 3 muscle':20s} mean P={np.mean(precisions):.2f} (sd={np.std(precisions):.2f})  "
        f"mean R={np.mean(recalls):.2f} (sd={np.std(recalls):.2f})  "
        f"mean F1={np.mean(f1s):.2f} (sd={np.std(f1s):.2f})"
    )

    precisions = [r["layer4"]["precision"] for r in rows]
    recalls = [r["layer4"]["recall"] for r in rows]
    f1s = [r["layer4"]["f1"] for r in rows]
    print(
        f"{'Layer 4 cardiac':20s} mean P={np.mean(precisions):.2f} (sd={np.std(precisions):.2f})  "
        f"mean R={np.mean(recalls):.2f} (sd={np.std(recalls):.2f})  "
        f"mean F1={np.mean(f1s):.2f} (sd={np.std(f1s):.2f})"
    )


if __name__ == "__main__":
    main()
