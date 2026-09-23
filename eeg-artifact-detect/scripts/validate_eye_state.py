"""Validate eeg_autoclean.eye_state.detect_eye_state against real labeled data.

MNE's eegbci (PhysioNet Motor Movement/Imagery) dataset includes two
baseline resting-state runs per subject: run 1 is eyes-open, run 2 is
eyes-closed (https://physionet.org/content/eegmmidb/1.0.0/) -- real ground
truth, not synthetic. This script runs detect_eye_state on both runs for a
handful of subjects and reports classification accuracy against that known
per-run label.

Usage:
    python scripts/validate_eye_state.py
"""

import mne

from eeg_autoclean.eye_state import detect_eye_state

SUBJECTS = [1, 2, 3, 4, 5]
RUNS = {1: "eyes_open", 2: "eyes_closed"}


def _load_run(subject, run):
    paths = mne.datasets.eegbci.load_data(subjects=subject, runs=[run], verbose=False)
    raw = mne.io.read_raw_edf(paths[0], preload=True, verbose=False)
    # eegbci ships channel names padded with dots (e.g. "O1.."); standardize()
    # strips them to plain 10-20 names. detect_eye_state's own channel
    # matching already tolerates the dots, so this is for readability of the
    # printed output, not a functional requirement.
    mne.datasets.eegbci.standardize(raw)
    return raw


def main():
    rows = []
    for subject in SUBJECTS:
        for run, true_state in RUNS.items():
            raw = _load_run(subject, run)
            result = detect_eye_state(raw)
            correct = result["state"] == true_state
            rows.append(
                {
                    "subject": subject,
                    "run": run,
                    "true_state": true_state,
                    "predicted_state": result["state"],
                    "bump_ratio": result["bump_ratio"],
                    "correct": correct,
                }
            )
            ratio_str = f"{result['bump_ratio']:.2f}" if result["bump_ratio"] is not None else "n/a"
            predicted_str = result["state"] if result["state"] is not None else "none"
            print(
                f"Subject {subject:>3} run {run} (true={true_state:11s}): "
                f"predicted={predicted_str:11s} bump_ratio={ratio_str:>8}  {'OK' if correct else 'WRONG'}"
            )
            if result["notes"]:
                print("  Notes:", "; ".join(result["notes"]))

    n_correct = sum(r["correct"] for r in rows)
    n_total = len(rows)
    print(f"\nOverall accuracy: {n_correct}/{n_total} = {n_correct / n_total:.1%}")

    # Break down by true label too, since a single threshold could be biased
    # toward one class without this showing up in the overall number.
    for label in ("eyes_open", "eyes_closed"):
        subset = [r for r in rows if r["true_state"] == label]
        n_c = sum(r["correct"] for r in subset)
        print(f"  {label:11s}: {n_c}/{len(subset)} = {n_c / len(subset):.1%}")


if __name__ == "__main__":
    main()
