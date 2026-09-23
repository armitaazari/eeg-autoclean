"""Batch processing: run load_eeg + detect_artifacts + compute_quality_score
across every EEG file in a directory, writing a summary CSV.

A single bad/corrupt file must never abort the whole batch -- process_file()
catches any exception from loading or detection and reports it in the row
instead of raising, so process_directory() always finishes and always
produces a CSV covering every file it found, successes and failures alike.
"""

import csv
from pathlib import Path

from .detector import detect_artifacts, load_eeg
from .quality import compute_quality_score

SUPPORTED_EXTENSIONS = (".edf", ".bdf")

CSV_FIELDNAMES = [
    "filename",
    "status",
    "quality_score",
    "n_amplitude_flat_instances",
    "n_muscle_instances",
    "n_cardiac_instances",
    "n_ocular_components_flagged",
    "cardiac_used_ecg_channel",
    "error",
]

_EMPTY_ROW = {name: "" for name in CSV_FIELDNAMES if name != "filename"}


def _count_instances(annotations, description):
    return sum(1 for d in annotations.description if d == description)


def process_file(file_path, detect_kwargs=None, quality_kwargs=None):
    """Run load_eeg + detect_artifacts + compute_quality_score on one file.

    Never raises: any exception during loading, detection, or scoring is
    caught and reported in the returned row's "error" field with
    status="error" instead, so process_directory() can keep going.

    Returns
    -------
    row : dict
        Matches CSV_FIELDNAMES's keys except "filename" (the caller adds
        that, since this function only receives a path it may not be able
        to load at all).
    """
    detect_kwargs = detect_kwargs or {}
    quality_kwargs = quality_kwargs or {}
    try:
        raw = load_eeg(file_path)
        result = detect_artifacts(raw, **detect_kwargs)
        summary = compute_quality_score(raw, result, **quality_kwargs)
        annotations = result["annotations"]
        return {
            "status": "ok",
            "quality_score": round(summary["score"], 2),
            "n_amplitude_flat_instances": (
                _count_instances(annotations, "BAD_amplitude") + _count_instances(annotations, "BAD_flat")
            ),
            "n_muscle_instances": _count_instances(annotations, "BAD_muscle"),
            "n_cardiac_instances": _count_instances(annotations, "BAD_cardiac"),
            "n_ocular_components_flagged": len(result.get("ica_components_flagged") or []),
            "cardiac_used_ecg_channel": result.get("cardiac_used_ecg_channel", False),
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see module docstring
        return {**_EMPTY_ROW, "status": "error", "error": f"{type(exc).__name__}: {exc}"}


def process_directory(dir_path, output_csv=None, detect_kwargs=None, quality_kwargs=None, extensions=SUPPORTED_EXTENSIONS):
    """Run process_file() across every EEG file in dir_path, write a summary CSV.

    Parameters
    ----------
    dir_path : str or Path
        Directory to scan (non-recursive) for files with `extensions`.
    output_csv : str or Path or None
        Where to write the summary CSV. Defaults to "<dir_path>/batch_summary.csv".
    detect_kwargs, quality_kwargs : dict or None
        Extra keyword arguments forwarded to detect_artifacts() /
        compute_quality_score() for every file.
    extensions : tuple of str
        File extensions (case-insensitive) to treat as EEG recordings.

    Returns
    -------
    rows : list of dict
        One entry per file processed (each with a "filename" key), same
        content written to the CSV, in the order files were found.
    output_csv : Path
        Where the CSV was written.
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {dir_path}")

    output_csv = Path(output_csv) if output_csv is not None else dir_path / "batch_summary.csv"

    files = sorted(p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in extensions)

    rows = []
    for file_path in files:
        print(f"Processing {file_path.name}...")
        row = process_file(file_path, detect_kwargs=detect_kwargs, quality_kwargs=quality_kwargs)
        row = {"filename": file_path.name, **row}
        rows.append(row)
        if row["status"] == "error":
            print(f"  FAILED: {row['error']}")
        else:
            print(f"  quality_score={row['quality_score']}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\n{n_ok}/{len(rows)} files processed successfully. Summary written to {output_csv}")

    return rows, output_csv
