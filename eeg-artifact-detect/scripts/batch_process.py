"""CLI entry point for eeg_autoclean.batch.process_directory: run the full
detect_artifacts + compute_quality_score pipeline across every .edf/.bdf
file in a directory, writing a summary CSV.

Usage:
    python scripts/batch_process.py path/to/recordings
    python scripts/batch_process.py path/to/recordings --output summary.csv
"""

import argparse
import sys
from pathlib import Path

from eeg_autoclean.batch import process_directory


def main():
    parser = argparse.ArgumentParser(description="Batch-run eeg_autoclean over a directory of EEG recordings.")
    parser.add_argument("directory", type=Path, help="Directory containing .edf/.bdf files.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write the summary CSV (default: <directory>/batch_summary.csv).",
    )
    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"Error: {args.directory} is not a directory.", file=sys.stderr)
        sys.exit(1)

    _rows, output_csv = process_directory(args.directory, output_csv=args.output)
    print(f"Done. See {output_csv}")


if __name__ == "__main__":
    main()
