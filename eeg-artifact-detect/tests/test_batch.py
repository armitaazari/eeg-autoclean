"""Tests for eeg_autoclean.batch (process_file, process_directory).

load_eeg only reads .edf/.bdf, and this project has no EDF-writing backend
installed, so these tests write real recordings as .fif (MNE's own native
format, content-detected regardless of filename) and monkeypatch
eeg_autoclean.batch.load_eeg to read that format instead. Everything else
-- file discovery by extension, detect_artifacts, compute_quality_score,
per-file error handling, CSV writing -- runs for real, unmocked. This
specifically does NOT re-test EDF parsing itself (already covered by
tests/test_detector.py and tests/test_edge_cases.py); it tests the batch
orchestration logic.
"""

import csv
import os

import mne
import numpy as np
import pytest

import eeg_autoclean.batch as batch


def _write_clean_recording(path, seed=0, n_ch=6, duration_s=30.0, sfreq=250.0):
    # raw.save() strictly enforces a "*raw.fif"-style filename (hard error,
    # unlike read_raw_fif's read-side check, which is only a warning) --
    # save under a conforming temp name, then move to the target .edf path.
    rng = np.random.default_rng(seed)
    data_uv = rng.normal(0.0, 5.0, size=(n_ch, int(duration_s * sfreq)))
    info = mne.create_info(ch_names=[f"EEG{i:03d}" for i in range(n_ch)], sfreq=sfreq, ch_types=["eeg"] * n_ch)
    raw = mne.io.RawArray(data_uv * 1e-6, info, verbose=False)
    tmp_fif = path.with_name(path.stem + "_raw.fif")
    raw.save(tmp_fif, overwrite=True, verbose=False)
    os.replace(tmp_fif, path)


@pytest.fixture
def fif_loader(monkeypatch):
    """Point eeg_autoclean.batch's load_eeg at read_raw_fif for this test."""
    monkeypatch.setattr(batch, "load_eeg", lambda p: mne.io.read_raw_fif(p, preload=True, verbose=False))


def test_process_file_succeeds_on_valid_recording(tmp_path, fif_loader):
    path = tmp_path / "good.edf"
    _write_clean_recording(path)

    row = batch.process_file(path)

    assert row["status"] == "ok"
    assert row["error"] == ""
    assert isinstance(row["quality_score"], float)
    assert 0.0 <= row["quality_score"] <= 100.0


def test_process_file_reports_error_without_raising(tmp_path, fif_loader):
    path = tmp_path / "corrupt.edf"
    path.write_bytes(b"not a real recording")

    row = batch.process_file(path)  # must not raise

    assert row["status"] == "error"
    assert row["error"] != ""
    assert row["quality_score"] == ""


def test_process_directory_handles_mixed_good_and_bad_files(tmp_path, fif_loader):
    _write_clean_recording(tmp_path / "subject1.edf", seed=0)
    _write_clean_recording(tmp_path / "subject2.edf", seed=1)
    (tmp_path / "subject3.edf").write_bytes(b"garbage, not a recording")
    (tmp_path / "notes.txt").write_text("this should be ignored, wrong extension")

    rows, output_csv = batch.process_directory(tmp_path)

    assert output_csv == tmp_path / "batch_summary.csv"
    assert output_csv.exists()
    # notes.txt excluded by extension; exactly the 3 .edf files processed.
    assert len(rows) == 3
    by_name = {r["filename"]: r for r in rows}
    assert by_name["subject1.edf"]["status"] == "ok"
    assert by_name["subject2.edf"]["status"] == "ok"
    assert by_name["subject3.edf"]["status"] == "error"
    assert by_name["subject3.edf"]["error"] != ""

    with open(output_csv, newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 3
    assert set(csv_rows[0].keys()) == set(batch.CSV_FIELDNAMES)


def test_process_directory_custom_output_path(tmp_path, fif_loader):
    _write_clean_recording(tmp_path / "subject1.edf")
    custom_csv = tmp_path / "results" / "my_summary.csv"

    rows, output_csv = batch.process_directory(tmp_path, output_csv=custom_csv)

    assert output_csv == custom_csv
    assert custom_csv.exists()
    assert len(rows) == 1


def test_process_directory_raises_for_non_directory(tmp_path):
    with pytest.raises(ValueError, match="Not a directory"):
        batch.process_directory(tmp_path / "does_not_exist")


def test_process_directory_empty_directory_writes_header_only_csv(tmp_path, fif_loader):
    rows, output_csv = batch.process_directory(tmp_path)

    assert rows == []
    assert output_csv.exists()
    with open(output_csv, newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert csv_rows == []
