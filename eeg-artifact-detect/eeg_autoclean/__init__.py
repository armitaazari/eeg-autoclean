from .batch import process_directory, process_file
from .cleaning import apply_cleaning
from .consensus import run_consensus
from .detector import detect_artifacts, load_eeg
from .eye_state import detect_eye_state
from .quality import compute_quality_score
from .visualize import plot_artifacts

__all__ = [
    "load_eeg",
    "detect_artifacts",
    "plot_artifacts",
    "detect_eye_state",
    "compute_quality_score",
    "process_directory",
    "process_file",
    "apply_cleaning",
    "run_consensus",
]
