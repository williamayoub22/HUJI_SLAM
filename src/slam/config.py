"""Project-wide paths and dataset configuration."""

from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = CODE_DIR.parent

DATASET_DIR = PROJECT_DIR / "dataset"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

SEQUENCE_ID = "00"
SEQUENCE_DIR = DATASET_DIR / "sequences" / SEQUENCE_ID

CALIBRATION_PATH = SEQUENCE_DIR / "calib.txt"

EX3_OUTPUT_DIR = OUTPUTS_DIR / "ex3"
EX4_OUTPUT_DIR = OUTPUTS_DIR / "ex4"
EX5_OUTPUT_DIR = OUTPUTS_DIR / "ex5"
EX6_OUTPUT_DIR = OUTPUTS_DIR / "ex6"
EX7_OUTPUT_DIR = OUTPUTS_DIR / "ex7"
EX8_OUTPUT_DIR = OUTPUTS_DIR / "ex8"
FINAL_ANALYSIS_OUTPUT_DIR = OUTPUTS_DIR / "final_analysis"

CACHE_DIR = OUTPUTS_DIR / "cache"
FINAL_ANALYSIS_OUTPUT_DIR = OUTPUTS_DIR / "final_analysis"

DB_PATH = EX4_OUTPUT_DIR / "tracking_db"
GLOBAL_CAMERA_MATRICES_PATH = EX3_OUTPUT_DIR / "global_camera_matrices.npy"
GT_POSES_PATH = DATASET_DIR / "poses" / f"{SEQUENCE_ID}.txt"

LEFT_IMAGES_DIR = SEQUENCE_DIR / "image_0"
RIGHT_IMAGES_DIR = SEQUENCE_DIR / "image_1"
