from pathlib import Path

from slam.analysis.tracking_stats import compute_tracking_statistics
from slam.pipeline.database_pipeline import build_tracking_database
from slam.tracking_database import TrackingDB

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent

SEQUENCE_DIR = PROJECT_DIR / "dataset" / "sequences" / "00"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "ex4"
DB_PATH = OUTPUT_DIR / "tracking_db"

NUM_FRAMES = None
REBUILD_DB = True


def main():
    if REBUILD_DB:
        db, inlier_percentages = build_tracking_database(
            sequence_dir=SEQUENCE_DIR,
            num_frames=NUM_FRAMES,
            feature_type="orb",
            num_features=3000,
            ratio_threshold=0.75,
            deviation_threshold=2.0,
        )

        db._check_consistency()
        db.serialize(str(DB_PATH))

    else:
        db = TrackingDB()
        db.load(str(DB_PATH))

    # Question 4.2
    stats = compute_tracking_statistics(db)
    print(stats)

    # Question 4.3



if __name__ == "__main__":
    main()