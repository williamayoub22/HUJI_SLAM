import matplotlib.pyplot as plt
from src.slam.pipeline.caching import load_or_build_db
from src.slam.config import FINAL_ANALYSIS_OUTPUT_DIR
from tqdm import tqdm

def main():
    FINAL_ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    db = load_or_build_db()
    manager_2d = db.manager_2d
    frame_ids = sorted(manager_2d.all_frames())
    
    frame_connectivities = []
    for current_frame, next_frame in tqdm(list(zip(frame_ids[:-1], frame_ids[1:])), desc="Connectivity"):
        current_tracks = set(manager_2d.tracks(current_frame))
        next_tracks = set(manager_2d.tracks(next_frame))
        frame_connectivities.append(len(current_tracks.intersection(next_tracks)))
        
    plt.figure(figsize=(10, 4))
    plt.plot(frame_ids[:-1], frame_connectivities, linewidth=0.5)
    mean_val = sum(frame_connectivities)/len(frame_connectivities) if frame_connectivities else 0
    plt.axhline(mean_val, linestyle="--", color="C1", label=f"Mean: {mean_val:.1f}")
    plt.xlabel("Frame ID")
    plt.ylabel("Outgoing Tracks")
    plt.title("Connectivity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FINAL_ANALYSIS_OUTPUT_DIR / "3_connectivity.png", dpi=150)

if __name__ == "__main__": main()
