# utils/chunk_saver_min.py
import os, time, numpy as np

class ChunkSaverMin:
    def __init__(self, root="dbg_chunks"):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self._ctr = 0

    def save(self, actions_raw: np.ndarray):
        """Save RAW 50x8 chunk."""
        self._ctr += 1
        ts = time.strftime("%Y%m%d-%H%M%S")
        base = f"{ts}_c{self._ctr:03d}"
        npz_path = os.path.join(self.root, base + ".npz")
        csv_path = os.path.join(self.root, base + ".csv")

        # exact values
        np.savez_compressed(npz_path, actions=actions_raw)

        # quick-look CSV
        with open(csv_path, "w") as f:
            H, D = actions_raw.shape
            f.write("t," + ",".join(f"d{i}" for i in range(D)) + "\n")
            for t in range(H):
                f.write(f"{t}," + ",".join(f"{v:.6f}" for v in actions_raw[t]) + "\n")

        # pointer for your visualizer
        with open(os.path.join(self.root, "chunk_latest_path.txt"), "w") as f:
            f.write(csv_path)

        return npz_path, csv_path
