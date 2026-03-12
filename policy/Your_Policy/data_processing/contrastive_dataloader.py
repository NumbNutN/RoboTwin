"""
Contrastive DataLoader for pos/neg sample pairs.

Sampling strategy:
  - Positive pairs: pos samples from DIFFERENT episodes (both demonstrate
    correct behavior for the same phase, so they are semantically similar)
  - Negative pairs: pos sample + neg sample from the SAME episode
    (one succeeds, one fails — contrastive signal)

Each batch item is a dict:
  {
    "anchor":   Tensor,   # from a pos branch of episode A
    "positive": Tensor,   # from a pos branch of episode B (B != A)
    "negative": Tensor,   # from a neg branch of episode A
    "phase":    str,      # e.g. "grasp"
    "meta": {...}
  }

Usage:
    dataset = ContrastiveDataset(
        data_root="./data/place_bread_basket/demo_clean",
        phase="grasp",
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    for batch in loader:
        anchor = batch["anchor"]     # (B, T, C, H, W)
        positive = batch["positive"]
        negative = batch["negative"]
"""

import os
import json
import random
import numpy as np
import h5py
from typing import Optional
from torch.utils.data import Dataset, DataLoader


class ContrastiveDataset(Dataset):
    """
    Dataset that yields (anchor, positive, negative) triplets for
    contrastive learning across episodes.

    Args:
        data_root: path containing .cache/episodeN/ directories
        phase: which phase to load samples for (e.g. "grasp", "lift", "place").
               If None, loads all phases.
        obs_key: observation key path in HDF5, e.g. "observation/head_camera/rgb"
        max_seq_len: max number of frames to load per sample (truncate/pad)
        transform: optional callable applied to each frame array
    """

    def __init__(
        self,
        data_root: str,
        phase: Optional[str] = "grasp",
        obs_key: str = "observation/head_camera/rgb",
        max_seq_len: int = 200,
        transform=None,
    ):
        self.data_root = data_root
        self.phase = phase
        self.obs_key = obs_key
        self.max_seq_len = max_seq_len
        self.transform = transform

        # Discover episodes
        self.episodes = self._discover_episodes()
        if len(self.episodes) < 2:
            raise ValueError(
                f"Need at least 2 episodes for contrastive pairs, "
                f"found {len(self.episodes)} in {data_root}")

        # Build index: list of (ep_idx, pos_key) for anchoring
        self.index = []
        for ep_idx, ep in enumerate(self.episodes):
            for pos_key in ep["pos_keys"]:
                if ep["neg_keys"]:  # need at least one neg
                    self.index.append((ep_idx, pos_key))

        if not self.index:
            raise ValueError("No valid anchor entries found "
                             "(each episode needs both pos and neg samples)")

    def _discover_episodes(self):
        """Scan data_root for episode metadata."""
        episodes = []
        cache_root = os.path.join(self.data_root, ".cache")
        if not os.path.isdir(cache_root):
            raise FileNotFoundError(f"Cache root not found: {cache_root}")

        ep_dirs = sorted([
            d for d in os.listdir(cache_root)
            if d.startswith("episode") and os.path.isdir(
                os.path.join(cache_root, d))
        ], key=lambda x: int(x.replace("episode", "")))

        for ep_dir in ep_dirs:
            ep_path = os.path.join(cache_root, ep_dir)
            meta_path = os.path.join(ep_path, "metadata.json")
            h5_path = os.path.join(ep_path, "merged_data.h5")

            if not os.path.exists(h5_path):
                continue

            # Load or infer metadata
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
            else:
                meta = self._infer_metadata(h5_path)

            # Filter by phase
            pos_keys = [
                k for k, v in meta.get("positive_branches", {}).items()
                if self.phase is None or v.get("phase") == self.phase
            ]
            neg_keys = [
                k for k, v in meta.get("negative_branches", {}).items()
                if self.phase is None or v.get("phase") == self.phase
            ]

            if pos_keys:
                episodes.append({
                    "ep_dir": ep_dir,
                    "h5_path": h5_path,
                    "meta": meta,
                    "pos_keys": pos_keys,
                    "neg_keys": neg_keys,
                })

        return episodes

    @staticmethod
    def _infer_metadata(h5_path):
        """Infer metadata from HDF5 structure when JSON is missing."""
        meta = {"positive_branches": {}, "negative_branches": {}}
        try:
            with h5py.File(h5_path, "r") as f:
                if "positive_trajs" in f:
                    for k in f["positive_trajs"]:
                        phase = f["positive_trajs"][k].attrs.get("phase", "")
                        meta["positive_branches"][k] = {
                            "phase": phase, "branch_idx": k}
                if "negative_trajs" in f:
                    for k in f["negative_trajs"]:
                        phase = f["negative_trajs"][k].attrs.get("phase", "")
                        meta["negative_branches"][k] = {
                            "phase": phase, "branch_idx": k}
        except Exception:
            pass
        return meta

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep_idx, pos_key = self.index[idx]
        ep = self.episodes[ep_idx]

        # Anchor: pos sample from this episode
        anchor = self._load_branch(ep["h5_path"], "positive_trajs", pos_key)

        # Positive: pos sample from a DIFFERENT episode
        other_eps = [i for i in range(len(self.episodes)) if i != ep_idx]
        other_idx = random.choice(other_eps)
        other_ep = self.episodes[other_idx]
        other_pos_key = random.choice(other_ep["pos_keys"])
        positive = self._load_branch(
            other_ep["h5_path"], "positive_trajs", other_pos_key)

        # Negative: neg sample from the SAME episode
        neg_key = random.choice(ep["neg_keys"])
        negative = self._load_branch(ep["h5_path"], "negative_trajs", neg_key)

        # Extract phase from metadata
        phase = ep["meta"]["positive_branches"].get(
            pos_key, {}).get("phase", self.phase or "")

        return {
            "anchor": anchor,
            "positive": positive,
            "negative": negative,
            "phase": phase,
            "meta": {
                "anchor_ep": ep["ep_dir"],
                "positive_ep": other_ep["ep_dir"],
                "anchor_key": pos_key,
                "positive_key": other_pos_key,
                "negative_key": neg_key,
            },
        }

    def _load_branch(self, h5_path, group_name, branch_key):
        """
        Load observation sequence from an HDF5 branch.

        Returns:
            np.ndarray of shape (T, C, H, W) or (T, ...) depending on obs_key
        """
        with h5py.File(h5_path, "r") as f:
            path = f"{group_name}/{branch_key}/{self.obs_key}"
            if path not in f:
                # Fallback: try flattened path
                parts = self.obs_key.split("/")
                node = f[f"{group_name}/{branch_key}"]
                for p in parts:
                    if p in node:
                        node = node[p]
                    else:
                        return np.zeros((self.max_seq_len, 3, 64, 64),
                                        dtype=np.float32)
                data = node[:]
            else:
                data = f[path][:]

        # data shape: (T, H, W, C) typically
        if data.ndim == 4 and data.shape[-1] in (3, 4):
            # HWC -> CHW
            data = np.transpose(data, (0, 3, 1, 2))

        # Normalize to [0, 1] if uint8
        if data.dtype == np.uint8:
            data = data.astype(np.float32) / 255.0

        # Truncate or pad to max_seq_len
        T = data.shape[0]
        if T > self.max_seq_len:
            data = data[:self.max_seq_len]
        elif T < self.max_seq_len:
            pad_shape = (self.max_seq_len - T,) + data.shape[1:]
            data = np.concatenate(
                [data, np.zeros(pad_shape, dtype=data.dtype)])

        if self.transform:
            data = self.transform(data)

        return data


def build_contrastive_loader(
    data_root: str,
    phase: str = "grasp",
    batch_size: int = 8,
    max_seq_len: int = 200,
    num_workers: int = 2,
    **kwargs,
) -> DataLoader:
    """Convenience function to create a ContrastiveDataset + DataLoader."""
    dataset = ContrastiveDataset(
        data_root=data_root,
        phase=phase,
        max_seq_len=max_seq_len,
        **kwargs,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test contrastive dataloader")
    parser.add_argument("--data-root",
                        default="./data/place_bread_basket/demo_clean",
                        help="Path to task data root")
    parser.add_argument("--phase", default="grasp",
                        help="Phase to load (grasp/lift/place)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=100)
    args = parser.parse_args()

    print(f"Loading data from: {args.data_root}")
    print(f"Phase: {args.phase}")

    dataset = ContrastiveDataset(
        data_root=args.data_root,
        phase=args.phase,
        max_seq_len=args.max_seq_len,
    )
    print(f"Dataset size: {len(dataset)} triplets "
          f"from {len(dataset.episodes)} episodes")

    for i, ep in enumerate(dataset.episodes):
        print(f"  Episode {ep['ep_dir']}: "
              f"pos={ep['pos_keys']}, neg={ep['neg_keys']}")

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, drop_last=True)

    for batch_idx, batch in enumerate(loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  anchor:   {batch['anchor'].shape}")
        print(f"  positive: {batch['positive'].shape}")
        print(f"  negative: {batch['negative'].shape}")
        print(f"  phases:   {batch['phase']}")
        for k in ("anchor_ep", "positive_ep"):
            print(f"  {k}: {batch['meta'][k]}")
        if batch_idx >= 2:
            break

    print("\nDataloader test complete.")
