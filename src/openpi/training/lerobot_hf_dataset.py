"""Compatibility loader for LeRobot datasets via HuggingFace parquet (no LeRobotDataset class).

Why this exists:
- Some LeRobot versions crash when initializing `LeRobotDataset` for older v2.0 datasets
  (e.g. `physical-intelligence/libero`) due to `torch.stack(hf_dataset["timestamp"])` where
  HF returns a Column.
- OpenPI's training pipeline only needs `__len__`/`__getitem__` plus LeRobot-style delta
  action querying (action chunks).

This module provides `HFParquetLeRobotDataset`, which:
- loads the parquet split via `datasets.load_dataset`
- optionally expands keys like `actions` into a horizon-length sequence via `delta_timestamps`
- adds `task` string based on `task_index` + `meta.tasks` (matching LeRobotDataset)

Limitations:
- Video-backed datasets are not supported in this fallback (meta.video_keys must be empty).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import torch


@dataclass(frozen=True)
class Meta:
    fps: int
    tasks: dict[int, str]
    episodes: dict[int, dict[str, Any]]
    video_keys: tuple[str, ...] = ()


def _get_episode_data_index(episodes: dict[int, dict[str, Any]]) -> dict[str, torch.Tensor]:
    # Mirror lerobot.common.datasets.utils.get_episode_data_index but keep it local for tests.
    lengths = [episodes[i]["length"] for i in range(len(episodes))]
    cum = np.cumsum(lengths).tolist()
    return {
        "from": torch.LongTensor([0] + cum[:-1]),
        "to": torch.LongTensor(cum),
    }


def _get_delta_indices(delta_timestamps: dict[str, list[float]], fps: int) -> dict[str, list[int]]:
    # Mirror lerobot.common.datasets.utils.get_delta_indices.
    return {k: [int(round(ts * fps)) for ts in v] for k, v in delta_timestamps.items()}


class HFParquetLeRobotDataset:
    """LeRobot-like dataset backed by HF parquet, with delta-index querying."""

    def __init__(
        self,
        *,
        repo_id: str,
        root: Path,
        delta_timestamps: dict[str, list[float]] | None,
        meta: Meta,
        episodes: list[int] | None = None,
    ):
        self.repo_id = repo_id
        self.root = root
        self.meta = meta
        self.episodes = episodes
        self.episode_data_index = _get_episode_data_index(meta.episodes)
        self.delta_indices = None if delta_timestamps is None else _get_delta_indices(delta_timestamps, meta.fps)

        if meta.video_keys:
            raise NotImplementedError("HFParquetLeRobotDataset does not support video-backed datasets.")

        data_dir = root / "data"
        if not data_dir.exists():
            raise FileNotFoundError(f"Expected LeRobot parquet data under {data_dir}")

        if episodes is None:
            self.hf_dataset = datasets.load_dataset("parquet", data_dir=str(data_dir), split="train")
        else:
            # Conservative: load all and filter by episode_index.
            ds = datasets.load_dataset("parquet", data_dir=str(data_dir), split="train")
            self.hf_dataset = ds.filter(lambda ex: int(ex["episode_index"]) in set(episodes))

        # Match LeRobotDataset behavior: transform HF rows into torch tensors.
        from lerobot.common.datasets.utils import hf_transform_to_torch

        self.hf_dataset.set_transform(hf_transform_to_torch)

    @classmethod
    def from_hf_dataset(
        cls,
        *,
        repo_id: str,
        hf_dataset: datasets.Dataset,
        delta_timestamps: dict[str, list[float]] | None,
        fps: int,
        tasks: dict[int, str],
        episodes: dict[int, dict[str, Any]],
    ) -> "HFParquetLeRobotDataset":
        """Test helper: build without touching disk."""

        obj = object.__new__(cls)
        obj.repo_id = repo_id
        obj.root = Path(".")
        obj.meta = Meta(fps=fps, tasks=tasks, episodes=episodes, video_keys=())
        obj.episodes = None
        obj.episode_data_index = _get_episode_data_index(episodes)
        obj.delta_indices = None if delta_timestamps is None else _get_delta_indices(delta_timestamps, fps)
        obj.hf_dataset = hf_dataset
        return obj

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = int(item["episode_index"].item() if hasattr(item["episode_index"], "item") else item["episode_index"])

        if self.delta_indices is not None:
            ep_start = int(self.episode_data_index["from"][ep_idx].item())
            ep_end = int(self.episode_data_index["to"][ep_idx].item())
            query_indices = {
                key: [max(ep_start, min(ep_end - 1, idx + delta)) for delta in delta_idx]
                for key, delta_idx in self.delta_indices.items()
            }
            padding = {
                f"{key}_is_pad": torch.BoolTensor(
                    [(idx + delta < ep_start) | (idx + delta >= ep_end) for delta in delta_idx]
                )
                for key, delta_idx in self.delta_indices.items()
            }

            # Query and stack non-video keys.
            query_result = {}
            for key, q_idx in query_indices.items():
                if key not in self.hf_dataset.column_names:
                    continue
                col = self.hf_dataset.select(q_idx)[key]
                # HF returns a Column; transforms are not guaranteed to apply here.
                # Convert each element to a tensor before stacking.
                query_result[key] = torch.stack([v if isinstance(v, torch.Tensor) else torch.as_tensor(v) for v in col])
            item = {**item, **padding, **query_result}

        # Add task as string like LeRobotDataset.
        if "task_index" in item:
            t_idx = int(item["task_index"].item() if hasattr(item["task_index"], "item") else item["task_index"])
            if t_idx in self.meta.tasks:
                item["task"] = self.meta.tasks[t_idx]

        return item
