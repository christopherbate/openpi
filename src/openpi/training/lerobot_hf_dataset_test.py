import numpy as np

from openpi.training.lerobot_hf_dataset import HFParquetLeRobotDataset


def test_hf_parquet_lerobot_dataset_delta_actions_and_padding():
    # Build a tiny in-memory HF dataset with 2 episodes of length 2 (total 4 frames).
    import datasets

    # Per-frame action is 7D; delta query will stack into (horizon, 7).
    actions = [np.arange(7) + i for i in range(4)]
    episode_index = [0, 0, 1, 1]
    task_index = [0, 0, 1, 1]

    hf = datasets.Dataset.from_dict(
        {
            "episode_index": episode_index,
            "task_index": task_index,
            "actions": actions,
        }
    )

    # Episodes metadata must be keyed by episode index.
    episodes = {
        0: {"length": 2},
        1: {"length": 2},
    }
    tasks = {
        0: "task0",
        1: "task1",
    }

    # Action horizon 3 @ fps=10 => delta indices [0,1,2].
    delta_timestamps = {"actions": [0.0, 0.1, 0.2]}
    ds = HFParquetLeRobotDataset.from_hf_dataset(
        repo_id="dummy/libero",
        hf_dataset=hf,
        delta_timestamps=delta_timestamps,
        fps=10,
        tasks=tasks,
        episodes=episodes,
    )

    # idx=1 is last frame of episode 0; idx+2 would cross episode boundary and should be padded/clamped.
    item = ds[1]
    assert item["task"] == "task0"
    assert item["actions"].shape == (3, 7)
    # First two are within episode, last one should be padded (clamped to ep_end-1 == idx 1).
    assert item["actions_is_pad"].shape == (3,)
    # Episode length is 2, so idx+1 and idx+2 are out of range and padded.
    assert item["actions_is_pad"].tolist() == [False, True, True]
