# ROS 2 bag splitting according to sync signals

Script to split the rosbag according to the start-end signals, and discarding segments as per the discard signal.

## Splitting Logic

With respect to signals on the sync command topic, the logic is indicated below:

- `start` --- *other data* --- `end` : Trajectory is split.
- `start` --- *other data* --- `discard` : Start signal is discarded.
- `end`/`discard` with no corresponding `start`: End/discard signal is ignored.


## Input parameters

Loaded from [params.yaml](./params.yaml):

```yaml
rosbag_split_params:
  base_path:  str                  # path to directory containing bag files
  storage_id: str                  # {sqlite3, mcap}
  compression: bool                # whether rosbags are compressed
  sync_command_topic: str          # sync command topic
  sync_command_data_type: str      # currently supported {String, Int64}
```

The script `split_rosbags.py` will split all rosbags located at the root of the `base_path` directory.

## Static transforms

The script assumes that the static transform data is available on the `/tf_static` topic. It is read when processing the sync signals, and is written to every split rosbag with updated timestamps.

## Output

The split bag files to the same base directory, in a directory with a suffix **_split** for every bag file directory in the base directory.

Note: The current implementation does not produce compressed/split bag files. 

## TODOs before running the script

1. Please replace the values of the Enum `SyncSignal` as per the values of the sync signal messages' `data` fields. This has been indicated by a **TODO** tag in the script.
2. If the sync_command_data_type is not among the supported ones i.e. {String, Int64}, a minor refactoring is required. Please refer to the **TODO** tag in the `get_sync_signal_data_type` function.

