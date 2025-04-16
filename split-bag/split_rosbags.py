import rosbag2_py
import rclpy.serialization
from std_msgs.msg import String, Int64      #TODO Add import here if sync command data type is other than the ones included
import os
import yaml
from enum import Enum
from tqdm import tqdm
import tempfile

#TODO replace with custom values if required.
class SyncSignal(Enum):
    """
    Enum representing different synchronization commands used for timestamp syncing. The `data` field of the sync command message will be compared with the value of the corresponding enum.

    Attributes:
        DISCARD: Command to discard a timestamp.
        BEGIN: Command indicating the start of a recorded trajectory.
        END: Command indicating the end of a recorded trajectory.
    """
    BEGIN = "start"
    END = "goal"
    DISCARD = "discard"

def get_sync_signal_data_type(type):
    #TODO Add to map here if sync command data type is other than the ones included
    type_map = {
        "String": String,
        "Int64": Int64
    }
    type = type_map.get(type)
    return type

def load_input_params(file_path):
    with open(file=file_path) as file:
        return yaml.safe_load(file).get("rosbag_split_params")

def extract_valid_intervals(bag_path, storage_id, compression, sync_command_topic, sync_command_data_type):
    """
    Extract valid start-stop timestamp pairs from the `sync_command_topic` topic while handling discard signals.
    """
    if not compression:
        reader = rosbag2_py.SequentialReader()
    else:
        reader = rosbag2_py.SequentialCompressionReader()

    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions()
    reader.open(storage_options, converter_options)

    reader.set_filter(rosbag2_py.StorageFilter(topics=[sync_command_topic]))

    intervals = []
    start_time = None

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        msg = rclpy.serialization.deserialize_message(data, sync_command_data_type).data.strip()

        if msg == SyncSignal.BEGIN.value:                               # Start signal
            if start_time is None:                                      # Considers first start if multiple start pressed
                start_time = timestamp

        elif msg == SyncSignal.END.value and start_time is not None:    # End signal
            intervals.append((start_time, timestamp))
            start_time = None                                           # Reset start_time after pairing

        elif msg == SyncSignal.DISCARD.value:                           # Discard signal
            start_time = None                                           # Reset start_time to ignore the previous start signal

    del reader  # Close reader
    return intervals

def split_rosbag(bag_path, storage_id, intervals, compression, output_bag_path):
    """
    Reads messages from the original bag and writes valid ones to new bag files based on intervals.
    """
    if not compression:
        reader = rosbag2_py.SequentialReader()
    else:
        reader = rosbag2_py.SequentialCompressionReader()

    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions()
    reader.open(storage_options, converter_options)

    metadata = reader.get_metadata()
    topic_list = [t.topic_metadata.name for t in metadata.topics_with_message_count if t.topic_metadata.name != "/sync_command"]

    # Initialize writer
    writer = None
    bag_count = 0
    writing = False

    print(f"Writing to output bag path: {output_bag_path}")

    with tqdm(total=len(intervals), desc="Processing intervals", unit="bag") as pbar:
        while reader.has_next():
            topic, data, timestamp = reader.read_next()

            # Check if within any valid interval
            in_valid_range = any(start <= timestamp <= stop for start, stop in intervals)

            if in_valid_range and not writing:
                # Start writing a new bag
                bag_count += 1
                bag_name = bag_path.split('/')[-1]
                uri = os.path.join(output_bag_path, f"{bag_name}_{bag_count}")
                
                writer = rosbag2_py.SequentialWriter()
                writer.open(
                    rosbag2_py.StorageOptions(uri=uri, storage_id=storage_id),
                    rosbag2_py.ConverterOptions()
                )

                # Create topics in new bag
                for topic_name in topic_list:
                    for topic_info in reader.get_metadata().topics_with_message_count:
                        if topic_info.topic_metadata.name == topic_name:
                            writer.create_topic(topic_info.topic_metadata)

                writing = True

            if writing and in_valid_range and topic != "/sync_command":
                writer.write(topic, data, timestamp)

            elif writing and not in_valid_range:
                # Stop writing when out of range
                del writer
                writer = None
                writing = False
                pbar.update(1)  # Move progress

    del reader  # Close reader properly

if __name__ == "__main__":
    # Load params
    params = load_input_params("params.yaml")
    base_path = params.get("base_path")
    storage_id = params.get("storage_id")
    compression = params.get("compression")
    sync_command_topic = params.get("sync_command_topic")
    sync_command_data_type = get_sync_signal_data_type(params.get("sync_command_data_type"))

    # Validate params
    assert isinstance(base_path, str), "Invalid param base_path"
    assert storage_id in ['sqlite3', 'mcap'], "Invalid param storage_id"
    assert isinstance(compression, bool), "Invalid param compression"
    assert isinstance(sync_command_topic, str), "Invalid param sync_command_topic"
    assert sync_command_data_type is not None, "Invalid param sync_command_data_type"
    print(f"Splitting bag files using base_path: {base_path}, storage_id: {storage_id}, compression: {compression}.")

    bag_files = [f for f in os.listdir(base_path)]

    for bag_file_name in bag_files:
        bag_file_path = os.path.join(base_path, bag_file_name)

        # Saves the bag file to the same directory (base_path) with a suffix _split
        output_bag_path = os.path.join(base_path, f'{bag_file_name}_split')
        valid_intervals = extract_valid_intervals(bag_path=bag_file_path, storage_id=storage_id, compression=compression, sync_command_topic=sync_command_topic, sync_command_data_type=sync_command_data_type)
        print(f"Extracted trajectory interval timestamps: {valid_intervals}")
        split_rosbag(bag_path=bag_file_path, storage_id=storage_id, intervals=valid_intervals, compression=compression, output_bag_path=output_bag_path)

