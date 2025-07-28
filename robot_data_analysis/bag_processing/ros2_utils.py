import time
import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import PointCloud2, CompressedImage, LaserScan, Image
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer
from IPython import embed
import velodyne_decoder as vd

    
def get_synced_raw_messages_from_bag(
    b: tuple,
    imtopics,
    pcltopics,
    odomtopics,
    scantopics,
    trackingtopics=None,
    masktopics=None,
    rate: float = 4.0,
    ):
    bag, type_map = b
    message_count = {}
    for topic in bag.get_metadata().topics_with_message_count:
        message_count[topic.topic_metadata.name] = topic.message_count
    
    # Handle multiple image topics - collect ALL available topics
    if type(imtopics) == str:
        active_imtopics = [imtopics] if message_count.get(imtopics, 0) > 0 else []
    else:
        active_imtopics = []
        for imt in imtopics:
            if message_count.get(imt, 0) > 0:
                active_imtopics.append(imt)
    
    # Handle other topics (find first available)
    active_odomtopic = None
    if type(odomtopics) == str:
        if message_count.get(odomtopics, 0) > 0:
            active_odomtopic = odomtopics
    else:
        for ot in odomtopics:
            if message_count.get(ot, 0) > 0:
                active_odomtopic = ot
                break
    
    active_pcltopic = None
    if pcltopics:
        if type(pcltopics) == str:
            if message_count.get(pcltopics, 0) > 0:
                active_pcltopic = pcltopics
        else:
            for pt in pcltopics:
                if message_count.get(pt, 0) > 0:
                    active_pcltopic = pt
                    break
    
    active_scantopic = None
    if scantopics:
        if type(scantopics) == str:
            if message_count.get(scantopics, 0) > 0:
                active_scantopic = scantopics
        else:
            for st in scantopics:
                if message_count.get(st, 0) > 0:
                    active_scantopic = st
                    break
    
    active_trackingtopic = None
    if trackingtopics:
        if type(trackingtopics) == str:
            if message_count.get(trackingtopics, 0) > 0:
                active_trackingtopic = trackingtopics
        else:
            for tt in trackingtopics:
                if message_count.get(tt, 0) > 0:
                    active_trackingtopic = tt
                    break
    
    # Handle multiple mask topics - collect ALL available topics
    active_masktopics = []
    if masktopics:
        if type(masktopics) == str:
            if message_count.get(masktopics, 0) > 0:
                active_masktopics = [masktopics]
        else:
            for mt in masktopics:
                if message_count.get(mt, 0) > 0:
                    active_masktopics.append(mt)
    
    # Check if we have at least one image topic and required topics
    if not active_imtopics:
        print("!!!!!!!!!!!!!!!!!!No image topics found with messages!!!!!!!!!!!!!!!!!")
        return None, None, None, None, None, None, None
    
    if not active_odomtopic:
        print("!!!!!!!!!!!!!!!!!No odometry topics found with messages!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return None, None, None, None, None, None, None

    print(f"Active image topics: {active_imtopics}")
    print(f"Active odom topic: {active_odomtopic}")
    if active_pcltopic:
        print(f"Active PCL topic: {active_pcltopic}")
    if active_scantopic:
        print(f"Active scan topic: {active_scantopic}")
    if active_trackingtopic:
        print(f"Active tracking topic: {active_trackingtopic}")
    if active_masktopics:
        print(f"Active mask topics: {active_masktopics}")
    
    # Initialize data structures for multi-camera
    synced_imdata = {topic: [] for topic in active_imtopics}
    synced_odomdata = []
    synced_pcldata = []
    synced_scandata = []
    synced_trackingdata = []
    synced_maskdata = {topic: [] for topic in active_masktopics}
    
    currtime = None
    curr_imdata = {topic: None for topic in active_imtopics}
    curr_odomdata = None 
    curr_pcldata = None
    curr_scandata = None
    curr_trackingdata = None
    curr_maskdata = {topic: None for topic in active_masktopics}
    timestamps = []
    while bag.has_next():    
        topic, msg, t = bag.read_next()
        if topic not in type_map:
            continue
        if currtime is None:
            currtime = t/1e9
        t = t/1e9
        
        # Handle multiple image topics
        if topic in active_imtopics:
            curr_imdata[topic] = deserialize_message(msg, type_map[topic])
        elif topic == active_odomtopic:
            curr_odomdata = deserialize_message(msg, type_map[topic])
        elif topic == active_pcltopic:
            curr_pcldata = deserialize_message(msg, type_map[topic])
        elif topic == active_scantopic:
            curr_scandata = deserialize_message(msg, type_map[topic])
        elif topic == active_trackingtopic:
            curr_trackingdata = deserialize_message(msg, type_map[topic])
        elif topic in active_masktopics:
            curr_maskdata[topic] = deserialize_message(msg, type_map[topic])
        
        if (t - currtime) >= 1.0 / rate:
            # Sync logic - require at least one image and odom
            # For optional topics, only require them if they're being processed
            has_required_images = any(curr_imdata[topic] is not None for topic in active_imtopics)
            has_required_odom = curr_odomdata is not None
            has_required_pcl = curr_pcldata is not None if active_pcltopic else True
            has_required_scan = curr_scandata is not None if active_scantopic else True
            has_required_tracking = curr_trackingdata is not None if active_trackingtopic else True
            has_required_masks = any(curr_maskdata[topic] is not None for topic in active_masktopics) if active_masktopics else True
            
            if (has_required_images and has_required_odom and 
                has_required_pcl and has_required_scan and has_required_tracking and has_required_masks):
                
                # Add synchronized data for all available image topics
                for topic in active_imtopics:
                    if curr_imdata[topic] is not None:
                        synced_imdata[topic].append(curr_imdata[topic])
                        curr_imdata[topic] = None
                    else:
                        # If this camera doesn't have data at this timestamp, 
                        # we might want to skip this sync point or interpolate
                        # For now, we'll skip if any active camera is missing data
                        break
                # Only proceed if we successfully added data from all active cameras
                synced_odomdata.append(curr_odomdata)
                curr_odomdata = None
                
                if active_pcltopic and curr_pcldata is not None:
                    synced_pcldata.append(curr_pcldata)
                    curr_pcldata = None
                
                if active_scantopic and curr_scandata is not None:
                    synced_scandata.append(curr_scandata)
                    curr_scandata = None
                
                if active_trackingtopic and curr_trackingdata is not None:
                    synced_trackingdata.append(curr_trackingdata)
                    curr_trackingdata = None
                
                # Add synchronized mask data for all available mask topics
                for topic in active_masktopics:
                    if curr_maskdata[topic] is not None:
                        synced_maskdata[topic].append(curr_maskdata[topic])
                        curr_maskdata[topic] = None
                    
                currtime = t
                timestamps.append(t)

    return timestamps, synced_imdata, synced_odomdata, synced_pcldata, synced_scandata, synced_trackingdata, synced_maskdata