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
    rate: float = 4.0,
    ):
    bag,type_map = b
    message_count = {}
    for topic in bag.get_metadata().topics_with_message_count:
        message_count[topic.topic_metadata.name] = topic.message_count
    #embed()
    if type(imtopics) == str:
        imtopic = imtopics
    else:
        for imt in imtopics:
            if message_count.get(imt,0) > 0:
                imtopic = imt
                break
    if type(odomtopics) == str:
        odomtopic = odomtopics
    else:
        for ot in odomtopics:
            if message_count.get(ot,0) > 0:
                odomtopic = ot
                break
    if type(pcltopics) == str:
        pcltopic = pcltopics
    else:
        pcltopic = None
        for ot in pcltopics:
            if message_count.get(ot,0) > 0:
                pcltopic = ot
                break
    if type(scantopics) == str:
        scantopic = scantopics
    else:
        scantopic = None
        for st in scantopics:
            if message_count.get(st,0) > 0:
                scantopic = st
                break
    if type(trackingtopics) == str:
        trackingtopic = trackingtopics
    else:
        trackingtopic = None
        for tt in trackingtopics:
            if message_count.get(tt,0) > 0:
                trackingtopic = tt
                break
            
    if not imtopic or (len(pcltopics) > 0 and not pcltopic) or (len(scantopics) > 0 and not scantopic) or (len(odomtopics) > 0 and not odomtopic) or (len(trackingtopics) > 0 and not trackingtopic):
        return None, None, None, None, None
    
    synced_imdata = []
    synced_odomdata = []
    synced_pcldata = []
    synced_scandata = []
    synced_trackingdata = []
    
    currtime = None
    curr_imdata = None
    curr_odomdata = None 
    curr_pcldata = None
    curr_scandata = None
    curr_trackingdata = None
    
    while bag.has_next():    
        topic, msg, t = bag.read_next()
        # if topic == '/tracks_camera_camera_color_image_raw_compressed':
        #     print("s")
        if topic not in type_map:
            continue
        if currtime is None:
            currtime = t/1e9
        t = t/1e9
        if topic == imtopic:
            curr_imdata = deserialize_message(msg, type_map[imtopic])
        elif topic == odomtopic:
            curr_odomdata = deserialize_message(msg, type_map[odomtopic])
        elif topic == pcltopic:
            curr_pcldata = deserialize_message(msg, type_map[pcltopic])
        elif topic == scantopic:
            curr_scandata = deserialize_message(msg, type_map[scantopic])
        elif topic == trackingtopic:
            curr_trackingdata = deserialize_message(msg, type_map[trackingtopic])
        if (t - currtime) >= 1.0 / rate:
            if curr_imdata is not None and curr_odomdata is not None and (curr_pcldata is not None or pcltopic==None) and (curr_scandata is not None or scantopic == None):
                synced_imdata.append(curr_imdata)
                synced_odomdata.append(curr_odomdata)
                synced_pcldata.append(curr_pcldata)
                synced_scandata.append(curr_scandata)
                synced_trackingdata.append(curr_trackingdata)
                
                # curr_imdata = None
                # curr_odomdata = None 
                # curr_pcldata = None
                # curr_scandata = None
                currtime = t
                curr_trackingdata = None

    return synced_imdata, synced_odomdata, synced_pcldata, synced_scandata, synced_trackingdata