import rosbag
import ros_numpy
import numpy as np

def process_pointcloud2(msg) -> np.ndarray:
    """
    Process pcl data from a topic that publishes sensor_msgs/PointCloud2 and convert it to a numpy array
    """
    return ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)

def get_synced_raw_messages_from_bag(
    bag: rosbag.Bag,
    imtopics: List[str] or str,
    pcltopics: List[str] or str,
    odomtopics: List[str] or str,
    scantopics: List[str] or str,
    rate: float = 4.0,
):
    """
        Get image, PCL, scan and odom data from a bag file

        Args:
            bag (rosbag.Bag): bag file
            imtopics (list[str] or str): topic name(s) for image data
            pcltopics (list[str] or str): topic name(s) for pcl data
            odomtopics (list[str] or str): topic name(s) for odom data
            scantopics (list[str] or str): topic name(s) for scan data
            img_process_func (Any): function to process image data
            odom_process_func (Any): function to process odom data
            rate (float, optional): rate to sample data. Defaults to 4.0.
            ang_offset (float, optional): angle offset to add to odom data. Defaults to 0.0.
        Returns:
            img_data (list): list of PIL images
            pcl_data (list): list of ROS PCL objects
            scan_data (list): list of ROS scan objects
            traj_data (list): list of odom data
    """
    # check if bag has all topics   
    print(odomtopics, imtopics, pcltopics, scantopics) 
    odomtopic = None
    imtopic = None
    pcltopic = None
    scantopic = None
    if type(imtopics) == str:
        imtopic = imtopics
    else:
        for imt in imtopics:
            if bag.get_message_count(imt) > 0:
                imtopic = imt
                break
    if type(odomtopics) == str:
        odomtopic = odomtopics
    else:
        for ot in odomtopics:
            if bag.get_message_count(ot) > 0:
                odomtopic = ot
                break
    if type(pcltopics) == str:
        pcltopic = pcltopics
    else:
        pcltopic = None
        for ot in pcltopics:
            if bag.get_message_count(ot) > 0:
                pcltopic = ot
                break
    if type(scantopics) == str:
        scantopic = scantopics
    else:
        scantopic = None
        for st in scantopics:
            if bag.get_message_count(st) > 0:
                scantopic = st
                break
    if not imtopic or (len(pcltopics) > 0 and not pcltopic) or (len(scantopics) > 0 and not scantopic):
        return None, None, None, None, None
    
    synced_imdata = []
    synced_odomdata = []
    synced_pcldata = []
    synced_scandata = []
    # get start time of bag in seconds
    currtime = bag.get_start_time()

    curr_imdata = None
    curr_odomdata = None
    curr_pcldata = None
    curr_scandata = None
        
    for topic, msg, t in bag.read_messages(topics=[imtopic, odomtopic, pcltopic, scantopic]):
        if topic == imtopic:
            curr_imdata = msg
        elif topic == odomtopic:
            curr_odomdata = msg
        elif topic == pcltopic:
            curr_pcldata = msg
        elif topic == scantopic:
            curr_scandata = msg
        if (t.to_sec() - currtime) >= 1.0 / rate:
            if curr_imdata is not None and curr_odomdata is not None and (curr_pcldata is not None or pcltopic==None) and (curr_scandata is not None or scantopic == None):
                synced_imdata.append(curr_imdata)
                synced_odomdata.append(curr_odomdata)
                synced_pcldata.append(curr_pcldata)
                synced_scandata.append(curr_scandata)
                currtime = t.to_sec()
    return synced_imdata, synced_odomdata, synced_pcldata, synced_scandata
