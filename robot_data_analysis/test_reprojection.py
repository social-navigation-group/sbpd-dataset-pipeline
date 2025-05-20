import open3d as opd
import numpy as np
import math
import cv2
from scipy.spatial.transform import Rotation as R
import os
import PIL
# gets the rotation matrix
def eulerangles_to_rotmat(roll, pitch, yaw):
  rotmat_roll = np.array(
    [
      [1, 0, 0],
      [0, math.cos(roll), -math.sin(roll)],
      [0, math.sin(roll), math.cos(roll)]
    ]
  )
  rotmat_pitch = np.array(
    [
      [math.cos(pitch), 0, math.sin(pitch)],
      [0, 1, 0],
      [-math.sin(pitch), 0, math.cos(pitch)]
    ]
  )
  rotmat_yaw = np.array(
    [
      [math.cos(yaw), -math.sin(yaw), 0],
      [math.sin(yaw),  math.cos(yaw), 0],
      [0, 0, 1]
    ]
  )
  rotmat = np.matmul(np.matmul(rotmat_yaw , rotmat_pitch), rotmat_roll)
  return rotmat

n=47
root = '/media/shashank/T/processed/go2nus/utown_new_1_2025-04-29_11-16-17_10_merged_0/'
sample_image_path = os.path.join(root,'imgs',f'{n}.jpg')
sample_image = np.asarray(PIL.Image.open(sample_image_path))
sample_pcd_path = os.path.join(root,'pcd',f'{n}.npz')
sample_pcd = np.load(sample_pcd_path)['arr_0']# path the pcd file to load the pcd file
point_cloud = sample_pcd

# optical extrinsic from the tool.
# roll = math.radians(-91.441)
# pitch = math.radians(0.133)
# yaw = math.radians(-90.209)
# px = 0.090
# py = 0.030
# pz = -0.238
quat = np.array([0.4951063278176592,
      -0.5021507076582918,
      0.5055448231067193,
      -0.4971305892652202])
translation_vector = np.array([0.1398121462236138,
      0.05367758784500739,
      -0.012879661462024712,])
rotation_matrix = R.from_quat(quat).as_matrix()


# convert lidar 3d points to image 3d points
all_image_3d_points = []
for lidar_3d_point in point_cloud:
    image_3d_point = np.matmul(np.linalg.inv(rotation_matrix), lidar_3d_point - translation_vector)
    all_image_3d_points.append(image_3d_point)

# read the image file
image = cv2.imread(sample_image_path)

# intrinsic from the tool
fx= 607.6533813476563
fy= 607.3688354492188
cx= 420.5920104980469
cy= 247.97695922851563

k1= 0
k2= 0
k3= 0
k4= 0

camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float64)

distortion = np.array([[k1], [k2], [k3], [k4]], dtype=np.float64)
rotation_vector = np.array([[0.],
       [0.],
       [0.]])
translation_vector = np.array([[0.],
       [0.],
       [0.]])


for image_3d_point in all_image_3d_points:
    if image_3d_point[2] >=0:
        projected_points, _ = cv2.fisheye.projectPoints( np.array([np.array([image_3d_point], dtype='float32')], dtype='float32'), rotation_vector,
                                                        translation_vector, camera_matrix, distortion)
        projected_point_in_2d = projected_points[0][0]
        if projected_point_in_2d[0] >= 0 and projected_point_in_2d[0] < image.shape[1] and projected_point_in_2d[1] >= 0 and projected_point_in_2d[1] < image.shape[0]:
          image = cv2.circle(image,(int(projected_point_in_2d[0]),int(projected_point_in_2d[1])),1, (0,0,255))


cv2.imwrite('project_lidar_points_image.png',image)
