import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
import glob, random
import os 

matplotlib.rcParams['interactive'] = True

def get_mask(img, visualize=False):
    # Create single-channel mask instead of multi-channel
    mask_img = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)*255
    for i in range(int(mask_img.shape[0]/2)):
        for j in range(i):
            mask_img[i, j] = 0
    for i in range(int(mask_img.shape[0]/2), mask_img.shape[0]):
        for j in range(0, mask_img.shape[1]-i-1):
            mask_img[i, j] = 0

    if visualize:
        cv2.imshow('disp', mask_img)
        cv2.waitKey(0)

    return mask_img


def get_affine_mat(x, y, theta):
    """
    Returns the affine transformation matrix for the given parameters.
    """
    theta = np.deg2rad(theta)
    return np.array([[np.cos(theta), -np.sin(theta), x],
                     [np.sin(theta), np.cos(theta) , y],
                     [0            , 0             , 1]])


def get_affine_matrix_quat(x, y, quaternion):
    theta = R.from_quat(quaternion).as_euler('XYZ')[2]
    return np.array([[np.cos(theta), -np.sin(theta), x],
                     [np.sin(theta), np.cos(theta), y],
                     [0, 0, 1]])

if __name__ == '__main__':
    plt.ion()

    # open the pickle file
    # data = pickle.load(open('data/2021-11-13-17-09-18_data.pkl', 'rb'), encoding='latin1')
    bp = '/media/dataset_access/cmu_test_move_base/processed/'
    traj = 'deck_5_with_overhead_2025-03-26_12-59-49_1_corrected_merged'
    # file_paths = glob.glob(os.path.join(bp,traj))
    file_paths = [os.path.join(bp,traj+'_data.pkl')]
    for file_path in file_paths:
        print('reading file : ', file_path)

        data = pickle.load(open(file_path, 'rb'), encoding='latin1')
        print(data.keys())

        
        cmd_vel = []

        # plot the joystick data
        plt.figure(figsize=(10, 5))
        data['joystick'] = np.asarray(data['joystick'])
        x_goal, y_goal = [], []

        # plt.figure(figsize=(10, 5))
        # plt.subplot(2, 1, 1)
        # plt.plot(data['joystick'][:, 0])
        # plt.subplot(2, 1, 2)
        # plt.plot(data['joystick'][:, 2])
        # data['joystick'][:, 0] = savgol_filter(np.asarray(data['joystick'])[:, 0], 19, 3)
        # data['joystick'][:, 2] = savgol_filter(np.asarray(data['joystick'])[:, 2], 19, 3)
        # plt.subplot(2, 1, 1)
        # plt.plot(data['joystick'][:, 0])
        # plt.subplot(2, 1, 2)
        # plt.plot(data['joystick'][:, 2])
        # plt.show()

        for i in range(len(data['pose'])):
            print(f"Reading: {os.path.join(bp,traj+'_data', f'{i+1}.png')}")
            img = np.asarray(cv2.imread(os.path.join(bp,traj+'_data', f'{i+1}.png')))
            x_map_c, y_map_c, yaw_map_c = data['pose'][i]

            # img = data['bevlidarimg'][i]
            mask_img = get_mask(img, visualize=False)
            # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # draw the spot robot as a green rectangle
            img = cv2.rectangle(img, (190, 197), (200, 203), (0, 255, 0), -1)

            T_map_c = get_affine_mat(x_map_c, y_map_c, yaw_map_c)

            # for k in range(i+1, len(data['pose'])-10):
            #     x_map_f, y_map_f, yaw_map_f = data['pose'][k+1]
            #     T_map_f = get_affine_mat(x_map_f, y_map_f, yaw_map_f)
            #
            #     T_c_f = np.linalg.inv(T_map_c) @ T_map_f
            #
            #     dist = np.linalg.norm(T_c_f[:2, 2])
            #     # if dist > 5:
            #     #     x_goal.append(T_c_f[0, 2])
            #     #     y_goal.append(T_c_f[1, 2])
            #     #     break
            #     # if k >= 200: break
            #
            #     t_f_pixels = [int(T_c_f[0, 2]/0.05)+200, int(-T_c_f[1, 2]/0.05)+200]
            #
            #     # draw circle at t_f_pixel
            #     img = cv2.circle(img, (t_f_pixels[0], t_f_pixels[1]), 1, (0, 0, 255), -1)

            T_odom_robot = get_affine_matrix_quat(data['odom'][i][0], data['odom'][i][1], data['odom'][i][2])
            for goal in data['human_expert_odom'][i]:
                T_odom_goal = get_affine_matrix_quat(goal[0], goal[1], goal[2])
                T_robot_goal = np.linalg.pinv(T_odom_robot) @ T_odom_goal
                # local_goal_list.append([T_robot_goal[0, 2], T_robot_goal[1, 2]])
                t_f_pixels = [int(T_robot_goal[0, 2] / 0.05) + 200, int(-T_robot_goal[1, 2] / 0.05) + 200]
                img = cv2.circle(img, (t_f_pixels[0], t_f_pixels[1]), 1, (0, 0, 255), -1)

            print('points in red line : ', len(data['human_expert_odom'][i]))

            local_goal_list = []
            if data['move_base_path'][i] is not None:
                for k in range(len(data['move_base_path'][i])):
                    # if k >= 200: break
                    x_map_f, y_map_f, orientation_f = data['move_base_path'][i][k]
                    T_map_f = get_affine_matrix_quat(x_map_f, y_map_f, orientation_f)
                    T_c_f = np.linalg.inv(T_map_c) @ T_map_f
                    t_f_pixels = [int(T_c_f[0, 2]/0.05)+200, int(-T_c_f[1, 2]/0.05)+200]
                    local_goal_list.append(t_f_pixels)

                if len(local_goal_list) > 200:
                    # more than 200 points exist on the blue line - need to subsample
                    local_goal_list = [local_goal_list[i] for i in sorted(random.sample(range(len(local_goal_list)), 200))]

                for x in range(len(local_goal_list)):
                    # draw circle at t_f_pixel
                    img = cv2.circle(img, (local_goal_list[x][0], local_goal_list[x][1]), 1, (255, 0, 0), -1)

                print('actual points in blue line : ', len(data['move_base_path'][i]))
                print('filtered points in blue line : ', len(local_goal_list))



            # # mask img with mask_img
            # img = cv2.bitwise_and(img, img, mask=mask_img)


            # img = cv2.circle(img, (200, 200), 2, (0, 255, 0), -1)

            # add text on the img
            #img = cv2.putText(img, 'v_x: {:.2f}'.format(data['joystick'][i][0]), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            #img = cv2.putText(img, 'v_y: {:.2f}'.format(data['joystick'][i][1]), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            #img = cv2.putText(img, 'v_w: {:.2f}'.format(data['joystick'][i][2]), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow('dummy', img)
            cv2.waitKey(0)


        # plt.figure(figsize=(10, 5))
        # plt.plot(np.arange(len(x_goal)), x_goal, 'r')
        # plt.figure(figsize=(10, 5))
        # plt.plot(np.arange(len(y_goal)), y_goal, 'b')
        # plt.show()
