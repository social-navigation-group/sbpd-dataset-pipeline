# visualizes a trajectory by displaying images and point clouds side by side
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted  # Import natural sorting library
from trajectory_utils import pcl_to_grid
import pickle as pkl
def visualize_trajectory_interactive(folder_path):
    # Define paths to the subfolders
    imgs_folder = os.path.join(folder_path, "imgs")
    pcd_folder = os.path.join(folder_path, "pcd")

    # Get sorted lists of files in each subfolder

    img_files = natsorted([f for f in os.listdir(imgs_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    pcd_files = natsorted([f for f in os.listdir(pcd_folder) if f.endswith('.npz')])
    actions = {}
    if os.path.exists(os.path.join(folder_path, "actions.pkl")):
        with open(os.path.join(folder_path, "actions.pkl"), "rb") as f:
            actions = pkl.load(f)
    # Ensure the number of images matches the number of point clouds
    print("Number of images:", len(img_files))
    print("Number of point clouds:", len(pcd_files))
    if len(img_files) != len(pcd_files):
        print("Error: Number of images and point clouds do not match.")
        return

    # Create a matplotlib figure
    fig = plt.figure(figsize=(12, 6))

    # Create subplots for the image and point cloud
    ax1 = fig.add_subplot(1, 2, 1)
    img_plot = ax1.imshow(np.zeros((100, 100, 3), dtype=np.uint8))  # Placeholder
    ax1.set_title("Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2)
    scatter = ax2.scatter([], [], s=1, c='b')  # Placeholder
    ax2.set_title("Point Cloud (Top-Down View)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")   
    ax2.set_xlim([-10, 10])
    ax2.set_ylim([-10, 10])
    ax2.grid(False)  # Disable grid lines
    # Variable to track the current frame
    #ax3 = fig.add_subplot(1, 3, 3)
    current_frame = [0]
    occ_grid_params = {
        "width": 10,
        "height": 10,
        "resolution": 0.05 ,
        "occupied_cell_value": 100.0,
        "unoccupied_cell_value": 5.0,
        "inflation_radius": 0.25,
        "inflation_scale_factor": 0.2,
        "robot_width": 0.1,
        "robot_height": 0.1,
        }
    def update(frame):
        # Load the image
        img_path = os.path.join(imgs_folder, img_files[frame])
        print("Loading image:", img_files[frame])
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_plot.set_data(img)

        # Load the point cloud
        pcd_path = os.path.join(pcd_folder, pcd_files[frame])
        print("Loading point cloud:", pcd_files[frame])
        data = np.load(pcd_path)
        points = data['arr_0']  # Assuming the .npz file contains an array named 'points'

        # Filter points within a 10m sphere
        points = points[points[:,2]>-0.2]
        #filter out points that are too close to the robot
        #remove points that are too close to the robot
        points = points[np.linalg.norm(points[:,:2],axis=1)>1.0]  # Keep only points above the ground

        # Rotate the point cloud 90 degrees about the Z-axis
        # rotation_matrix = np.array([[1, 0, 0],
        #                  [0,  1, 0],
        #                  [0,  0, 1]])
        #points = points @ rotation_matrix.T

        # Update the scatter plot (top-down view: X-Y plane)
        scatter.set_offsets(points[:, :2])

        ax2.clear()  # Clear the existing plot
        ax2.scatter(points[:, 0], points[:, 1], s=1, c='b')  # Re-plot the scatter points
        
        # Plot 'good' and 'bad' actions if available
        if frame in actions:
            if 'good' in actions[frame]:
                good_points = np.array(actions[frame]['good'])
                good_points = np.hstack((good_points, -0.4 * np.ones((good_points.shape[0], 1))))  # Add z = -0.4
                ax2.scatter(good_points[:, 0], good_points[:, 1], s=10, c='g', label='Good Actions')  # Green points

            if 'bad' in actions[frame]:
                bad_points = np.array(actions[frame]['bad'])
                bad_points = np.hstack((bad_points, -0.4 * np.ones((bad_points.shape[0], 1))))  # Add z = -0.4
                ax2.scatter(bad_points[:, 0], bad_points[:, 1], s=10, c='r', label='Bad Actions')  # Red points
                print(f"Distance of last point from goal:{np.linalg.norm(bad_points[-1,:2]-good_points[-1,:2])}")
        
        ''''
        points = points[points[:,2]>-0.2]
        points = points[np.linalg.norm(points[:,:2],axis=1)>1.0]
        occ_grid = pcl_to_grid(points,occ_grid_params)
        ax3.imshow(occ_grid.T,origin='lower')
        ax3.scatter(occ_grid_params['height']/(occ_grid_params['resolution']*2),occ_grid_params['width']/(occ_grid_params['resolution']*2),s=100,c='r',label='robot')
        ax3.grid(False)
        '''
        # ax2.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, color='r', label='X-axis')  # X-axis
        # ax2.quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, color='g', label='Y-axis')  # Y-axis
        ax2.set_title(f"Point Cloud (Top-Down View) - Timestep {frame + 1}")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_xlim([-10, 10])
        ax2.set_ylim([-10, 10])
        ax2.grid(False)  # Disable grid lines

        # Update titles
        ax1.set_title(f"Image - Timestep {frame + 1}")
        ax2.set_title(f"Point Cloud (Top-Down View) - Timestep {frame + 1}")

        fig.canvas.draw_idle()
        #save the figure to a file
        fig.savefig(os.path.join(folder_path, f"frame_{frame}.png"))
        
    def on_key(event):
        if event.key == 'right':
            current_frame[0] = (current_frame[0] + 1) % len(img_files)
            update(current_frame[0])
        elif event.key == 'left':
            current_frame[0] = (current_frame[0] - 1) % len(img_files)
            update(current_frame[0])

    # Connect the key press event
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Initialize the first frame
    update(current_frame[0])

    # Show the figure
    plt.show()

def save_trajectory_images(folder_path):
    # Define paths to the subfolders
    imgs_folder = os.path.join(folder_path, "imgs")
    pcd_folder = os.path.join(folder_path, "pcd")

    # Get sorted lists of files in each subfolder

    img_files = natsorted([f for f in os.listdir(imgs_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    pcd_files = natsorted([f for f in os.listdir(pcd_folder) if f.endswith('.npz')])
    actions = {}
    if os.path.exists(os.path.join(folder_path, "actions.pkl")):
        with open(os.path.join(folder_path, "actions.pkl"), "rb") as f:
            actions = pkl.load(f)
    # Ensure the number of images matches the number of point clouds
    print("Number of images:", len(img_files))
    print("Number of point clouds:", len(pcd_files))
    if len(img_files) != len(pcd_files):
        print("Error: Number of images and point clouds do not match.")
        return

    # Create a matplotlib figure
    fig = plt.figure(figsize=(12, 6))

    # Create subplots for the image and point cloud
    ax1 = fig.add_subplot(1, 2, 1)
    img_plot = ax1.imshow(np.zeros((100, 100, 3), dtype=np.uint8))  # Placeholder
    ax1.set_title("Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2)
    scatter = ax2.scatter([], [], s=1, c='b')  # Placeholder
    ax2.set_title("Point Cloud (Top-Down View)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")   
    ax2.set_xlim([-10, 10])
    ax2.set_ylim([-10, 10])
    ax2.grid(False)  # Disable grid lines
    # Variable to track the current frame
    #ax3 = fig.add_subplot(1, 3, 3)
    current_frame = [0]
    occ_grid_params = {
        "width": 10,
        "height": 10,
        "resolution": 0.05 ,
        "occupied_cell_value": 100.0,
        "unoccupied_cell_value": 5.0,
        "inflation_radius": 0.25,
        "inflation_scale_factor": 0.2,
        "robot_width": 0.1,
        "robot_height": 0.1,
        }
    for frame in range(len(img_files)):
        # Load the image
        img_path = os.path.join(imgs_folder, img_files[frame])
        print("Loading image:", img_files[frame])
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_plot.set_data(img)

        # Load the point cloud
        pcd_path = os.path.join(pcd_folder, pcd_files[frame])
        print("Loading point cloud:", pcd_files[frame])
        data = np.load(pcd_path)
        points = data['arr_0']  # Assuming the .npz file contains an array named 'points'

        # Filter points within a 10m sphere
        points = points[points[:,2]>-0.2]
        #filter out points that are too close to the robot
        #remove points that are too close to the robot
        points = points[np.linalg.norm(points[:,:2],axis=1)>0.6]  # Keep only points above the ground

        # Rotate the point cloud 90 degrees about the Z-axis
        # rotation_matrix = np.array([[1, 0, 0],
        #                  [0,  1, 0],
        #                  [0,  0, 1]])
        #points = points @ rotation_matrix.T

        # Update the scatter plot (top-down view: X-Y plane)
        scatter.set_offsets(points[:, :2])

        ax2.clear()  # Clear the existing plot
        ax2.scatter(points[:, 0], points[:, 1], s=1, c='b')  # Re-plot the scatter points
        
        # Plot 'good' and 'bad' actions if available
        if frame in actions:
            if 'good' in actions[frame]:
                good_points = np.array(actions[frame]['good'])
                good_points = np.hstack((good_points, -0.4 * np.ones((good_points.shape[0], 1))))  # Add z = -0.4
                ax2.scatter(good_points[:, 0], good_points[:, 1], s=10, c='g', label='Good Actions')  # Green points

            if actions[frame].get('bad') is not None:
                bad_points = np.array(actions[frame]['bad'])
                bad_points = np.hstack((bad_points, -0.4 * np.ones((bad_points.shape[0], 1))))  # Add z = -0.4
                ax2.scatter(bad_points[:, 0], bad_points[:, 1], s=10, c='r', label='Bad Actions')  # Red points
                print(f"Distance of last point from goal:{np.linalg.norm(bad_points[-1,:2]-good_points[-1,:2])}")
        
        ''''
        points = points[points[:,2]>-0.2]
        points = points[np.linalg.norm(points[:,:2],axis=1)>1.0]
        occ_grid = pcl_to_grid(points,occ_grid_params)
        ax3.imshow(occ_grid.T,origin='lower')
        ax3.scatter(occ_grid_params['height']/(occ_grid_params['resolution']*2),occ_grid_params['width']/(occ_grid_params['resolution']*2),s=100,c='r',label='robot')
        ax3.grid(False)
        '''
        # ax2.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, color='r', label='X-axis')  # X-axis
        # ax2.quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, color='g', label='Y-axis')  # Y-axis
        ax2.set_title(f"Point Cloud (Top-Down View) - Timestep {frame + 1}")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_xlim([-10, 10])
        ax2.set_ylim([-10, 10])
        ax2.grid(False)  # Disable grid lines

        # Update titles
        ax1.set_title(f"Image - Timestep {frame + 1}")
        ax2.set_title(f"Point Cloud (Top-Down View) - Timestep {frame + 1}")

        fig.canvas.draw_idle()
        #save the figure to a file
        fig.savefig(os.path.join(folder_path, f"frame_{frame}.png"))
        
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_trajectory.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        sys.exit(1)

    save_trajectory_images(folder_path)