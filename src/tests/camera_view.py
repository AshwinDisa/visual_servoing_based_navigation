import cv2
import numpy as np
import pdb

def get_camera_view(state):

    # canvas_image and canvas_mask should be 3072x3072 images as described
    canvas_size = 3072  # The size of the full canvas
    camera_view_size = 300  # The size of the camera view window (300x300)
    focal_length = 3.2  # Focal length in pixels
    pixel_per_meter = 300  # Conversion factor (1 meter = 300 pixels)

    canvas_image = cv2.imread("images/canvas_single_tower.png")
    canvas_mask = cv2.imread("images/mask_single_tower.png")

    # cv2.imshow('Canvas Image', canvas_image)
    # cv2.imshow('Canvas Mask', canvas_mask)
    # cv2.waitKey(0)  # Wait for a key press for visualization

    x, y, z, _, _, _, _ = state

    # The canvas is at 2.75m from the world frame, and the drone is initialized at (4, 1.5, 2)
    # Hence, the distance between the drone and the canvas is 2.75 - 1.5 = 1.25 meters
    canvas_distance = 2.75  # corrected distance in meters
    # canvas_distance = 2  # corrected distance in meters

    # Convert the robot's (x, y, z) coordinates from meters to pixels using the scaling factor
    x_px = int(x * pixel_per_meter + canvas_size / 2)
    z_px = int(-z * pixel_per_meter + canvas_size / 2)
    
    # The robot is looking directly at the canvas, we crop a 300x300px area around the current position
    half_view_size = camera_view_size // 2
    
    # Ensure the crop window stays within canvas bounds
    top_left_x = max(0, x_px - half_view_size)
    top_left_y = max(0, z_px - half_view_size)
    bottom_right_x = min(canvas_size, top_left_x + camera_view_size)
    bottom_right_y = min(canvas_size, top_left_y + camera_view_size)

    # Crop the camera view from the canvas image and mask
    camera_view = canvas_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    camera_mask = canvas_mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    # Optionally, you can visualize or save the camera view
    cv2.imshow('Camera View', camera_view)
    cv2.imshow('Camera Mask', camera_mask)
    cv2.waitKey(0)  # Wait for a key press for visualization

    return camera_view, camera_mask

if __name__ == "__main__":

    # state = np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # state = np.array([4.0, 1.5, 2.0, 0.0, 0.0, 0.0, 0.0])
    state = np.array([-1, 0.0, -3, 0.0, 0.0, 0.0, 0.0])
    # state = np.array([4, 1.5, 2.0, 0.0, 0.0, 0.0, 0.0])
    camera_view, camera_mask = get_camera_view(state)