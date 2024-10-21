import cv2
import numpy as np
import pdb

def get_robot_position(state):
    """
    Extracts the robot's (x, y, z) position from its state vector.
    
    Parameters:
    - state: List or tuple representing the robot's full state (e.g., position, velocity).
    
    Returns:
    - x, y, z: The robot's x, y, and z coordinates in the world frame.
    """
    x, y, z, *_ = state  # Unpack the first three elements (x, y, z), ignoring the rest
    return x, y, z

def get_camera_pixel_coordinates(x, z, depth, focal_length, canvas_size):
    """
    Converts the robot's (x, z) world coordinates to pixel coordinates in the camera frame.
    
    Parameters:
    - x, z: Robot's x and z world coordinates.
    - depth: Distance from the robot to the object in the depth (camera's view) direction.
    - focal_length: Focal length of the camera.
    - canvas_size: Size of the canvas image (assumed to be square).
    
    Returns:
    - x_px, z_px: Pixel coordinates corresponding to the robot's (x, z) position.
    """
    # Convert world coordinates (x, z) into image pixel coordinates
    x_px = int((x / depth) * focal_length)  # Scale x by depth and focal length
    y_px = int((-z / depth) * focal_length) + canvas_size  # Scale z similarly, adjust for canvas size, negative for y-axis
    return x_px, y_px

def crop_view(x_px, y_px, camera_view_size, canvas_size, canvas_image, canvas_mask):
    """
    Crops a subregion (camera view) from the larger canvas image based on pixel coordinates.
    
    Parameters:
    - x_px, z_px: Pixel coordinates of the robot's position in the camera frame.
    - camera_view_size: Size of the camera view window to crop.
    - canvas_size: Size of the full canvas image.
    - canvas_image: Image of the full environment (canvas).
    - canvas_mask: Mask image corresponding to the canvas (for object detection).
    
    Returns:
    - camera_view: Cropped portion of the canvas image around the robot's position.
    - camera_mask: Cropped mask corresponding to the same area for object detection.
    """
    half_view_size = camera_view_size // 2  # Calculate half of the view size to center the crop

    # Ensure the crop stays within the canvas boundaries
    top_left_x = max(0, x_px - half_view_size)
    top_left_y = max(0, y_px - half_view_size)
    bottom_right_x = min(canvas_size, top_left_x + camera_view_size)
    bottom_right_y = min(canvas_size, top_left_y + camera_view_size)

    # Crop both the image and the mask based on the computed coordinates
    camera_view = canvas_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x].copy()
    camera_mask = canvas_mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x].copy()

    return camera_view, camera_mask

def detect_contours(mask_view, lower, upper, label):
    """
    Detects contours of objects within a specified color range and extracts bounding boxes.
    
    Parameters:
    - mask_view: Masked view of the camera for object detection.
    - lower: Lower bound of the color range for detection (in BGR format).
    - upper: Upper bound of the color range for detection (in BGR format).
    - label: Label for identifying the type of object detected (e.g., 'vertical', 'horizontal').
    
    Returns:
    - bounding_boxes: A list of bounding boxes with parameters (x, y, width, height, angle, label).
    """
    # Create a binary mask for the specified color range
    mask = cv2.inRange(mask_view, np.array(lower), np.array(upper))
    
    # Find contours of objects in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract bounding boxes from each contour using the minimum area rectangle
    bounding_boxes = [
        [px, py, wbb, hbb, theta_bb, label]  # (x, y, width, height, angle, label)
        for contour in contours
        for (px, py), (wbb, hbb), theta_bb in [cv2.minAreaRect(contour)]
    ]
    return bounding_boxes

def draw_bounding_boxes(image, bounding_boxes):
    """
    Draws bounding boxes on the image for visualization purposes.
    
    Parameters:
    - image: Original image on which to draw the bounding boxes.
    - bounding_boxes: List of bounding boxes with their properties (x, y, width, height, angle, label).
    
    Returns:
    - image_with_boxes: Image with the bounding boxes and center points drawn on it.
    """
    image_with_boxes = image.copy()  # Make a copy to avoid modifying the original image

    for bounding_box in bounding_boxes:
        px, py, wbb, hbb, theta_bb, bar_label = bounding_box
        
        # Get the four corners of the bounding box from the rectangle
        box_points = cv2.boxPoints(((px, py), (wbb, hbb), theta_bb))
        box_points = np.intp(box_points)  # Convert to integer points for drawing
        
        # Use different colors for different labels (vertical = purple, horizontal = green)
        color = (128, 0, 128) if bar_label == 'vertical' else (0, 255, 0)
        
        # Draw the contour (rectangle) and the center point
        cv2.drawContours(image_with_boxes, [box_points], 0, color, 2)
        cv2.circle(image_with_boxes, (int(px), int(py)), 5, (0, 0, 255), -1)  # Draw the center as a red dot

    return image_with_boxes

def get_matching_bounding_box(bounding_boxes, bar_label):
    """
    Finds the bounding box that matches a specified label (e.g., 'vertical', 'horizontal').
    
    Parameters:
    - bounding_boxes: List of bounding boxes with their properties.
    - bar_label: Label of the bounding box to search for.
    
    Returns:
    - bounding_box: The bounding box that matches the specified label, or None if not found.
    """
    for bounding_box in bounding_boxes:
        if bounding_box[-1] == bar_label:  # Check if the label matches
            return bounding_box
    return None  # Return None if no matching bounding box is found

def update_trajectory_state(robot, bounding_boxes, params):
    """
    Updates the robot's trajectory state based on the detected bounding boxes.
    
    Parameters:
    - robot: Robot object or data structure holding the robot's state and parameters.
    - bounding_boxes: List of bounding boxes detected in the current frame.
    - params: Dictionary of parameters to control the trajectory logic (e.g., counter thresholds).
    
    Returns:
    - None: The robot's state is updated in place.
    """
    count_bounding_boxes = len(bounding_boxes)  # Count how many bounding boxes are detected

    # If the difference in bounding boxes matches predefined values, update the robot's counter
    if count_bounding_boxes - robot.prev_count_bounding_boxes in params['count_difference_values']:
        counter_name = params['counter']
        setattr(robot, counter_name, getattr(robot, counter_name) + 1)  # Increment the counter
        print(f"Detected {getattr(robot, counter_name)} {counter_name} bar")  # Print the current progress

    # If the counter exceeds the threshold, transition to the next state
    if getattr(robot, params['counter']) > params['counter_threshold']:
        robot.execute = params['next_state']  # Update the robot's execution state
        setattr(robot, params['counter'], 0)  # Reset the counter
        print("--------------------------")
        print(f"{params['message']}")  # Print the transition message
        print("--------------------------")

    # Update the previous bounding box count for the next iteration
    robot.prev_count_bounding_boxes = count_bounding_boxes
