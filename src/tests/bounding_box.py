import cv2
from camera_view import *
def get_bounding_box():
    # Example state (you can modify this as needed)
    state = np.array([-1.5, 1.5, 4.0, 0.0, 0.0, 0.0, 0.0])
    # state = np.array([-1.5, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0])

    # Define color thresholds for green and purple bars
    lower_green = np.array([20, 120, 120])
    upper_green = np.array([255, 255, 255])

    lower_purple = np.array([120, 20, 120])
    upper_purple = np.array([255, 255, 255])

    # Get the camera view and mask view (this is a placeholder, replace with actual function)
    camera_view, mask_view = get_camera_view(state)  # Ensure get_camera_view() is implemented

    # Apply color thresholding to get binary masks
    mask_green = cv2.inRange(mask_view, lower_green, upper_green)
    mask_purple = cv2.inRange(mask_view, lower_purple, upper_purple)

    # Detect contours from the green and purple masks
    green_contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    purple_contours, _ = cv2.findContours(mask_purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Combine both sets of contours for green and purple
    all_contours = green_contours + purple_contours

    bounding_boxes = []

    # Iterate over each detected contour
    for contour in all_contours:
        # Get the minimum area bounding box (rotated rectangle)
        rect = cv2.minAreaRect(contour)
        (px, py), (wbb, hbb), theta_bb = rect

        # Calculate box points for visualization
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # Classify the bar as vertical or horizontal based on aspect ratio
        if wbb < hbb:
            bar_label = 'horizontal'
            cv2.drawContours(camera_view, [box], 0, (0, 255, 0), 2)
        else:
            bar_label = 'vertical'
            cv2.drawContours(camera_view, [box], 0, (128, 0, 128), 2)

        # Store the bounding box information
        bounding_boxes.append([px, py, wbb, hbb, theta_bb, bar_label])

    print(bounding_boxes)

    # Display the camera view with bounding boxes (for visualization)
    cv2.imshow('RGB view', camera_view)
    cv2.imshow('Mask view', mask_view)
    cv2.imshow('Binary Mask', mask_green | mask_purple)
    cv2.waitKey(0)  # Wait for a key press to close the window

    return bounding_boxes

if __name__ == '__main__':

    get_bounding_box()