import argparse
import cv2
import numpy as np
from base_visual_servoing import BaseVisualServoing
from controllers_integrators import Controller, Integrator  # Import from the new module

class VisualServoingRobot(BaseVisualServoing):
    def __init__(self, integrator, controller):
        super().__init__()

        self.pixel_per_meter = 300
        self.canvas_size = 3072
        self.camera_view_size = 300
        self.focal_length = 3.2
        self.canvas_image = cv2.imread("images/canvas_single_tower.png")
        self.canvas_mask = cv2.imread("images/mask_single_tower.png")

        self.controller = controller
        self.integrator = integrator

        self.prev_errors = [-31, 78, 0]
        self.state = np.array([4, 1.5, 2, 0, 0, 0, 0])
        self.controls = np.array([0, 0, 0])

        self.dt = 0.01

    def vehicle_dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        x, y, z, theta, x_dot, y_dot, z_dot = state
        thrust, roll_rate, y_ddot = control

        g = 9.81
        d = 4

        x_ddot = thrust * np.sin(theta) - d * x_dot
        y_ddot = y_ddot - d * y_dot
        z_ddot = thrust * np.cos(theta) - g - d * z_dot
        theta_dot = roll_rate

        state_dot = np.array([
            x_dot,
            y_dot,
            z_dot,
            theta_dot,
            x_ddot,
            y_ddot,
            z_ddot
        ])

        return state_dot

    def step(self):
        self.controls, self.prev_errors = self.controller.compute_control(self.get_bounding_box(), self.state, self.prev_errors, self.dt)
        self.state = self.integrator.integrate(self.vehicle_dynamics, self.state, self.controls, self.dt)

    def get_camera_view(self):
        """
        Generates a synthetic camera view based on the robot's current pose and the canvas image.
        This function crops an image from the larger canvas according to the robot's state.
        The resulting cropped image is expected to simulate the camera perspective, accounting for focal length.

        Returns:
        tuple: (camera_view, camera_mask) where both are 300x300 pixel images.
        """

        # Extract the robot's position in world coordinates (X, Y, Z)
        x, y, z, _, _, _, _ = self.state

        # Fixed distance between the drone and the canvas (drone is 2 meters away)
        Z_drone = 2.0  # The depth of the drone from the canvas

        # The focal length of the camera in pixels (provided as 900 pixels)
        focal_length_px = 900

        # Convert the robot's (x, y) coordinates to pixel coordinates on the canvas using the projection formula
        # u = f * (X / Z), v = f * (Y / Z)
        # We are only using X (horizontal) and Z (vertical) here for simplicity.
        x_px = int((x / Z_drone) * focal_length_px + self.canvas_size / 2)
        z_px = int((-z / Z_drone) * focal_length_px + self.canvas_size / 2)  # Mapping Z to the vertical axis in the canvas

        # The robot is looking directly at the canvas, we crop a 300x300px area around the current position
        half_view_size = self.camera_view_size // 2

        # Ensure the crop window stays within canvas bounds
        top_left_x = max(0, x_px - half_view_size)
        top_left_y = max(0, z_px - half_view_size)
        bottom_right_x = min(self.canvas_size, top_left_x + self.camera_view_size)
        bottom_right_y = min(self.canvas_size, top_left_y + self.camera_view_size)

        # Crop the camera view from the canvas image and mask
        camera_view = self.canvas_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x].copy()
        camera_mask = self.canvas_mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x].copy()

        return camera_view, camera_mask

    def get_bounding_box(self):
        """
        Extracts bounding boxes around detected structures in the camera image using the segmentation mask.

        Returns:
        A list of bounding box parameters for different structures in the camera image
        """

        # Define color thresholds for green and purple bars
        lower_green = np.array([20, 120, 120])
        upper_green = np.array([255, 255, 255])

        lower_purple = np.array([120, 20, 120])
        upper_purple = np.array([255, 255, 255])

        # Get the camera view and mask view (this is a placeholder, replace with actual function)
        camera_view, mask_view = self.get_camera_view()  # Ensure get_camera_view() is implemented

        # Apply color thresholding to get binary masks
        mask_green = cv2.inRange(mask_view, lower_green, upper_green)
        mask_purple = cv2.inRange(mask_view, lower_purple, upper_purple)

        # Detect contours from the green and purple masks
        green_contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        purple_contours, _ = cv2.findContours(mask_purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []

        camera_view_with_boxes = camera_view.copy()

        # Iterate over each detected green contour
        for contour in green_contours:
            # Get the minimum area bounding box (rotated rectangle)
            rect = cv2.minAreaRect(contour)
            (px, py), (wbb, hbb), theta_bb = rect

            # Calculate box points for visualization
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            # Label as green and draw green contours
            bar_label = 'horizontal'
            cv2.drawContours(camera_view_with_boxes, [box], 0, (0, 255, 0), 2)  # Green color for green bars

            cv2.circle(camera_view_with_boxes, (int(px), int(py)), 5, (0, 0, 255), -1)

            # Store the bounding box information
            bounding_boxes.append([px, py, wbb, hbb, theta_bb, bar_label])

        # Iterate over each detected purple contour
        for contour in purple_contours:
            # Get the minimum area bounding box (rotated rectangle)
            rect = cv2.minAreaRect(contour)
            (px, py), (wbb, hbb), theta_bb = rect

            # Calculate box points for visualization
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            # Label as purple and draw purple contours
            bar_label = 'vertical'
            cv2.drawContours(camera_view_with_boxes, [box], 0, (128, 0, 128), 2)  # Purple color for purple bars

            cv2.circle(camera_view_with_boxes, (int(px), int(py)), 5, (0, 0, 255), -1)

            # Store the bounding box information
            bounding_boxes.append([px, py, wbb, hbb, theta_bb, bar_label])


        cv2.imshow('RGB view', camera_view_with_boxes)
        cv2.imshow('Mask view', mask_view)
        cv2.waitKey(100)  # Wait for a key press to close the window

        return bounding_boxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Servoing Robot Simulation")
    parser.add_argument('--integrator', choices=['euler', 'runge-kutta'], default='euler', help='Choose the integration method')
    parser.add_argument('--controller', choices=['PD', 'LQR'], default='PD', help='Choose the controller type')

    args = parser.parse_args()

    # Initialize the controller and integrator based on the arguments
    controller = Controller(control_type=args.controller)
    integrator = Integrator(integration_type=args.integrator)

    robot = VisualServoingRobot(integrator, controller)
    robot.run(duration=6.0)