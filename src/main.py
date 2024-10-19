import numpy as np
import cv2

from base_visual_servoing import BaseVisualServoing

np.set_printoptions(precision=4, suppress=True)


class VisualServoingRobot(BaseVisualServoing):
    def __init__(self):
        super().__init__()

        self.pixel_per_meter = 300
        self.canvas_size = 3072
        self.camera_view_size = 300
        self.focal_length = 3.2
        self.canvas_image = cv2.imread("images/canvas_single_tower.png")
        self.canvas_mask = cv2.imread("images/mask_single_tower.png")

    def vehicle_dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        Computes the dynamics of a 4DOF robotic vehicle based on its current state and control inputs.

        Parameters:
        state (np.ndarray): A numpy array representing the current state of the vehicle,
                           where the elements correspond to [x, y, z, theta, x_dot, y_dot, z_dot].
        control (np.ndarray): A numpy array representing the control inputs,
                              where the elements correspond to [thrust, theta_rate, y_acceleration].

        Returns:
        np.ndarray: A numpy array representing the rate of change of the state,
                    including position, roll, and velocities.
        """

        # Unpack states and control inputs
        x, y, z, theta, x_dot, y_dot, z_dot = state
        thrust, roll_rate, y_ddot = control

        # Constants
        g = 9.81    # Gravity in m/s^2  
        d = 4   # Drag coefficient

        # Dynamics
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

    def step(self, time_step: float):
        """
        Integrates the vehicle's dynamics over a specified time step.

        Parameters:
        time_step (float): The duration over which to integrate the vehicle dynamics.

        This function calculates the new state of the vehicle by performing a
        discrete time-step integration on the current state and control inputs.
        It updates the vehicle's state based on the computed dynamics at different
        points within the time step.

        Returns:
        None: The vehicle's state is updated in place.
        """

        # Compute the rate of change of the state
        state_dot = self.vehicle_dynamics(self.state, self.controls)

        # Euler integration
        self.state += state_dot * time_step

    def get_camera_view(self):
        """
        Generates a synthetic camera view based on the robot's current pose and the canvas image.

        This function crops an image from the larger canvas according to the robot's state.
        The resulting cropped image is expected to simulate the camera perspective

        Returns:
        None: The camera view image and mask are updated in place.
        """

        x, y, z, _, _, _, _ = self.state

        # The canvas is at 2.75m from the world frame, and the drone is initialized at (4, 1.5, 2)
        # Hence, the distance between the drone and the canvas is 2.75 - 1.5 = 1.25 meters
        # canvas_distance = 2.75 - 1.5  # corrected distance in meters
        canvas_distance = 2  # corrected distance in meters

        # Convert the robot's (x, y, z) coordinates from meters to pixels using the scaling factor
        x_px = int((x / canvas_distance) * self.pixels_per_meter + self.canvas_size / 2)
        z_px = int((z / canvas_distance) * self.pixels_per_meter + self.canvas_size / 2)
        
        # The robot is looking directly at the canvas, we crop a 300x300px area around the current position
        half_view_size = self.camera_view_size // 2
        
        # Ensure the crop window stays within canvas bounds
        top_left_x = max(0, x_px - half_view_size)
        top_left_y = max(0, z_px - half_view_size)
        bottom_right_x = min(self.canvas_size, top_left_x + self.camera_view_size)
        bottom_right_y = min(self.canvas_size, top_left_y + self.camera_view_size)

        # Crop the camera view from the canvas image and mask
        camera_view = self.canvas_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        camera_mask = self.canvas_mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    def get_bounding_box(self):
        """
        Extracts bounding boxes around detected structures in the camera image using the segmentation mask.

        Returns:
        A list of bounding box parameters for different structures in camera image
        """

        # Define color thresholds for green and purple bars
        lower_green = np.array([20, 120, 120])
        upper_green = np.array([255, 255, 255])

        lower_purple = np.array([120, 20, 120])
        upper_purple = np.array([255, 255, 255])

        # Get the camera view and mask view (this is a placeholder, replace with actual function)
        camera_view, mask_view = self.get_camera_view(self.state)  # Ensure get_camera_view() is implemented

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

    def get_visual_servoing_inputs(self):
        """
        Implements the visual servoing algorithm for navigation and control of the robot.

        Returns:
        np.ndarray: A numpy array containing the control inputs:
                    [thrust, roll_rate, desired_acceleration_y].
        """


if __name__ == "__main__":
    robot = VisualServoingRobot()
    robot.run(duration=6.0)
