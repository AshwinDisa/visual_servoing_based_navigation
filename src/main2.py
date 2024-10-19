import numpy as np
import cv2
import pdb

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

        # Initialize PD gains for X, Y, and Z axes
        self.Kp_x = 0.01
        self.Kd_x = 0.01

        self.Kp_y = 0.005
        self.Kd_y = 0.01

        self.Kp_depth = 10
        self.Kd_depth = 0.5

        # Initialize errors for derivative terms
        self.prev_error_px = 0
        self.prev_error_py = 0
        self.prev_error_depth = 0

        # Fixed time step
        self.dt = 0.01

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
        x_px = int(x * self.pixel_per_meter + self.canvas_size / 2)
        z_px = int(-z * self.pixel_per_meter + self.canvas_size / 2)

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

        return camera_view, camera_mask

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
        camera_view, mask_view = self.get_camera_view()  # Ensure get_camera_view() is implemented

        # Apply color thresholding to get binary masks
        mask_green = cv2.inRange(mask_view, lower_green, upper_green)
        mask_purple = cv2.inRange(mask_view, lower_purple, upper_purple)

        # Detect contours from the green and purple masks
        green_contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        purple_contours, _ = cv2.findContours(mask_purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []

        camera_view_with_boxes = camera_view.copy()

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


        cv2.imshow('RGB view', camera_view_with_boxes)
        cv2.imshow('Mask view', mask_view)
        cv2.waitKey(100)  # Wait for a key press to close the window

        # print(bounding_boxes)

        return bounding_boxes

    # def get_visual_servoing_inputs(self):
    #     """
    #     Implements the visual servoing algorithm to generate control inputs for the robot's navigation.

    #     Returns:
    #     np.ndarray: A numpy array containing the control inputs:
    #                 [thrust, roll_rate, desired_acceleration_y].
    #     """
    #     bounding_boxes = self.get_bounding_box()

    #     if not bounding_boxes:
    #         return np.array([0, 0, 0])  # Return zero inputs if no bounding box is detected

    #     # Extract bounding box properties
    #     px, py, wbb, hbb, theta_bb, bar_label = bounding_boxes[0]

    #     # Desired bounding box center position (center of camera view)
    #     desired_px = self.camera_view_size / 2
    #     desired_py = self.camera_view_size / 2

    #     # Compute errors in X, Y, and Z (depth) axes
    #     error_px = desired_px - px
    #     error_py = desired_py - py
    #     error_depth = self.state[2]  # Current depth (Z-axis)

    #     # print(f"Bounding box error: [X: {error_px:.2f}, Y: {error_py:.2f}, Z: {error_depth:.2f}]")

    #     print(px, py)

    #     # Define proportional control gains
    #     Kp_x = 0.05 # Gain for X-axis (roll)
    #     Kp_y = 0.005  # Gain for Y-axis (vertical)
    #     Kp_depth = 3  # Gain for Z-axis (depth)

    #     # Compute control inputs
    #     thrust = - Kp_depth * error_depth  # Thrust control (Z-axis)
    #     roll_rate = - Kp_x * error_px  # Roll control (X-axis)
    #     y_ddot = - Kp_y * error_py  # Vertical acceleration (Y-axis)

    #     return np.array([thrust, roll_rate, y_ddot])

    def get_visual_servoing_inputs(self):
        """
        Implements the visual servoing algorithm using a PD controller
        to generate control inputs for the robot's navigation.

        Returns:
        np.ndarray: A numpy array containing the control inputs:
                    [thrust, roll_rate, desired_acceleration_y].
        """
        bounding_boxes = self.get_bounding_box()

        if not bounding_boxes:
            return np.array([0, 0, 0])  # Return zero inputs if no bounding box is detected

        # Extract bounding box properties
        px, py, wbb, hbb, theta_bb, bar_label = bounding_boxes[0]

        # Desired bounding box center position (center of camera view)
        desired_px = self.camera_view_size / 2
        desired_py = self.camera_view_size / 2

        # Compute errors in X, Y, and Z (depth) axes
        error_px = desired_px - px
        error_py = desired_py - py
        error_depth = 0.0

        # Compute derivative terms (rate of change of error)
        derivative_px = (error_px - self.prev_error_px) / self.dt
        derivative_py = (error_py - self.prev_error_py) / self.dt
        derivative_depth = (error_depth - self.prev_error_depth) / self.dt

        # PD control for each axis
        thrust = - (self.Kp_depth * error_depth +
                    self.Kd_depth * derivative_depth)  # Thrust control (Z-axis)

        roll_rate = - (self.Kp_x * error_px +
                       self.Kd_x * derivative_px)  # Roll control (X-axis)

        y_ddot = - (self.Kp_y * error_py +
                    self.Kd_y * derivative_py)  # Vertical acceleration (Y-axis)

        # Save current errors for the next iteration
        self.prev_error_px = error_px
        self.prev_error_py = error_py
        self.prev_error_depth = error_depth

        # print(f"Errors: [X: {error_px:.2f}, Y: {error_py:.2f}, Z: {error_depth:.2f}]")
        print(f"Controls: [Thrust: {thrust:.2f}, Roll Rate: {roll_rate:.2f}, Y Acceleration: {y_ddot:.2f}]")

        return np.array([9.81, roll_rate, y_ddot])


if __name__ == "__main__":
    robot = VisualServoingRobot()
    robot.run(duration=6.0)