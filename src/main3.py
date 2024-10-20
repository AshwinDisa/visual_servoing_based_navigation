import argparse
import cv2
import numpy as np
from base_visual_servoing import BaseVisualServoing
from controllers_integrators import Controller, Integrator  # Import from the new module

np.set_printoptions(precision=4, suppress=True)


class VisualServoingRobot(BaseVisualServoing):
    def __init__(self):
        super().__init__()

        self.pixel_per_meter = 300
        self.canvas_size = 3072
        self.camera_view_size = 300
        self.focal_length = 3.2

        # Initialize PD gains for X, Y, and Z axes
        self.Kp_x = 0.001
        self.Kd_x = 0.005

        self.Kp_y = 0.01
        self.Kd_y = 0.01

        self.Kp_forward = 0.2
        self.Kd_forward = 0.5

        # Initialize errors for derivative terms
        self.prev_error_px = -31
        self.prev_error_py = 78
        self.prev_error_forward = 0

        # Fixed time step
        self.dt = 0.01

        self.traj_start = False
        self.execute = 'climb_purple'
        self.prev_count_bounding_boxes = -1
        self.green_bar_passed = 0
        self.purple_bar_passed = 0

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

        # print(x, y, z)

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
        The resulting cropped image is expected to simulate the camera perspective, accounting for focal length.

        Returns:
        tuple: (camera_view, camera_mask) where both are 300x300 pixel images.
        """

        # Extract the robot's position in world coordinates (X, Y, Z)
        x, y, z, _, _, _, _ = self.state

        # Fixed distance between the drone and the canvas (drone is 2 meters away)
        # Z_drone = 2.0  # The depth of the drone from the canvas

        Z_drone = 2

        # The focal length of the camera in pixels (provided as 900 pixels)
        focal_length_px = 900

        # Convert the robot's (x, y) coordinates to pixel coordinates on the canvas using the projection formula
        # u = f * (X / Z), v = f * (Y / Z)
        # We are only using X (horizontal) and Z (vertical) here for simplicity.
        x_px = int((x / Z_drone) * focal_length_px + self.canvas_size / 2)

        # z is negative, assuming it moves the drone up in the image 
        z_px = int((-z / Z_drone) * focal_length_px + self.canvas_size / 2)

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
        
        # Desired bounding box center position (center of camera view)
        desired_px = self.camera_view_size / 2
        desired_py = self.camera_view_size / 2

        # Extract bounding box properties
        # px, py, wbb, hbb, theta_bb, bar_label = bounding_boxes[0]

        for bounding_box in bounding_boxes:

            px, py, wbb, hbb, theta_bb, bar_label = bounding_box

            if self.traj_start == False:

                self.traj_start = True
                self.execute = 'climb_purple'
                print("----------------------")
                print("Climbing Purple bar")
                print("----------------------")

            if self.execute == 'climb_purple' and bar_label == 'vertical':

                error_px = desired_px - px - 100
                error_py = desired_py - py + 300
                error_forward = 0.0

                count_bounding_boxes = len(bounding_boxes)

                if count_bounding_boxes - self.prev_count_bounding_boxes == 1:

                    print(f"Detected {self.green_bar_passed+1}th green bar")
                    self.green_bar_passed += 1

                self.prev_count_bounding_boxes = count_bounding_boxes

                if self.green_bar_passed > 2:
                        
                    self.execute = 'travel_green'
                    self.prev_count_bounding_boxes = 0
                    print("----------------------")
                    print("Onto Green bar")
                    print("----------------------")

            elif self.execute == 'travel_green' and bar_label == 'horizontal':

                error_px = desired_px - px - 100
                error_py = desired_py - py + 30
                error_forward = 0.0

                count_bounding_boxes = len(bounding_boxes)

                if count_bounding_boxes - self.prev_count_bounding_boxes == -1 or count_bounding_boxes - self.prev_count_bounding_boxes == 1:
                    
                    print(f"Detected {self.purple_bar_passed+1}th purple bar")
                    self.purple_bar_passed += 1 

                if self.purple_bar_passed > 1:

                    self.execute = 'descend_purple'
                    print("----------------------")
                    print("Descending Purple bar")
                    print("----------------------")

                self.prev_count_bounding_boxes = count_bounding_boxes

            elif self.execute == 'descend_purple' and bar_label == 'vertical':

                error_px = desired_px - px + 100
                error_py = desired_py - py - 300
                error_forward = 0.0
    

        # Compute derivative terms (rate of change of error)
        derivative_px = (error_px - self.prev_error_px) / self.dt
        derivative_py = (error_py - self.prev_error_py) / self.dt
        derivative_forward = (error_forward - self.prev_error_forward) / self.dt

        # PD control for each axis
        thrust = (self.Kp_y * error_py +
                    self.Kd_y * error_py) + 9.81 # Thrust control (Z-axis)

        roll_rate = - (self.Kp_x * error_px +
                       self.Kd_x * derivative_px) # Roll control (X-axis)

        y_ddot = - (self.Kp_y * error_py +
                    self.Kd_y * derivative_py)  # Vertical acceleration (Y-axis)

        # Save current errors for the next iteration
        self.prev_error_px = error_px
        self.prev_error_py = error_py
        self.prev_error_forward = error_forward

        # print(f"Errors: [X: {error_px:.2f}, Y: {error_py:.2f}, Z: {error_forward:.2f}]")
        # print(f"Control inputs: [thrust: {thrust:.2f}, roll_rate: {roll_rate:.2f}, y_ddot: {y_ddot:.2f}]")

        # print(f"{roll_rate:.2f}, error: {error_px:.2f}")

        return np.array([thrust, roll_rate, 0])

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