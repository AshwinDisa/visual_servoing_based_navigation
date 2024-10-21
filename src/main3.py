import argparse
import cv2
import numpy as np
from base_visual_servoing import BaseVisualServoing
from controller import Controller
from integrator import Integrator
from visualize import Visualizer
import yaml
import sys
import pdb

np.set_printoptions(precision=4, suppress=True)

def load_state_parameters(config_path):
    """Load the state parameters from a YAML or JSON configuration file."""
    try:
        with open(config_path, 'r') as file:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                state_parameters = yaml.safe_load(file)
            else:
                raise ValueError("Configuration file must be a .yaml or .json file.")
        return state_parameters
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)

class VisualServoingRobot(BaseVisualServoing):
    def __init__(self, integrator, controller, state_parameters, visualizer):
        super().__init__()

        self.pixel_per_meter = 300
        self.canvas_size = 3072
        self.camera_view_size = 300
        self.focal_length = 900
        self.depth = 3.0

        self.integrator = integrator
        self.controller = controller
        self.state_parameters = state_parameters
        self.visualizer = visualizer

        # Initialize PD gains for X, Y, and Z axes
        self.Kp_x = 0.001
        self.Kd_x = 0.005

        self.Kp_y = 0.02
        self.Kd_y = 0.01

        self.Kp_forward = 0.2
        self.Kd_forward = 0.5

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

    # def step(self, time_step: float):
    #     """
    #     Integrates the vehicle's dynamics over a specified time step.

    #     Parameters:
    #     time_step (float): The duration over which to integrate the vehicle dynamics.

    #     This function calculates the new state of the vehicle by performing a
    #     discrete time-step integration on the current state and control inputs.
    #     It updates the vehicle's state based on the computed dynamics at different
    #     points within the time step.

    #     Returns:
    #     None: The vehicle's state is updated in place.
    #     """

    #     # Compute the rate of change of the state
    #     state_dot = self.vehicle_dynamics(self.state, self.controls)

    #     # Euler integration
    #     self.state += state_dot * time_step

    def step(self, time_step: float):
        """
        Integrates the vehicle's dynamics using the Runge-Kutta 4th order method (RK4).

        Parameters:
        time_step (float): The duration over which to integrate the vehicle dynamics.

        This function calculates the new state of the vehicle by performing a
        discrete time-step integration using RK4 on the current state and control inputs.

        Returns:
        None: The vehicle's state is updated in place.
        """

        self.state = self.integrator.integrate(self.vehicle_dynamics, self.state, self.controls, time_step)

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

        # Convert the robot's (x, y) coordinates to pixel coordinates on the canvas using the projection formula
        # u = f * (X / Z), v = f * (Y / Z)
        # We are only using X (horizontal) and Z (vertical) here for simplicity.
        # x_px = int((x / Z_drone) * focal_length_px + self.canvas_size / 2)

        # # z is negative, assuming it moves the drone up in the image 
        # z_px = int((-z / Z_drone) * focal_length_px + self.canvas_size / 2)

        x_px = int((x / self.depth) * self.focal_length)

        # z is negative, assuming it moves the drone up in the image 
        z_px = int((-z / self.depth) * self.focal_length) + self.canvas_size

        # The robot is looking directly at the canvas, we crop a 300x300px area around the current position
        half_view_size = self.camera_view_size // 2

        # Ensure the crop window stays within canvas bounds
        top_left_x = max(0, x_px - half_view_size)
        top_left_y = max(0, z_px - half_view_size)
        bottom_right_x = min(self.canvas_size, top_left_x + self.camera_view_size)
        bottom_right_y = min(self.canvas_size, top_left_y + self.camera_view_size)

        # pdb.set_trace()

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

        # cv2.imshow('RGB view', camera_view_with_boxes)
        # cv2.imshow('Mask view', mask_view)
        # cv2.waitKey(30)  # Wait for a key press to close the window

        # print(bounding_boxes)

        return bounding_boxes, camera_view_with_boxes , mask_view

    def get_visual_servoing_inputs(self):
        """
        Implements the visual servoing algorithm using a PD controller
        to generate control inputs for the robot's navigation.

        Returns:
        np.ndarray: A numpy array containing the control inputs:
                    [thrust, roll_rate, desired_acceleration_y].
        """

        bounding_boxes, camera_view_with_boxes, mask_view = self.get_bounding_box()

        self.visualizer.visualization(self.state, camera_view_with_boxes, mask_view)

        if not bounding_boxes:
            return np.array([0, 0, 0])  # Return zero inputs if no bounding box is detected

        desired_px = self.camera_view_size / 2
        desired_py = self.camera_view_size / 2

        # Initialize the trajectory if it hasn't started
        if not self.traj_start:
            self.traj_start = True
            self.execute = 'climb_purple'
            print("----------------------")
            print("Climbing Purple bar")
            print("----------------------")

        if self.execute == 'Trajectory completed':
            return np.array([9.81, 0, 0])  # Hover in place when trajectory is completed

        error_px = error_py = error_forward = 0.0  # Default error values

        if self.execute in self.state_parameters:
            params = self.state_parameters[self.execute]

            # Find the bounding box with the matching bar_label
            matching_bounding_box = None
            for bounding_box in bounding_boxes:
                px, py, _, _, _, bar_label = bounding_box
                if bar_label == params['bar_label']:
                    matching_bounding_box = bounding_box
                    break

            if matching_bounding_box:
                px, py, _, _, _, bar_label = matching_bounding_box

                # Compute errors with offsets
                error_px = desired_px - px + params['error_offsets']['px']
                error_py = desired_py - py + params['error_offsets']['py']
                error_forward = 0.0

                # Count bounding boxes and update counters
                count_bounding_boxes = len(bounding_boxes)
                diff_count = count_bounding_boxes - self.prev_count_bounding_boxes

                if diff_count in params['count_difference_values']:
                    counter_name = params['counter']
                    setattr(self, counter_name, getattr(self, counter_name) + 1)
                    print(f"Detected {getattr(self, counter_name)}th bar")

                if getattr(self, params['counter']) > params['counter_threshold']:
                    self.execute = params['next_state']
                    setattr(self, params['counter'], 0)
                    self.prev_count_bounding_boxes = 0
                    print("----------------------")
                    print(params['message'])
                    print("----------------------")

                self.prev_count_bounding_boxes = count_bounding_boxes

        errors = [error_px, error_py, error_forward]

        # Compute control inputs using the PD controller
        thrust, roll_rate, y_ddot = self.controller.compute_control(errors, self.dt)

        return np.array([thrust, roll_rate, y_ddot])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visual Servoing Robot Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Controller and Integrator arguments
    parser.add_argument(
        '--integrator',
        choices=['euler', 'RK'],
        default='euler',
        help='Choose the integration method'
    )
    parser.add_argument(
        '--controller',
        choices=['PD', 'PID'],
        default='PD',
        help='Choose the controller type'
    )

    parser.add_argument(
        '--visualize',
        choices=['True', 'False'],
        default='True',
        help='Visualize the robot trajectory'
    )

    # Configuration file argument
    parser.add_argument(
        '--config',
        type=str,
        default='src/state_parameters.yaml',
        help='Path to the state parameters configuration file'
    )

    args = parser.parse_args()

    # Load state parameters
    state_parameters = load_state_parameters(args.config)

    # Initialize the controller and integrator based on the arguments
    controller = Controller(control_type=args.controller)
    integrator = Integrator(integration_type=args.integrator)
    visualizer = Visualizer(visualize=args.visualize)

    # Pass state_parameters to the robot
    robot = VisualServoingRobot(integrator, controller, state_parameters, visualizer)
    robot.run(duration=6.0)
