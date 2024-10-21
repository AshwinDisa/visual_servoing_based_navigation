import argparse
import cv2
import numpy as np
from pathlib import Path
import yaml
import sys

from base_visual_servoing import BaseVisualServoing
from controller_integrator import Controller, Integrator
from visualizer import Visualizer
from helpers import (
    get_robot_position,
    get_camera_pixel_coordinates,
    crop_view,
    detect_contours,
    draw_bounding_boxes,
    get_matching_bounding_box,
    update_trajectory_state
)

# Set numpy print options for cleaner output
np.set_printoptions(precision=4, suppress=True)


def load_state_parameters(config_path):
    """Load the state parameters from a YAML configuration file."""
    try:
        with open(config_path, 'r') as file:
            # Check if the file is YAML format
            if config_path.endswith(('.yaml', '.yml')):
                return yaml.safe_load(file)
            else:
                raise ValueError("Configuration file must be a .yaml or .yml file.")
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)


class VisualServoingRobot(BaseVisualServoing):
    """Class representing a visual servoing robot for trajectory-based navigation."""

    def __init__(self, integrator, controller, state_parameters, visualizer):
        super().__init__()

        # Initialize robot parameters such as pixel resolution, canvas size, etc.
        self._initialize_robot_parameters()

        # Set up integrator, controller, state parameters, and visualizer objects
        self.integrator = integrator
        self.controller = controller
        self.state_parameters = state_parameters
        self.visualizer = visualizer

        # Set simulation time step
        self.dt = 0.01

        # Flags and counters to track trajectory execution and progress
        self.traj_start = False
        self.execute = 'climb_purple'  # Initial state in the trajectory
        self.prev_count_bounding_boxes = -1
        self.green_bar_passed = 0
        self.purple_bar_passed = 0

    def _initialize_robot_parameters(self):
        """Initialize robot-specific parameters like size and view properties."""
        self.pixel_per_meter = 300  # Conversion factor for meters to pixels
        self.canvas_size = 3072  # Size of the canvas in pixels
        self.camera_view_size = 300  # Size of the camera view in pixels
        self.focal_length = 900  # Focal length for projection
        self.depth = 3.0  # Depth between robot and canvas

    def vehicle_dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        Computes the vehicle dynamics for a 4DOF robotic vehicle.
        
        Parameters:
        - state: The current state of the vehicle [x, y, z, theta, x_dot, y_dot, z_dot]
        - control: Control inputs [thrust, roll_rate, y_acceleration]
        
        Returns:
        - State derivatives (x_dot, y_dot, z_dot, theta_dot, x_ddot, y_ddot, z_ddot)
        """
        x, y, z, theta, x_dot, y_dot, z_dot = state
        thrust, roll_rate, y_ddot = control

        g = 9.81  # Gravitational constant in m/s^2
        d = 4     # Drag coefficient

        # Compute accelerations and angular velocity
        x_ddot = thrust * np.sin(theta) - d * x_dot
        y_ddot = y_ddot - d * y_dot
        z_ddot = thrust * np.cos(theta) - g - d * z_dot
        theta_dot = roll_rate

        return np.array([x_dot, y_dot, z_dot, theta_dot, x_ddot, y_ddot, z_ddot])

    def step(self, time_step: float):
        """
        Integrates the vehicle's dynamics using the selected integrator(euler/Runge-Kutta).
        
        The robot's state is updated based on the computed dynamics.
        """
        self.state = self.integrator.integrate(self.vehicle_dynamics, self.state, self.controls, time_step)

    def get_camera_view(self):
        """
        Generates a synthetic camera view based on the robot's current pose.
        
        Returns:
        - camera_view: The cropped camera view based on robot position
        - camera_mask: The cropped mask of the camera view
        """
        # Get robot's current x and z positions
        x, _, z = get_robot_position(self.state)

        # Convert robot's world coordinates to pixel coordinates for the camera
        x_px, y_px = get_camera_pixel_coordinates(x, z, self.depth, self.focal_length, self.canvas_size)

        # Crop the camera view and mask based on pixel coordinates
        camera_view, camera_mask = crop_view(x_px, y_px, self.camera_view_size, self.canvas_size, self.canvas_image, self.canvas_mask)

        return camera_view, camera_mask

    def get_bounding_box(self):
        """
        Extracts bounding boxes for detected structures from the segmentation mask.
        
        Returns:
        - bounding_boxes: A list of bounding boxes detected in the camera view
        - camera_view_with_boxes: The camera view with drawn bounding boxes
        - mask_view: The corresponding segmentation mask for the view
        """
        # Get the current camera view and mask
        camera_view, mask_view = self.get_camera_view()

        # Detect bounding boxes for green (horizontal) and purple (vertical) bars
        bounding_boxes = detect_contours(mask_view, [20, 120, 120], [255, 255, 255], 'horizontal') + \
                         detect_contours(mask_view, [120, 20, 120], [255, 255, 255], 'vertical')

        # Draw the detected bounding boxes on the camera view
        camera_view_with_boxes = draw_bounding_boxes(camera_view, bounding_boxes)

        return bounding_boxes, camera_view_with_boxes, mask_view

    def get_visual_servoing_inputs(self):
        """
        Implements the visual servoing algorithm using the controller of choice.
        
        This method computes control inputs based on the detected bounding boxes in the camera view.
        
        Returns:
        - Control inputs [thrust, roll_rate, y_ddot] to adjust the robot's movement.
        """
        # Get the bounding boxes and camera view with bounding boxes
        bounding_boxes, camera_view_with_boxes, mask_view = self.get_bounding_box()

        # Visualize the current state with bounding boxes and mask view
        self.visualizer.visualization(self.state, camera_view_with_boxes, mask_view)

        # If no bounding boxes are detected, return zero control inputs
        if not bounding_boxes:
            return np.array([0, 0, 0])

        # Calculate control inputs based on the detected bounding boxes
        return self._calculate_control_inputs(bounding_boxes)

    def _calculate_control_inputs(self, bounding_boxes):
        """
        Calculate control inputs using the controller of choice based on detected bounding boxes.
        
        Returns:
        - Thrust, roll rate, and y-axis acceleration for the robot.
        """
        # Compute the error between the desired and detected bounding box positions
        error_px, error_py, error_forward = self._compute_errors(bounding_boxes)

        # Use the controller to compute control inputs
        thrust, roll_rate, y_ddot = self.controller.compute_control([error_px, error_py, error_forward], self.dt)

        return np.array([thrust, roll_rate, y_ddot])

    def _compute_errors(self, bounding_boxes):
        """
        Compute position errors based on the detected bounding boxes and target position.
        
        Returns:
        - Errors in pixel positions and forward movement
        """
        # Desired pixel positions (center of the camera view)
        desired_px = desired_py = self.camera_view_size / 2

        error_px = error_py = error_forward = 0.0

        # Check if the current execution state is in the state parameters
        if self.execute in self.state_parameters:
            # Get parameters for the current state
            params = self.state_parameters[self.execute]

            # Find the matching bounding box for the current bar (purple/green)
            matching_box = get_matching_bounding_box(bounding_boxes, params['bar_label'])

            # If a matching box is found, calculate the position errors
            if matching_box:
                px, py, *_ = matching_box

                # Calculate errors in x and y (horizontal and vertical) with offsets
                error_px = desired_px - px + params['error_offsets']['px']
                error_py = desired_py - py + params['error_offsets']['py']

                # Update the trajectory state based on the bounding box detection
                update_trajectory_state(self, bounding_boxes, params)

        return error_px, error_py, error_forward


if __name__ == "__main__":
    # Set up argument parsing for the command-line interface
    parser = argparse.ArgumentParser(description="Visual Servoing Robot Simulation")

    # Add arguments for integrator, controller, and visualization options
    parser.add_argument('--integrator', choices=['euler', 'RK'], default='euler', help='Choose the integration method')
    parser.add_argument('--controller', choices=['PD', 'PID'], default='PD', help='Choose the controller type')
    parser.add_argument('--visualize', choices=['True', 'False'], default='True', help='Enable/disable visualization')

    # Define the path to the configuration file
    path = str(Path(__file__).parent.parent) + "/src/"
    parser.add_argument('--config', type=str, default=path + 'state_parameters.yaml', help='Path to config file')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load the state parameters from the configuration file
    state_parameters = load_state_parameters(args.config)

    # Initialize the controller, integrator, and visualizer based on the arguments
    controller = Controller(control_type=args.controller)
    integrator = Integrator(integration_type=args.integrator)
    visualizer = Visualizer(visualize=args.visualize)

    # Initialize and run the visual servoing robot with the loaded parameters
    robot = VisualServoingRobot(integrator, controller, state_parameters, visualizer)
    robot.run(duration=6.0)
