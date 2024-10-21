import cv2
import numpy as np
from pathlib import Path

class Visualizer():
    def __init__(self, visualize=True):

        self.trajectory = []  # To store the trajectory points
        self.canvas_size = 3072
        self.focal_length = 900
        self.depth = 3.0
        path = str(Path(__file__).parent.parent) + "/images/"
        self.canvas_image = cv2.imread(path + "canvas_single_tower.png")

        self.visualize = visualize

    def visualization(self, state, camera_view, mask_view):

        if self.visualize == 'False':
            return

        # Extract x and z from the state
        x = state[0]
        z = state[2]

        # Append the current coordinates to the trajectory
        self.trajectory.append((x, z))

        # Create a copy of the canvas to draw on
        canvas = self.canvas_image.copy()

        # Convert trajectory coordinates to pixel values using the provided scaling
        trajectory_pixels = [
            (
                int(x_point * self.focal_length / self.depth),
                int(-z_point * self.focal_length / self.depth + self.canvas_size)
            )
            for x_point, z_point in self.trajectory
        ]

        # Draw each point on the canvas
        for x_pixel, z_pixel in trajectory_pixels:
            cv2.circle(canvas, (x_pixel, z_pixel), radius=2, color=(0, 0, 255), thickness=7)

        # Resize camera_view and mask_view to 300x300
        camera_view_resized = cv2.resize(camera_view, (300, 300))
        mask_view_resized = cv2.resize(mask_view, (300, 300))

        # Create an empty combined canvas of size 600x900
        combined_canvas = np.zeros((600, 900, 3), dtype=np.uint8)

        # Resize canvas to 600x600 if it's not already
        canvas_resized = cv2.resize(canvas, (600, 600))

        # Place the canvas on the left
        combined_canvas[0:600, 0:600] = canvas_resized

        # Place the camera_view_resized on the top-right
        combined_canvas[0:300, 600:900] = camera_view_resized

        # Place the mask_view_resized on the bottom-right
        combined_canvas[300:600, 600:900] = mask_view_resized

        # Display the combined canvas
        cv2.imshow('Combined View', combined_canvas)
        cv2.waitKey(10)