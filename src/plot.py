import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
import pdb

FOCAL_LENGTH = 900.0
DEPTH = 3.0
HALF_CANVAS = 1536.0

def plot_trajectory_on_canvas(file_path):
    # Load the canvas image (using your existing path to the canvas)
    path = str(Path(__file__).parent.parent) + "/images/"
    canvas_image = cv2.imread(path + "canvas_single_tower.png")

    # Convert canvas_image from BGR (OpenCV default) to RGB for plotting with matplotlib
    canvas_image_rgb = cv2.cvtColor(canvas_image, cv2.COLOR_BGR2RGB)

    # Read the logged trajectory from the file
    trajectory = []
    with open(file_path, "r") as log_file:
        for line in log_file:
            x, z = map(float, line.strip().split(","))
            trajectory.append((x, z))

    # Convert the trajectory coordinates to pixel coordinates (based on your pixel-per-meter scale)
    pixel_per_meter = 300  # Scaling factor (pixels per meter)
    canvas_size = 3072     # Canvas size in pixels (10.24 meters side)
    center_pixel = canvas_size // 2  # The center of the canvas in pixels (1536, 1536)

    # trajectory_pixels = [
    #     (x * pixel_per_meter + center_pixel, -z * pixel_per_meter + center_pixel)
    #     for x, z in trajectory
    # ]

    trajectory_pixels = [(x*FOCAL_LENGTH/DEPTH, -z*FOCAL_LENGTH/DEPTH + HALF_CANVAS*2) for x, z in trajectory] 

    # Separate x and z for plotting
    x_pixels, z_pixels = zip(*trajectory_pixels)

    # Plot the trajectory on the canvas using matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(canvas_image_rgb)
    plt.plot(x_pixels, z_pixels, 'r-', linewidth=2, label='Trajectory')  # Red line for the trajectory
    plt.scatter(x_pixels, z_pixels, c='blue', s=10, label='Points')  # Plot trajectory points as blue dots

    # Set title and axis labels
    plt.title('Trajectory on Canvas')
    plt.xlabel('X (pixels)')
    plt.ylabel('Z (pixels)')
    
    # Add legend
    plt.legend()

    # Show the plot
    plt.show()

# Call the function with the path to the trajectory log file
plot_trajectory_on_canvas("src/trajectory_log.txt")
