import cv2
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

np.set_printoptions(precision=4, suppress=True)


class BaseVisualServoing:
    def __init__(self):
        self.state = np.zeros(7)  # x, y, z, roll, x_dot, y_dot, z_dot
        self.controls = np.zeros(3)  # thrust, roll_rate, zoom_acc
        self.__noise_variance = 0 * np.diag(
            [0.01, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001]
        )
        self.integrator_time_step = 0.01

        path = str(Path(__file__).parent.parent) + "/images/"
        self.canvas_image = cv2.imread(path + "canvas_single_tower.png")
        self.canvas_mask = cv2.imread(path + "mask_single_tower.png")
        self.state[0] = 4.0
        self.state[1] = 1.5
        self.state[2] = 2

        #green
        # self.state[0] = -1
        # self.state[1] = 0.0
        # self.state[2] = -2.0

        # #purple
        # self.state[0] = -1
        # self.state[1] = 0.0
        # self.state[2] = -3.0

        # #hrishi
        # self.state[0] = -0.75
        # self.state[1] = 0.75
        # self.state[2] = -2.0

        # self.state[0] = -1.0
        # self.state[1] = 0.75
        # self.state[2] = -3

    def set_controls(self, controls: np.ndarray):
        self.controls = controls

    def __step(self):
        noise = self.__noise_variance @ (np.random.random(7) - 0.5)
        self.step(time_step=self.integrator_time_step)
        self.state += noise

    def run(self, duration: float):
        iterations = int(duration / self.integrator_time_step)

        data = []
        with open("src/trajectory_log.txt", "w") as f:  
            for i in range(iterations):
                control_input = self.get_visual_servoing_inputs()
                self.set_controls(control_input)
                self.__step()
                point = self.state.tolist()
                point.append(i * self.integrator_time_step)
                data.append(point)

                # log the coordinates
                x = self.state[0]
                z = self.state[2]
                f.write(f"{x}, {z}\n")

        data = np.array(data)
        fig = make_subplots(rows=3, cols=3)
        fig.add_trace(
            go.Scatter(x=data[:, 7], y=data[:, 0]),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=data[:, 7], y=data[:, 1]),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(x=data[:, 7], y=data[:, 2]),
            row=1,
            col=3,
        )
        fig.add_trace(
            go.Scatter(x=data[:, 0], y=data[:, 2]),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=data[:, 7], y=data[:, 3]),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Scatter(x=data[:, 7], y=data[:, 4]),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=data[:, 7], y=data[:, 5]),
            row=3,
            col=2,
        )
        fig.add_trace(
            go.Scatter(x=data[:, 7], y=data[:, 6]),
            row=3,
            col=3,
        )
        fig.update_xaxes(title_text="time (sec)")
        fig.update_xaxes(title_text="X (m)", range=[-0.5, 1.5], row=2, col=1)
        fig.update_yaxes(title_text="Z (m)", range=[-0.1, 5.0], row=2, col=1)
        fig.update_yaxes(title_text="X (m)", row=1, col=1)
        fig.update_yaxes(title_text="Y (m)", row=1, col=2)
        fig.update_yaxes(title_text="Z (m)", row=1, col=3)
        fig.update_yaxes(title_text="Roll (rad)", row=2, col=2)
        fig.update_yaxes(title_text="$\dot X$ (m/sec)", row=3, col=1)
        fig.update_yaxes(title_text="$\dot Y$ (m/sec)", row=3, col=2)
        fig.update_yaxes(title_text="$\dot Z$ (m/sec)", row=3, col=3)
        fig.show()

    def get_camera_view(self):
        raise NotImplementedError

    def get_bounding_box(self) -> tuple:
        raise NotImplementedError

    def vehicle_dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def step(self, time_step: float):
        raise NotImplementedError

    def get_visual_servoing_inputs(self) -> np.ndarray:
        raise NotImplementedError
