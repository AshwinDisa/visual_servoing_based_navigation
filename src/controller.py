import numpy as np

class Controller:
    def __init__(self, control_type='PD'):
        self.control_type = control_type

        # PD Controller gains
        self.Kp_x = 0.001
        self.Kd_x = 0.005

        self.Kp_y = 0.02
        self.Kd_y = 0.01

        self.Kp_forward = 0.2
        self.Kd_forward = 0.5

    def compute_control(self, errors, prev_errors, dt):

        """
        Compute control inputs based on visual input.
        """

        if self.control_type == 'PD':
            return self._pd_control(errors, prev_errors, dt)
        elif self.control_type == 'PID':
            return self._pid_control(errors, prev_errors, dt)


    def _pd_control(self, errors, prev_errors, dt):
        """
        PD control logic based on visual input.

        Args:
        - bounding_boxes: List of detected bounding boxes.
        - prev_errors: Previous error values.
        - dt: Time step for derivative calculation.

        Returns:
        - control: A numpy array of control inputs [thrust, roll_rate, y_ddot].
        - prev_errors: Updated previous errors for next iteration.
        """

        # Compute derivative terms (rate of change of error)
        derivative_px = (errors[0] - prev_errors[0]) / dt
        derivative_py = (errors[1] - prev_errors[1]) / dt
        derivative_error_forward = (errors[2] - prev_errors[2]) / dt

        # PD control for each axis
        thrust = (self.Kp_y * errors[1] +
                    self.Kd_y * errors[1]) + 9.81 # Thrust control (Z-axis)

        roll_rate = - (self.Kp_x * errors[0] +
                       self.Kd_x * derivative_px) # Roll control (X-axis)

        y_ddot = - (self.Kp_y * errors[2] +
                    self.Kd_y * derivative_error_forward)  # Vertical acceleration (Y-axis)

        return np.array([thrust, roll_rate, y_ddot])

    def _pid_control(self, errors, prev_errors, dt):
        """
        PID control logic based on visual input.

        Args:
        - bounding_boxes: List of detected bounding boxes.
        - prev_errors: Previous error values.
        - dt: Time step for derivative and integral calculation.

        Returns:
        - control: A numpy array of control inputs [thrust, roll_rate, y_ddot].
        - prev_errors: Updated previous errors for next iteration.
        """

