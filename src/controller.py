import numpy as np

class Controller:
    def __init__(self, control_type='PD'):
        self.control_type = control_type

        # PD Controller gains
        self.Kp_pd_x = 0.001
        self.Kd_pd_x = 0.005

        self.Kp_pd_y = 0.02
        self.Kd_pd_y = 0.002

        self.Kp_pd_forward = 0.2
        self.Kd_pd_forward = 0.5

        # PID Controller gains
        self.Kp_pid_x = 0.001
        self.Ki_pid_x = 0.0001
        self.Kd_pid_x = 0.005

        self.Kp_pid_y = 0.02
        self.Ki_pid_y = 0.0001
        self.Kd_pid_y = 0.002

        self.Kp_pid_forward = 0.2
        self.Ki_pid_forward = 0.01
        self.Kd_pid_forward = 0.5

        self.integral_px = 0.0
        self.integral_py = 0.0
        self.integral_error_forward = 0.0

        # Initialize errors for derivative terms
        self.prev_error_px = 0
        self.prev_error_py = 0
        self.prev_error_forward = 0

    def compute_control(self, errors, dt):

        """
        Compute control inputs based on visual input.
        """

        if self.control_type == 'PD':
            return self._pd_control(errors, dt)
        elif self.control_type == 'PID':
            return self._pid_control(errors, dt)


    def _pd_control(self, errors, dt):
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
        derivative_px = (errors[0] - self.prev_error_px) / dt
        derivative_py = (errors[1] - self.prev_error_py) / dt
        derivative_error_forward = (errors[2] - self.prev_error_forward) / dt

        # PD control for each axis
        thrust = (self.Kp_pd_y * errors[1] +
                    self.Kd_pd_y * derivative_py) + 9.81 # Thrust control (Z-axis)

        roll_rate = - (self.Kp_pd_x * errors[0] +
                       self.Kd_pd_x * derivative_px) # Roll control (X-axis)

        y_ddot = - (self.Kp_pd_forward * errors[2] +
                    self.Kd_pd_forward * derivative_error_forward)  # Vertical acceleration (Y-axis)
        
        # Update previous errors
        self.prev_error_px = errors[0]
        self.prev_error_py = errors[1]
        self.prev_error_forward = errors[2]

        return np.array([thrust, roll_rate, y_ddot])

    def _pid_control(self, errors, dt):
        """
        PID control logic based on visual input.

        Args:
        - errors: Current error values [error_x, error_y, error_forward].
        - dt: Time step for derivative and integral calculations.

        Returns:
        - control: A numpy array of control inputs [thrust, roll_rate, y_ddot].
        """

        # Compute derivative terms (rate of change of error)
        derivative_px = (errors[0] - self.prev_error_px) / dt
        derivative_py = (errors[1] - self.prev_error_py) / dt
        derivative_error_forward = (errors[2] - self.prev_error_forward) / dt

        # Update integral errors
        self.integral_px += errors[0] * dt
        self.integral_py += errors[1] * dt
        self.integral_error_forward += errors[2] * dt

        # PID control for each axis
        # Thrust control (Z-axis)
        thrust = (self.Kp_pid_y * errors[1] +
                self.Ki_pid_y * self.integral_py +
                self.Kd_pid_y * derivative_py) + 9.81

        # Roll control (X-axis)
        roll_rate = - (self.Kp_pid_x * errors[0] +
                    self.Ki_pid_x * self.integral_px +
                    self.Kd_pid_x * derivative_px)

        # Vertical acceleration (Y-axis)
        y_ddot = - (self.Kp_pid_forward * errors[2] +
                    self.Ki_pid_forward * self.integral_error_forward +
                    self.Kd_pid_forward * derivative_error_forward)
        
        thrust = (self.Kp_pid_y * errors[1] +
                  self.Ki_pid_y * self.integral_py +
                    self.Kd_pid_y * derivative_py) + 9.81 # Thrust control (Z-axis)

        roll_rate = - (self.Kp_pid_x * errors[0] +
                       self.Ki_pid_x * self.integral_px +
                       self.Kd_pid_x * derivative_px) # Roll control (X-axis)

        y_ddot = - (self.Kp_pd_forward * errors[2] +
                    self.Kd_pd_forward * derivative_error_forward)  # Vertical acceleration (Y-axis)
        
        # Update previous errors
        self.prev_error_px = errors[0]
        self.prev_error_py = errors[1]
        self.prev_error_forward = errors[2]

        return np.array([thrust, roll_rate, y_ddot])
    
    
