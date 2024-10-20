import numpy as np

# Combined Controller class with P, PD, and PID control
class Controller:
    def __init__(self, control_type='P'):
        self.control_type = control_type

        # P, PD, and PID Controller gains
        self.Kp_x = 0.002
        self.Kp_y = 0.01

        self.Kd_x = 0.0  # Derivative gain for PD/PID
        self.Kd_y = 0.1  # Derivative gain for PD/PID

        self.Ki_x = 0.001  # Integral gain for PID
        self.Ki_y = 0.005  # Integral gain for PID

        # Initialize integral terms for PID
        self.integral_error_x = 0.0
        self.integral_error_y = 0.0

    def compute_control(self, bounding_boxes, prev_errors, dt):
        """
        Compute control inputs based on visual input (bounding box errors) using P, PD, or PID control.

        Args:
        - bounding_boxes: List of bounding boxes detected by the robot.
        - prev_errors: Previous error values for derivative and integral control.
        - dt: Time step for control.

        Returns:
        - control: A numpy array of control inputs [thrust, roll_rate, y_ddot].
        - prev_errors: Updated previous errors for next iteration.
        """
        if not bounding_boxes:
            return np.array([0, 0, 0]), prev_errors  # No control input if no bounding boxes

        if self.control_type == 'P':
            return self._p_control(bounding_boxes), prev_errors
        elif self.control_type == 'PD':
            return self._pd_control(bounding_boxes, prev_errors, dt)
        elif self.control_type == 'PID':
            return self._pid_control(bounding_boxes, prev_errors, dt)

    def _p_control(self, bounding_boxes):
        """
        Proportional (P) control logic based on visual input.

        Args:
        - bounding_boxes: List of detected bounding boxes.

        Returns:
        - control: A numpy array of control inputs [thrust, roll_rate, y_ddot].
        """
        px, py, wbb, hbb, theta_bb, bar_label = bounding_boxes[0]

        # Desired bounding box center (image center)
        desired_px = 150  # Assuming camera view size is 300x300
        desired_py = 150

        # Error calculation in pixel space
        error_px = desired_px - px
        error_py = desired_py - py

        # P control outputs
        thrust = self.Kp_y * error_py + 9.81  # Adding gravity compensation
        roll_rate = -(self.Kp_x * error_px)
        y_ddot = -(self.Kp_y * error_py)

        return np.array([thrust, roll_rate, y_ddot])

    def _pd_control(self, bounding_boxes, prev_errors, dt):
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
        px, py, wbb, hbb, theta_bb, bar_label = bounding_boxes[0]

        # Desired bounding box center (image center)
        desired_px = 150  # Assuming camera view size is 300x300
        desired_py = 150

        # Error calculation in pixel space
        error_px = desired_px - px
        error_py = desired_py - py

        # Derivative of error
        derivative_px = (error_px - prev_errors[0]) / dt
        derivative_py = (error_py - prev_errors[1]) / dt

        # PD control outputs
        thrust = (self.Kp_y * error_py + self.Kd_y * derivative_py) + 9.81  # Adding gravity compensation
        roll_rate = -(self.Kp_x * error_px + self.Kd_x * derivative_px)
        y_ddot = -(self.Kp_y * error_py + self.Kd_y * derivative_py)

        # Update previous errors for the next iteration
        prev_errors = [error_px, error_py]

        return np.array([thrust, roll_rate, y_ddot]), prev_errors

    def _pid_control(self, bounding_boxes, prev_errors, dt):
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
        px, py, wbb, hbb, theta_bb, bar_label = bounding_boxes[0]

        # Desired bounding box center (image center)
        desired_px = 150  # Assuming camera view size is 300x300
        desired_py = 150

        # Error calculation in pixel space
        error_px = desired_px - px
        error_py = desired_py - py

        # Derivative of error
        derivative_px = (error_px - prev_errors[0]) / dt
        derivative_py = (error_py - prev_errors[1]) / dt

        # Update integral of error (accumulating over time)
        self.integral_error_x += error_px * dt
        self.integral_error_y += error_py * dt

        # PID control outputs
        thrust = (self.Kp_y * error_py + self.Kd_y * derivative_py + self.Ki_y * self.integral_error_y) + 9.81  # Adding gravity compensation
        roll_rate = -(self.Kp_x * error_px + self.Kd_x * derivative_px + self.Ki_x * self.integral_error_x)
        y_ddot = -(self.Kp_y * error_py + self.Kd_y * derivative_py + self.Ki_y * self.integral_error_y)

        # Update previous errors for the next iteration
        prev_errors = [error_px, error_py]

        return np.array([thrust, roll_rate, y_ddot]), prev_errors


# Combined Integrator class
class Integrator:
    def __init__(self, integration_type='euler'):
        self.integration_type = integration_type

    def integrate(self, dynamics_func, state, control, dt):
        if self.integration_type == 'euler':
            return self._euler_integration(dynamics_func, state, control, dt)
        elif self.integration_type == 'runge-kutta':
            return self._runge_kutta_integration(dynamics_func, state, control, dt)

    def _euler_integration(self, dynamics_func, state, control, dt):
        state_dot = dynamics_func(state, control)
        return state + state_dot * dt

    def _runge_kutta_integration(self, dynamics_func, state, control, dt):
        k1 = dynamics_func(state, control)
        k2 = dynamics_func(state + 0.5 * k1 * dt, control)
        k3 = dynamics_func(state + 0.5 * k2 * dt, control)
        k4 = dynamics_func(state + k3 * dt, control)
        return state + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6)