import numpy as np

class Controller:
    def __init__(self, control_type='PD'):
        """Initialize the controller with PD or PID control gains."""
        self.control_type = control_type

        # PD Controller gains for X (roll rate), Y (thrust), and Forward (y_ddot) directions
        self.Kp_pd_x = 0.001  # Proportional gain for roll rate (X-axis)
        self.Kd_pd_x = 0.005  # Derivative gain for roll rate (X-axis)

        self.Kp_pd_y = 0.02   # Proportional gain for thrust (Y-axis)
        self.Kd_pd_y = 0.002  # Derivative gain for thrust (Y-axis)

        self.Kp_pd_forward = 0.2  # Proportional gain for forward acceleration (Z-axis)
        self.Kd_pd_forward = 0.5  # Derivative gain for forward acceleration (Z-axis)

        # PID Controller gains for X, Y, and Forward directions
        self.Kp_pid_x = 0.001
        self.Ki_pid_x = 0.0001
        self.Kd_pid_x = 0.005

        self.Kp_pid_y = 0.02
        self.Ki_pid_y = 0.0001
        self.Kd_pid_y = 0.002

        self.Kp_pid_forward = 0.2
        self.Ki_pid_forward = 0.01
        self.Kd_pid_forward = 0.5

        # Integral terms for PID control (accumulation of errors over time)
        self.integral_px = 0.0
        self.integral_py = 0.0
        self.integral_error_forward = 0.0

        # Initialize previous errors for derivative control
        self.prev_error_px = 0
        self.prev_error_py = 0
        self.prev_error_forward = 0

    def compute_control(self, errors, dt):
        """
        Compute control inputs based on the errors (visual feedback) and time step.

        Args:
        - errors: Current error values [error_x, error_y, error_forward].
        - dt: Time step (delta time) for derivative calculations.

        Returns:
        - control: A numpy array containing [thrust, roll_rate, y_ddot] control inputs.
        """
        if self.control_type == 'PD':
            return self._pd_control(errors, dt)
        elif self.control_type == 'PID':
            return self._pid_control(errors, dt)

    def _pd_control(self, errors, dt):
        """
        PD control logic for handling proportional-derivative control.

        Args:
        - errors: Current error values [error_x, error_y, error_forward].
        - dt: Time step for derivative calculations.

        Returns:
        - control: A numpy array containing [thrust, roll_rate, y_ddot] control inputs.
        """

        # Compute the derivative (rate of change of error)
        derivative_px = (errors[0] - self.prev_error_px) / dt
        derivative_py = (errors[1] - self.prev_error_py) / dt
        derivative_error_forward = (errors[2] - self.prev_error_forward) / dt

        # PD control for thrust (Y-axis movement, controls altitude)
        thrust = (self.Kp_pd_y * errors[1] +
                  self.Kd_pd_y * derivative_py) + 9.81  # Add gravity compensation for thrust

        # PD control for roll rate (X-axis movement, controls lateral motion)
        roll_rate = - (self.Kp_pd_x * errors[0] +
                       self.Kd_pd_x * derivative_px)  # Negative sign for corrective action

        # PD control for forward acceleration (Z-axis movement, controls forward motion)
        y_ddot = - (self.Kp_pd_forward * errors[2] +
                    self.Kd_pd_forward * derivative_error_forward)

        # Update previous errors for next iteration
        self.prev_error_px = errors[0]
        self.prev_error_py = errors[1]
        self.prev_error_forward = errors[2]

        return np.array([thrust, roll_rate, y_ddot])

    def _pid_control(self, errors, dt):
        """
        PID control logic for handling proportional-integral-derivative control.

        Args:
        - errors: Current error values [error_x, error_y, error_forward].
        - dt: Time step for derivative and integral calculations.

        Returns:
        - control: A numpy array containing [thrust, roll_rate, y_ddot] control inputs.
        """

        # Compute the derivative (rate of change of error)
        derivative_px = (errors[0] - self.prev_error_px) / dt
        derivative_py = (errors[1] - self.prev_error_py) / dt
        derivative_error_forward = (errors[2] - self.prev_error_forward) / dt

        # Update integral (accumulated) errors for integral control
        self.integral_px += errors[0] * dt
        self.integral_py += errors[1] * dt
        self.integral_error_forward += errors[2] * dt

        # PID control for thrust (Y-axis, controls altitude)
        thrust = (self.Kp_pid_y * errors[1] +
                  self.Ki_pid_y * self.integral_py +
                  self.Kd_pid_y * derivative_py) + 9.81  # Add gravity compensation for thrust

        # PID control for roll rate (X-axis, controls lateral motion)
        roll_rate = - (self.Kp_pid_x * errors[0] +
                       self.Ki_pid_x * self.integral_px +
                       self.Kd_pid_x * derivative_px)

        # PID control for forward acceleration (Z-axis, controls forward motion)
        y_ddot = - (self.Kp_pid_forward * errors[2] +
                    self.Ki_pid_forward * self.integral_error_forward +
                    self.Kd_pid_forward * derivative_error_forward)

        # Update previous errors for next iteration
        self.prev_error_px = errors[0]
        self.prev_error_py = errors[1]
        self.prev_error_forward = errors[2]

        return np.array([thrust, roll_rate, y_ddot])


class Integrator:
    def __init__(self, integration_type='euler'):
        """Initialize the integrator with the specified integration method."""
        self.integration_type = integration_type

    def integrate(self, dynamics_func, state, control, dt):
        """
        Integrates the system dynamics using the specified integration method.

        Args:
        - dynamics_func: Function that calculates the system's dynamics (rate of change).
        - state: Current state of the system.
        - control: Control inputs for the system.
        - dt: Time step for integration.

        Returns:
        - Updated state after integration.
        """
        if self.integration_type == 'euler':
            return self._euler_integration(dynamics_func, state, control, dt)
        elif self.integration_type == 'RK':
            return self._runge_kutta_integration(dynamics_func, state, control, dt)

    def _euler_integration(self, dynamics_func, state, control, dt):
        """
        Perform Euler integration to update the system's state.

        Args:
        - dynamics_func: Function that calculates the system's dynamics (rate of change).
        - state: Current state of the system.
        - control: Control inputs for the system.
        - dt: Time step for integration.

        Returns:
        - Updated state after Euler integration.
        """
        state_dot = dynamics_func(state, control)  # Compute the rate of change (dynamics)
        return state + state_dot * dt  # Update the state using Euler's method

    def _runge_kutta_integration(self, dynamics_func, state, control, dt):
        """
        Perform 4th order Runge-Kutta (RK4) integration to update the system's state.

        Args:
        - dynamics_func: Function that calculates the system's dynamics (rate of change).
        - state: Current state of the system.
        - control: Control inputs for the system.
        - dt: Time step for integration.

        Returns:
        - Updated state after RK4 integration.
        """
        # Compute the four RK4 slopes (k1, k2, k3, k4)
        k1 = dynamics_func(state, control)
        k2 = dynamics_func(state + 0.5 * dt * k1, control)
        k3 = dynamics_func(state + 0.5 * dt * k2, control)
        k4 = dynamics_func(state + dt * k3, control)

        # Combine the slopes to update the state
        state += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return state