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
        
        # Compute the four slopes
        k1 = dynamics_func(state, control)
        k2 = dynamics_func(state + 0.5 * dt * k1, control)
        k3 = dynamics_func(state + 0.5 * dt * k2, control)
        k4 = dynamics_func(state + dt * k3, control)

        # Combine the four slopes to get the next state
        state += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)    

        return state
