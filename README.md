# Software Robotics Spring Co-op 2025 Assignment Visual Servoing

### Additional dependencies

```bash
- argparse
- PyYAML
```

### File Structure

```bash
├── src/
│   ├── base_visual_servoing.py
│   ├── controller_integrator.py
│   ├── helpers.py
│   ├── main.py
│   ├── state_parameters.yaml
│   ├── visualizer.py
└── README.md
```

## Usage

You can run the simulation by executing the `main.py` file with various command-line arguments.

### Command-line Options

The following command-line arguments are supported:

- `--integrator`: Choose the integration method for the simulation. Options are:
  - `euler`: Use the Euler integration method (default).
  - `RK`: Use the Runge-Kutta integration method.

- `--controller`: Choose the controller type for the simulation. Options are:
  - `PD`: Proportional-Derivative controller (default).
  - `PID`: Proportional-Integral-Derivative controller.

- `--visualize`: Enable or disable visualization of the simulation. Options are:
  - `True`: Enable visualization (default).
  - `False`: Disable visualization.

- `--config`: Specify the path to the configuration file containing the state parameters (default: `src/state_parameters.yaml`).

### Example

```
python3 main.py --controller PID --integrator RK --visualize True

To list all options
python3 main.py --help
```

### Trajectory Logic for Visual Servoing

The trajectory logic is insprired from **State Machine** designed to ensure that the drone can traverse the tower rack by detecting visual features and adjusting its path accordingly. The flexibility of the approach lies in the use of a YAML file for configuration, which allows the system to adapt to similar but different rack structures with minimal modifications to the file and no changes to the main code. 

The YAML file defines the different states of the drone's movement and specifies parameters like error offsets, counters for tracking passed bars, and state transitions. By adjusting the values in this file, you can easily adapt the drone's behavior to different warehouse racks, minimizing the need for hard-coding.

