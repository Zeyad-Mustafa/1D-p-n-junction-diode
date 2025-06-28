import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class DeviceParameters:
    """Parameters for the p-n junction device"""
    L: float = 1e-6              # Device length (1 micron)
    N: int = 200                 # Number of grid points
    NA: float = 1e24             # Acceptor concentration (1/m^3)
    ND: float = 1e22             # Donor concentration (1/m^3)
    junction_pos: float = 0.5    # Junction position as fraction of L

def create_mesh(device: DeviceParameters) -> Tuple[np.ndarray, float]:
    """Create the computational mesh"""
    x = np.linspace(0, device.L, device.N)
    dx = x[1] - x[0]
    return x, dx

def create_doping_profile(x: np.ndarray, device: DeviceParameters) -> np.ndarray:
    """Create doping profile with smooth transition at junction"""
    transition_width = 10 * (device.L / device.N)
    junction_x = device.L * device.junction_pos
    doping = -device.NA * 0.5 * (1 + np.erf((junction_x - x)/transition_width)) + \
              device.ND * 0.5 * (1 + np.erf((x - junction_x)/transition_width))
    return doping
