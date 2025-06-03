import numpy as np
import matplotlib.pyplot as plt

# Constants
q = 1.602e-19         # Elementary charge (C)
k = 1.38e-23          # Boltzmann constant (J/K)
T = 300               # Temperature (K)
epsilon_0 = 8.85e-12  # Vacuum permittivity (F/m)
epsilon_si = 11.7     # Relative permittivity for silicon
ni = 1.5e16           # Intrinsic carrier concentration (1/m^3)

# Device parameters
L = 1e-6              # Device length (1 micron)
N = 200               # Number of grid points
x = np.linspace(0, L, N)
dx = x[1] - x[0]

# Doping profile
NA = 1e24             # Acceptor concentration (1/m^3)
ND = 1e22             # Donor concentration (1/m^3)
doping = np.where(x < L/2, -NA, ND)  # p-side negative, n-side positive

# Initial potential guess (linear)
phi = np.linspace(0, 0.7, N)

# Poisson solver (finite difference, simple iteration)
def solve_poisson(phi, doping, tol=1e-6, max_iter=10000):
    for it in range(max_iter):
        n = ni * np.exp(q * phi / (k * T))
        p = ni * np.exp(-q * phi / (k * T))
        rho = q * (doping + p - n)
        phi_new = phi.copy()
        for i in range(1, N-1):
            phi_new[i] = 0.5 * (phi[i-1] + phi[i+1] - dx**2 * rho[i] / (epsilon_si * epsilon_0))
        if np.max(np.abs(phi_new - phi)) < tol:
            break
        phi = phi_new
    return phi

phi = solve_poisson(phi, doping)

# Calculate carrier concentrations
n = ni * np.exp(q * phi / (k * T))
p = ni * np.exp(-q * phi / (k * T))

# Plot results
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(x*1e6, phi)
plt.title('Electrostatic Potential across the p-n Junction')
plt.ylabel('Potential (V)')
plt.subplot(2,1,2)
plt.plot(x*1e6, n, label='Electrons (n)')
plt.plot(x*1e6, p, label='Holes (p)')
plt.yscale('log')
plt.ylabel('Carrier concentration (1/m³)')
plt.xlabel('Position (micron)')
plt.legend()
plt.tight_layout()
plt.show()

# I-V characteristics (Shockley equation)
def diode_current(V, Is=1e-12):
    return Is * (np.exp(q*V/(k*T)) - 1)

V = np.linspace(-0.5, 0.8, 100)
I = diode_current(V)

plt.figure(figsize=(8,5))
plt.plot(V, I)
plt.title('Diode I-V Characteristic (Shockley Equation)')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.grid(True)
plt.show()
