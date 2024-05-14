import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# =============================================================================
# Problem parameters
# =============================================================================

# Material properties
materials = [
    {"SigmaA": 3.0, "SigmaS": 0.0},
    {"SigmaA": 0.05, "SigmaS": 0.02}
]

# Region card
reg_width = [5]  # size of region in cm
reg_N = [10]  # Number of cells per region
reg_mat = [0]  # Material index for each region
reg_S = [0]  # particles per cm^3 per region

# Method card
epsilon = 1E-7  # tolerance
phi_init = np.zeros(np.sum(reg_N))

# Angle card
quadrature_order = 4  # Gauss-Legendre order (aka number of angles to solve for. Use even-number)

# Define BCs

# "reflective", "vacuum"
left_bc = "vacuum"
right_bc = "vacuum"

# =============================================================================
# Preparation
# =============================================================================

# Angles and Weights
mus = np.polynomial.legendre.leggauss(quadrature_order)[0]
mus = mus[::-1]
weights = np.polynomial.legendre.leggauss(quadrature_order)[1]
print(f'mus = {mus}')
print(f'weights = {weights}')

# Number of regions
N_region = int(len(reg_width))
print(f'Number of Regions: {N_region}')
# Number of cell
N = int(sum(reg_N))
print(f'Number of cells: {N}')

# Allocate cell and cell-edge parameters
# ======================================

# psi[i][j] represents the angular flux at cell i and angle j
# where i is the index of the cell (from 0 to N-1)
# and j is the index of the angle (from 0 to quadrature_order - 1)

# Creating left, right, and center psi for each cell and angle combination
psi_minus_half = np.zeros((N, quadrature_order))
psi_plus_half = np.zeros((N, quadrature_order))
psi_i_m = np.zeros((N, quadrature_order))

# Creating left and right source for each cell and angle combination
S_left = np.zeros(N)
S_right = np.zeros(N)

# Initialize cross-sections and cell widths arrays
SigmaA_cells = np.zeros(N)
SigmaS_cells = np.zeros(N)
SigmaT_cells = np.zeros(N)
cell_widths = np.zeros(N)

# Calculate cross-sections and cell widths for each cell based on the assigned material
cell_idx = 0
for i, (width, num_cells, mat_idx) in enumerate(zip(reg_width, reg_N, reg_mat)):
    for j in range(num_cells):
        SigmaA_cells[cell_idx] = materials[mat_idx]["SigmaA"]
        SigmaS_cells[cell_idx] = materials[mat_idx]["SigmaS"]
        SigmaT_cells[cell_idx] = SigmaA_cells[cell_idx] + SigmaS_cells[cell_idx]
        cell_widths[cell_idx] = width / num_cells
        cell_idx += 1

# Grid Parameters
x_grid = np.zeros(N + 1)
x_grid[1:] = np.cumsum(cell_widths)
print('x_grid:', x_grid)

# Calculate midpoints of each cell
x_midpoints = (x_grid[:-1] + x_grid[1:]) / 2
print('x_midpoints:', x_midpoints)

# Assign source to each cell
for icell in range(N):
    # Determine the region index for the current cell
    region_idx = 0
    for i, num_cells in enumerate(reg_N):
        if icell < sum(reg_N[:i + 1]):
            region_idx = i
            break
    # Assign the source for the left and right sides of the cell based on its region
    S_left[icell] = reg_S[region_idx]
    S_right[icell] = reg_S[region_idx]

# =============================================================================
# Source Iteration
# =============================================================================
phi = phi_init[:]
converged = False
iterations = 0

while not converged:
    # store the old flux for convergence check
    phi_old = phi[:]

    # =====================================================================
    # Forward Sweep
    # =====================================================================

    # Set BCs
    if left_bc == "reflective":
        for m in range(int(quadrature_order / 2)):
            psi_minus_half[0][m] = psi_minus_half[0][quadrature_order - m - 1]
    psi_minus_half[0][0] = 10.0
    # Space sweep
    for icell in range(N):
        # Set isotropic source
        Q_L = 0.5 * (SigmaS_cells[icell] * phi[icell] + S_left[icell])
        Q_R = 0.5 * (SigmaS_cells[icell] * phi[icell] + S_right[icell])
        # Direction sweep
        for m in range(quadrature_order // 2):
            mu_val = mus[m]
            sigma_t = SigmaT_cells[icell]
            dx = cell_widths[icell]

            psi_L = (Q_L * dx ** 2 * sigma_t + Q_L * dx * mu_val - Q_R * dx * mu_val + 4 * dx * mu_val *
                     psi_minus_half[icell][m] * sigma_t + 6 * mu_val ** 2 * psi_minus_half[icell][m]) / (
                            dx ** 2 * sigma_t ** 2 + 4 * dx * mu_val * sigma_t + 6 * mu_val ** 2)
            psi_R = (3 * Q_L * dx * mu_val + Q_R * dx ** 2 * sigma_t + 3 * Q_R * dx * mu_val - 2 * dx * mu_val *
                     psi_minus_half[icell][m] * sigma_t + 6 * mu_val ** 2 * psi_minus_half[icell][m]) / (
                            dx ** 2 * sigma_t ** 2 + 4 * dx * mu_val * sigma_t + 6 * mu_val ** 2)
            # update psi
            if icell < N - 1:
                psi_minus_half[icell + 1][m] = psi_R

            # Update psi_plus_half for the final cell
            if icell == N - 1:
                psi_plus_half[icell][m] = psi_R
            psi_i_m[icell, m] = psi_L * (x_grid[icell + 1] - x_midpoints[icell]) / dx + psi_R * (
                    x_midpoints[icell] - x_grid[icell]) / dx  # Update psi_center

    # =====================================================================
    # Backward Sweep
    # =====================================================================

    # BC
    if right_bc == "reflective":
        for m in range(int(quadrature_order / 2), quadrature_order):
            psi_plus_half[-1][m] = psi_plus_half[-1][quadrature_order - m - 1]
    psi_plus_half[-1][-1] = 0.0
    # Space sweep
    for icell in range(N - 1, -1, -1):  # Reverse order
        # Set isotropic source (zero for now)
        Q_L = 0.5 * (SigmaS_cells[icell] * phi[icell] + S_left[icell])
        Q_R = 0.5 * (SigmaS_cells[icell] * phi[icell] + S_right[icell])
        # Direction sweep for negative mu values
        for m in range(quadrature_order // 2, quadrature_order):  # Second half of mus
            mu_val = mus[m]
            sigma_t = SigmaT_cells[icell]
            dx = cell_widths[icell]

            psi_L = (Q_L * dx ** 2 * sigma_t - 3 * Q_L * dx * mu_val - 3 * Q_R * dx * mu_val + 2 * dx * mu_val *
                     psi_plus_half[icell][m] * sigma_t + 6 * mu_val ** 2 * psi_plus_half[icell][m]) / (
                            dx ** 2 * sigma_t ** 2 - 4 * dx * mu_val * sigma_t + 6 * mu_val ** 2)
            psi_R = (Q_L * dx * mu_val + Q_R * dx ** 2 * sigma_t - Q_R * dx * mu_val - 4 * dx * mu_val *
                     psi_plus_half[icell][m] * sigma_t + 6 * mu_val ** 2 * psi_plus_half[icell][m]) / (
                            dx ** 2 * sigma_t ** 2 - 4 * dx * mu_val * sigma_t + 6 * mu_val ** 2)
            # Update psi
            if icell > 0:
                psi_plus_half[icell - 1][m] = psi_L
            psi_minus_half[icell][m] = psi_L
            psi_i_m[icell][m] = psi_L * (x_grid[icell + 1] - x_midpoints[icell]) / dx + \
                                psi_R * (x_midpoints[icell] - x_grid[icell]) / dx
    # =====================================================================
    # Update flux
    # =====================================================================

    # Update phi
    phi_new = np.zeros(N)  # initialize the new phi array

    for icell in range(N):
        # Sum up the contributions of all angles, weighted by their corresponding weights
        phi_new[icell] = np.dot(psi_i_m[icell, :], weights)

    # Check for convergence by ensuring all differences are within the tolerance
    if np.all(np.abs(phi_new - phi_old) < epsilon):
        converged = True

    phi[:] = phi_new  # update phi with the new values
    iterations += 1

print(f'total number of iterations: {iterations}')

# Calculate the current
J = np.zeros(N)

# Calculate the neutron current using centered values
for icell in range(N):
    J[icell] = np.dot(psi_i_m[icell, :], mus * weights)

# Plot the Scalar flux
plt.figure(figsize=(10, 6))
plt.plot(x_midpoints, phi, 'o-', label='Scalar Flux')
plt.plot(x_midpoints, J, 's-', label='Current')
plt.title('Scalar Flux and Current')
plt.xlabel('Position (cm)')
plt.ylabel('particles/cm^2-s')
plt.grid(True)
plt.legend()
plt.show()
