"""
Molecular-plasmonic coupling calculations using FFT-based Coulomb solver.

This module implements the calculation of electromagnetic coupling between
molecular transition densities and plasmonic excitations using Fast Fourier
Transform methods to solve the Poisson equation.
"""

import numpy as np
import numpy.fft as fft
import warnings
from ..io.cube_io import parse_cube_file, density_3d_from_flat
from ..utils.constants import E_CHARGE, BOHR_TO_M, ANG_TO_M, EPS0


# Global verbose flag (can be set by CLI)
VERBOSE = False


def set_verbose(value):
    """Set global verbose mode."""
    global VERBOSE
    VERBOSE = value


def get_density_bounding_box(rho_3d, threshold=1e-6):
    """
    Find minimal bounding box containing density above threshold.

    Parameters
    ----------
    rho_3d : np.ndarray, shape (NX, NY, NZ)
        3D density array.
    threshold : float, optional
        Density threshold. Default: 1e-6.

    Returns
    -------
    tuple of 3 slice objects
        Indices for X, Y, Z that crop to the bounding box.
    """
    mask = np.abs(rho_3d) > threshold

    # Find non-zero indices along each axis
    x_indices = np.where(mask.any(axis=(1, 2)))[0]
    y_indices = np.where(mask.any(axis=(0, 2)))[0]
    z_indices = np.where(mask.any(axis=(0, 1)))[0]

    if len(x_indices) == 0 or len(y_indices) == 0 or len(z_indices) == 0:
        # Empty density, return full grid
        return (slice(None), slice(None), slice(None))

    x_min, x_max = x_indices[0], x_indices[-1] + 1
    y_min, y_max = y_indices[0], y_indices[-1] + 1
    z_min, z_max = z_indices[0], z_indices[-1] + 1

    return (slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))


def crop_density_to_union_bbox(rho_A, rho_B, threshold, margin_phys, steps):
    """
    Crop two densities to their union bounding box + physical margin.

    Parameters
    ----------
    rho_A, rho_B : np.ndarray, shape (NX, NY, NZ)
        3D density arrays (in same units).
    threshold : float
        Density threshold in native units (e/bohr³ or e/Å³).
    margin_phys : float
        Safety margin in native units (bohr or Å).
    steps : tuple (dx, dy, dz)
        Grid spacing in same units as margin_phys.

    Returns
    -------
    rho_A_crop : np.ndarray
        Cropped density A.
    rho_B_crop : np.ndarray
        Cropped density B.
    crop_slices : tuple of slices
        The slices used for cropping.
    """
    # Get individual bboxes
    bbox_A = get_density_bounding_box(rho_A, threshold)
    bbox_B = get_density_bounding_box(rho_B, threshold)

    NX, NY, NZ = rho_A.shape
    dx, dy, dz = steps

    # Convert physical margin to cells
    margin_cells_x = int(np.ceil(margin_phys / dx))
    margin_cells_y = int(np.ceil(margin_phys / dy))
    margin_cells_z = int(np.ceil(margin_phys / dz))

    # Union of bboxes with physical margin
    x_min = max(0, min(bbox_A[0].start, bbox_B[0].start) - margin_cells_x)
    x_max = min(NX, max(bbox_A[0].stop, bbox_B[0].stop) + margin_cells_x)
    y_min = max(0, min(bbox_A[1].start, bbox_B[1].start) - margin_cells_y)
    y_max = min(NY, max(bbox_A[1].stop, bbox_B[1].stop) + margin_cells_y)
    z_min = max(0, min(bbox_A[2].start, bbox_B[2].start) - margin_cells_z)
    z_max = min(NZ, max(bbox_A[2].stop, bbox_B[2].stop) + margin_cells_z)

    crop_slices = (slice(x_min, x_max), slice(y_min, y_max), slice(z_min, z_max))

    rho_A_crop = rho_A[crop_slices]
    rho_B_crop = rho_B[crop_slices]

    return rho_A_crop, rho_B_crop, crop_slices


def verify_crop_quality(rho_orig, rho_crop, dV, label="Density"):
    """
    Verify that cropping did not lose significant charge.

    Parameters
    ----------
    rho_orig : np.ndarray
        Original density.
    rho_crop : np.ndarray
        Cropped density.
    dV : float
        Volume element in SI units (m³).
    label : str, optional
        Label for printing. Default: "Density".

    Returns
    -------
    loss_percent : float
        Percentage of charge lost by cropping.
    reduction_percent : float
        Percentage reduction in grid points.
    """
    # Use absolute value to handle transition densities
    Q_orig = np.sum(np.abs(rho_orig)) * dV
    Q_crop = np.sum(np.abs(rho_crop)) * dV

    if Q_orig > 1e-30:
        loss_percent = 100 * abs(Q_orig - Q_crop) / Q_orig
    else:
        loss_percent = 0.0

    size_orig = np.prod(rho_orig.shape)
    size_crop = np.prod(rho_crop.shape)
    reduction_percent = 100 * (1 - size_crop / size_orig)

    if VERBOSE:
        print(f"  {label}: grid {size_orig:,} → {size_crop:,} pts "
              f"({reduction_percent:.1f}% reduction), charge loss {loss_percent:.4f}%")

    # Warning if significant charge lost
    if loss_percent > 1.0:
        warnings.warn(
            f"{label}: Cropping lost {loss_percent:.2f}% of charge! "
            f"Consider decreasing --crop-threshold or increasing --crop-margin.",
            UserWarning
        )

    return loss_percent, reduction_percent


def _centered_pad(a, add_cells_xyz):
    """
    Add centered zero padding around 3D array a.

    Parameters
    ----------
    a : np.ndarray, shape (NX, NY, NZ)
        Input array.
    add_cells_xyz : tuple (px, py, pz)
        TOTAL number of extra cells along each axis.

    Returns
    -------
    np.ndarray
        Padded array.
    """
    NX, NY, NZ = a.shape
    px, py, pz = add_cells_xyz
    NXp, NYp, NZp = NX + px, NY + py, NZ + pz

    out = np.zeros((NXp, NYp, NZp), dtype=a.dtype)
    sx = px // 2
    sy = py // 2
    sz = pz // 2
    out[sx:sx+NX, sy:sy+NY, sz:sz+NZ] = a

    return out


def _cells_needed_for_margin(length_needed_m, d_m):
    """
    Calculate number of cells needed for a given margin in meters.

    Parameters
    ----------
    length_needed_m : float
        Margin in meters.
    d_m : float
        Voxel size in meters.

    Returns
    -------
    int
        Total number of cells to add (sum of both sides).
    """
    return int(np.ceil(length_needed_m / d_m))


def compute_g_fft_coulomb(rho_A, rho_B, steps, units_len,
                           eps_eff=1.0, pad=1.0, fft_margin_bohr=0.0,
                           use_crop=False, crop_threshold_bohr3=1e-6, 
                           crop_margin_bohr=2.0):
    """
    Compute Coulomb coupling between two 3D charge densities using FFT.

    This function solves the Poisson equation in Fourier space to compute
    the electrostatic coupling integral: g = ∫ ρ_A(r) Φ_B(r) dV

    Parameters
    ----------
    rho_A, rho_B : np.ndarray, shape (NX, NY, NZ)
        Charge density distributions (transition densities in e/bohr³ or e/Å³).
    steps : tuple (dx, dy, dz)
        Grid spacing in bohr or Å.
    units_len : str
        "bohr" or "ang".
    eps_eff : float, optional
        Effective dielectric constant. Default: 1.0.
    pad : float, optional
        Padding factor for FFT. Default: 1.0.
    fft_margin_bohr : float, optional
        Extra margin in bohr/angstrom for FFT. Default: 0.0.
    use_crop : bool, optional
        Crop to bounding box before FFT. Default: False.
    crop_threshold_bohr3 : float, optional
        Density threshold for cropping. Default: 1e-6.
    crop_margin_bohr : float, optional
        Physical margin for cropping. Default: 2.0.

    Returns
    -------
    float
        Coulomb coupling in eV (bare value, multiply by 2 for TDDFT Hamiltonian).
    """
    # Convert to float
    rho_A = np.asarray(rho_A, dtype=float)
    rho_B = np.asarray(rho_B, dtype=float)
    dx, dy, dz = steps

    # Unit conversion
    if units_len.lower().startswith("bohr"):
        L_to_m = BOHR_TO_M
        margin_m = fft_margin_bohr * BOHR_TO_M
    elif units_len.lower().startswith("ang"):
        L_to_m = ANG_TO_M
        margin_m = fft_margin_bohr * ANG_TO_M
    else:
        raise ValueError("units_len must be 'bohr' or 'ang'.")

    dx_m, dy_m, dz_m = dx * L_to_m, dy * L_to_m, dz * L_to_m
    dV_SI = dx_m * dy_m * dz_m

    # Charge density in C/m³
    rhoA_SI = rho_A * (E_CHARGE / (L_to_m**3))
    rhoB_SI = rho_B * (E_CHARGE / (L_to_m**3))

    # Cropping
    if use_crop:
        rhoA_SI, rhoB_SI, crop_slices = crop_density_to_union_bbox(
            rhoA_SI, rhoB_SI, 
            threshold=crop_threshold_bohr3 * (E_CHARGE / (L_to_m**3)),
            margin_phys=crop_margin_bohr,
            steps=(dx, dy, dz)
        )
        if VERBOSE:
            print(f"  Cropped grid: {rhoA_SI.shape} (original: {rho_A.shape})")

    # Padding
    NX, NY, NZ = rhoA_SI.shape
    NXp0 = int(NX * (1.0 + pad))
    NYp0 = int(NY * (1.0 + pad))
    NZp0 = int(NZ * (1.0 + pad))

    add_x_margin = _cells_needed_for_margin(2.0 * margin_m, dx_m)
    add_y_margin = _cells_needed_for_margin(2.0 * margin_m, dy_m)
    add_z_margin = _cells_needed_for_margin(2.0 * margin_m, dz_m)

    NXp = NXp0 + add_x_margin
    NYp = NYp0 + add_y_margin
    NZp = NZp0 + add_z_margin

    px = NXp - NX
    py = NYp - NY
    pz = NZp - NZ

    rhoA_p = _centered_pad(rhoA_SI, (px, py, pz))
    rhoB_p = _centered_pad(rhoB_SI, (px, py, pz))

    # FFT of density B
    rhoB_k = fft.fftn(rhoB_p)

    # k-grid in rad/m
    kx = 2.0 * np.pi * fft.fftfreq(rhoB_p.shape[0], d=dx_m)
    ky = 2.0 * np.pi * fft.fftfreq(rhoB_p.shape[1], d=dy_m)
    kz = 2.0 * np.pi * fft.fftfreq(rhoB_p.shape[2], d=dz_m)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX*KX + KY*KY + KZ*KZ

    # Solve Poisson in k-space: Φ_k = ρ_k / (ε₀ ε_eff k²)
    PhiB_k = np.zeros_like(rhoB_k, dtype=complex)
    mask = (K2 > 1.0e-20)
    PhiB_k[mask] = rhoB_k[mask] / (EPS0 * eps_eff * K2[mask])

    # Inverse FFT → Φ_B(r)
    PhiB_p = np.real(fft.ifftn(PhiB_k))

    # Integral: g = ∫ ρ_A(r) Φ_B(r) dV
    g_J = np.sum(rhoA_p * PhiB_p) * dV_SI
    g_eV = g_J / E_CHARGE

    return g_eV


def build_hamiltonian(omega_m, omega_p, g_matrix, V_matrix, J_matrix):
    """
    Build the full polariton Hamiltonian matrix.

    H = [[H_mol,  2g  ],
         [2g^T,   H_plas]]

    where H_mol includes molecular energies + molecule-molecule couplings,
    H_plas includes plasmon energies + plasmon-plasmon couplings.

    Parameters
    ----------
    omega_m : array-like, shape (N_m,)
        Molecular excitation energies (eV).
    omega_p : array-like, shape (N_p,)
        Plasmon excitation energies (eV).
    g_matrix : np.ndarray, shape (N_m, N_p)
        Molecule-plasmon couplings (eV, bare values).
    V_matrix : np.ndarray, shape (N_m, N_m)
        Molecule-molecule couplings (eV, bare values).
    J_matrix : np.ndarray, shape (N_p, N_p)
        Plasmon-plasmon couplings (eV, bare values).

    Returns
    -------
    np.ndarray, shape (N_m + N_p, N_m + N_p)
        Full Hamiltonian matrix.
    """
    omega_m = np.asarray(omega_m)
    omega_p = np.asarray(omega_p)
    N_m = len(omega_m)
    N_p = len(omega_p)
    dim = N_m + N_p

    H = np.zeros((dim, dim), dtype=float)

    # Molecular block: diagonal + V
    for m in range(N_m):
        H[m, m] = omega_m[m]
    for i in range(N_m):
        for j in range(N_m):
            if i != j:
                H[i, j] = 2 * V_matrix[i, j]

    # Plasmon block: diagonal + J
    for p in range(N_p):
        H[N_m + p, N_m + p] = omega_p[p]
    for i in range(N_p):
        for j in range(N_p):
            if i != j:
                H[N_m + i, N_m + j] = 2 * J_matrix[i, j]

    # Molecule-plasmon couplings (off-diagonal blocks)
    for m in range(N_m):
        for p in range(N_p):
            H[m, N_m + p] = 2 * g_matrix[m, p]
            H[N_m + p, m] = 2 * g_matrix[m, p]

    return H


def diagonalize_hamiltonian(H):
    """
    Diagonalize the Hamiltonian to get polariton energies and states.

    Parameters
    ----------
    H : np.ndarray, shape (N, N)
        Hamiltonian matrix.

    Returns
    -------
    energies : np.ndarray, shape (N,)
        Polariton energies (eigenvalues), sorted in ascending order.
    states : np.ndarray, shape (N, N)
        Polariton states (eigenvectors), columns correspond to states.
    """
    energies, states = np.linalg.eigh(H)
    return energies, states


def analyze_polariton_character(states, N_m, N_p):
    """
    Analyze the molecular vs plasmonic character of each polariton state.

    Parameters
    ----------
    states : np.ndarray, shape (N_m + N_p, N_states)
        Polariton eigenvectors.
    N_m : int
        Number of molecules.
    N_p : int
        Number of plasmons.

    Returns
    -------
    list of dict
        Character analysis for each state, containing:
        - 'weights': Hopfield coefficients (|C_i|²)
        - 'eigvec': eigvectors
        - 'mol_character': percentage molecular character
        - 'plas_character': percentage plasmonic character
    """
    N_states = states.shape[1]
    analysis = []

    for idx in range(N_states):
        eigvec = states[:, idx]
        weights = np.abs(eigvec)**2

        mol_char = np.sum(weights[:N_m]) * 100
        plas_char = np.sum(weights[N_m:]) * 100


        analysis.append({
            'weights': weights,
            'eigvec': eigvec,
            'mol_character': mol_char,
            'plas_character': plas_char,
        })

    return analysis
