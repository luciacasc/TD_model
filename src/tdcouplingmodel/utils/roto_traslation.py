"""
Rotation and translation utilities for cube files.

This module provides functions to apply rigid body transformations
(rotation + translation) to cube files, including atomic coordinates
and volumetric data resampling.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..io.cube_io import parse_cube_file, density_3d_from_flat, write_cube_file
from .constants import BOHR_TO_ANG, ANG_TO_BOHR


# ============================================================
# 0) Unit conversion
# ============================================================

def convert_vector(vec, from_unit, to_unit):
    """
    Convert a 3D vector between bohr and angstrom.

    Parameters
    ----------
    vec : array-like (3,)
        Vector components.
    from_unit : {"bohr", "ang"}
    to_unit : {"bohr", "ang"}

    Returns
    -------
    np.ndarray, shape (3,)
        Converted vector.
    """
    vec = np.array(vec, dtype=float)
    if from_unit == to_unit:
        return vec
    if from_unit == "bohr" and to_unit == "ang":
        return vec * BOHR_TO_ANG
    if from_unit == "ang" and to_unit == "bohr":
        return vec * ANG_TO_BOHR
    raise ValueError(f"Unsupported unit conversion {from_unit} -> {to_unit}")


# ============================================================
# 1) Grid utilities
# ============================================================

def grid_vectors_from_header(header):
    """
    Construct 1D grid coordinate vectors (x, y, z) from cube header.

    Assumes orthogonal axes and constant spacing dx, dy, dz.
    """
    (x0, y0, z0) = header["origin"]
    (NX, NY, NZ) = header["grid_size"]
    (dx, dy, dz) = header["steps"]

    x = x0 + np.arange(NX) * dx
    y = y0 + np.arange(NY) * dy
    z = z0 + np.arange(NZ) * dz

    return x, y, z


def make_interpolator_from_cube(header, density_3d, method="linear"):
    """
    Build a RegularGridInterpolator for the cube scalar field.
    """
    x, y, z = grid_vectors_from_header(header)
    interp = RegularGridInterpolator(
        (x, y, z),
        density_3d,
        method=method,
        bounds_error=False,
        fill_value=0.0,  # zero outside the original grid
    )
    return interp


# ============================================================
# 2) Rotation matrices
# ============================================================

def rotation_matrix_from_euler(angles_deg, order='ZYX'):
    """
    Build a 3x3 rotation matrix from Euler angles (in degrees).

    Parameters
    ----------
    angles_deg : (alpha, beta, gamma)
        Angles in degrees.
    order : {"ZYX", "XYZ"}

    Returns
    -------
    np.ndarray, shape (3, 3)
    """
    a, b, c = np.deg2rad(angles_deg)

    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(a), -np.sin(a)],
                   [0.0, np.sin(a), np.cos(a)]])

    Ry = np.array([[np.cos(b), 0.0, np.sin(b)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(b), 0.0, np.cos(b)]])

    Rz = np.array([[np.cos(c), -np.sin(c), 0.0],
                   [np.sin(c), np.cos(c), 0.0],
                   [0.0, 0.0, 1.0]])

    if order == 'ZYX':
        R = Rx @ Ry @ Rz
    elif order == 'XYZ':
        R = Rz @ Ry @ Rx
    else:
        raise ValueError(f"Unsupported Euler order '{order}'")

    return R


def rotation_matrix_from_dipoles(mu_2, mu_1):
    """
    Build a 3x3 rotation matrix that aligns mu_1 with mu_2 using
    the axis-angle (Rodrigues) formula.

    Parameters
    ----------
    mu_2 : array-like (3,)
        Target dipole vector.
    mu_1 : array-like (3,)
        Source dipole vector to be rotated.

    Returns
    -------
    R : ndarray (3,3)
        Rotation matrix.
    theta_deg : float
        Rotation angle in degrees.
    u_hat : ndarray (3,)
        Normalized rotation axis.
    """
    mu_2 = np.array(mu_2, dtype=float)
    mu_1 = np.array(mu_1, dtype=float)

    # 1. Normalize vectors
    mu_2_hat = mu_2 / np.linalg.norm(mu_2)
    mu_1_hat = mu_1 / np.linalg.norm(mu_1)

    # 2. Angle between vectors
    cos_theta = np.dot(mu_1_hat, mu_2_hat)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)

    # 3. Rotation axis
    u = np.cross(mu_1_hat, mu_2_hat)
    norm_u = np.linalg.norm(u)

    # Special cases
    if norm_u < 1e-10:
        if cos_theta > 0:
            # Already aligned
            u_hat = np.array([0.0, 0.0, 1.0])
            return np.eye(3), 0.0, u_hat
        else:
            # Opposite: 180° around any perpendicular axis
            if abs(mu_2_hat[0]) < 0.9:
                perp = np.array([1.0, 0.0, 0.0])
            else:
                perp = np.array([0.0, 1.0, 0.0])
            u_hat = np.cross(mu_2_hat, perp)
            u_hat = u_hat / np.linalg.norm(u_hat)
            theta_rad = np.pi
            theta_deg = 180.0
    else:
        u_hat = u / norm_u

    # 4. Rodrigues formula
    ux, uy, uz = u_hat
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    one_minus_cos = 1.0 - cos_t

    R = np.array([
        [cos_t + ux**2 * one_minus_cos,
         ux*uy*one_minus_cos - uz*sin_t,
         ux*uz*one_minus_cos + uy*sin_t],

        [uy*ux*one_minus_cos + uz*sin_t,
         cos_t + uy**2 * one_minus_cos,
         uy*uz*one_minus_cos - ux*sin_t],

        [uz*ux*one_minus_cos - uy*sin_t,
         uz*uy*one_minus_cos + ux*sin_t,
         cos_t + uz**2 * one_minus_cos],
    ])

    return R, theta_deg, u_hat


# ============================================================
# 3) Rigid-body transform utilities
# ============================================================

def transform_points_about_center(pts, R=None, t=None, center=None):
    """
    Apply a rigid body transform to a set of points:

        p' = R @ (p - center) + center + t
    """
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)
    if center is None:
        center = np.zeros(3)

    pts_shifted = pts - center
    pts_rot = pts_shifted @ R.T
    pts_new = pts_rot + center + t

    return pts_new


def atoms_coords_from_header(header):
    """
    Extract atomic Z, charges, and coordinates from header.
    """
    atom_Z, atom_q, ax, ay, az = header["atoms"]
    coords = np.vstack((ax, ay, az)).T
    return np.array(atom_Z), np.array(atom_q), coords


def transform_atoms(header, R, t, center):
    """
    Apply the same rigid body transform to the atoms as to the density.
    """
    Z, q, coords = atoms_coords_from_header(header)
    coords_t = transform_points_about_center(coords, R=R, t=t, center=center)
    return Z, q, coords_t


# ============================================================
# 4) Build new grid and resample density
# ============================================================

def build_new_grid_like_original(header, t):
    """
    New grid: same NX,NY,NZ and spacing, origin shifted by t.
    """
    (x0, y0, z0) = header["origin"]
    dx, dy, dz = header["steps"]
    NX, NY, NZ = header["grid_size"]

    x0_new = x0 + t[0]
    y0_new = y0 + t[1]
    z0_new = z0 + t[2]

    origin_new = (x0_new, y0_new, z0_new)
    grid_size_new = (NX, NY, NZ)
    steps_new = (dx, dy, dz)

    x_new = x0_new + np.arange(NX) * dx
    y_new = y0_new + np.arange(NY) * dy
    z_new = z0_new + np.arange(NZ) * dz

    return origin_new, grid_size_new, steps_new, (x_new, y_new, z_new)


def resample_density_to_new_grid(header_in, density_3d_in, R, t, center,
                                 x_new, y_new, z_new):
    """
    Resample the original density onto a new grid after a rigid body transform.
    """
    interp = make_interpolator_from_cube(header_in, density_3d_in)

    Xn, Yn, Zn = np.meshgrid(x_new, y_new, z_new, indexing='ij')
    pts_new = np.vstack((
        Xn.ravel(order='C'),
        Yn.ravel(order='C'),
        Zn.ravel(order='C'),
    )).T

    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)
    if center is None:
        center = np.zeros(3)

    # r_old = R.T @ (r_new - center - t) + center
    pts_shifted = pts_new - center - t
    pts_old = pts_shifted @ R + center

    vals_flat = interp(pts_old)
    NX, NY, NZ = header_in["grid_size"]
    density_new = vals_flat.reshape((NX, NY, NZ), order='C')

    return density_new


# ============================================================
# 5) High-level function with dipole support
# ============================================================

def rotate_translate_cube(
    cube_in,
    cube_out,
    angles=(0.0, 0.0, 0.0),
    trans=(0.0, 0.0, 0.0),
    center=None,
    input_unit="ang",
    mu_2=None,
    mu_1=None,
    verbose=True,
):
    """
    Apply rigid body rotation and translation to a cube file.

    Priority for rotation:
    1) If mu_2 and mu_1 are provided: use dipole alignment rotation.
    2) Else if angles != (0,0,0): use Euler angles.
    3) Else: identity (no rotation).
    """
    # --- Read input cube
    header_in, dens_flat_in = parse_cube_file(cube_in)
    dens3d_in = density_3d_from_flat(dens_flat_in, header_in["grid_size"])
    coord_unit = header_in.get("coord_unit", "bohr")

    if verbose:
        print("Input cube:", cube_in)
        print("Detected coordinate unit from cube grid:", coord_unit)
        print("User input unit (trans/center):", input_unit)

    # --- Build rotation matrix
    R = None
    used_dipoles = False

    if mu_2 is not None and mu_1 is not None:
        R, theta_deg, u_hat = rotation_matrix_from_dipoles(mu_2, mu_1)
        used_dipoles = True
        if verbose:
            print("\nUsing dipole alignment rotation")
            print("  mu_2   (target):", np.array(mu_2))
            print("  mu_1 (source):", np.array(mu_1))
            print(f"  Angle: {theta_deg:.2f}°")
            print("  Axis: ", u_hat)
    elif angles != (0.0, 0.0, 0.0):
        R = rotation_matrix_from_euler(angles, order='ZYX')
        if verbose:
            print("Using Euler angles rotation")
            print("  Angles (deg, ZYX):", angles)
    else:
        R = np.eye(3)
        if verbose:
            print("No rotation requested (identity).")

    # --- Translation and center: convert from input_unit to coord_unit
    t_user = np.array(trans, dtype=float)

    if center is not None:
        center_user = np.array(center, dtype=float)
        center_internal = convert_vector(center_user, input_unit, coord_unit)
    else:
        # center of the original grid, already in coord_unit
        x, y, z = grid_vectors_from_header(header_in)
        center_internal = np.array([
            0.5 * (x[0] + x[-1]),
            0.5 * (y[0] + y[-1]),
            0.5 * (z[0] + z[-1]),
        ])

    t = convert_vector(t_user, input_unit, coord_unit)

    if verbose:
        if not used_dipoles:
            print("Euler angles (deg, ZYX):", angles)
        print("Translation (internal unit):", t)
        print("Rotation center (internal unit):", center_internal)

    # --- Build new grid (same resolution, origin shifted by t)
    origin_new, grid_size_new, steps_new, (x_new, y_new, z_new) = \
        build_new_grid_like_original(header_in, t)

    # --- Resample density on new grid according to rigid-body transform
    dens3d_out = resample_density_to_new_grid(
        header_in, dens3d_in, R, t, center_internal, x_new, y_new, z_new
    )

    # --- Transform atoms
    atom_Z, atom_q, atom_coords_t = transform_atoms(header_in, R, t, center_internal)

    # --- Prepare output header
    header_out = {
        "natom": header_in["natom"],
        "origin": origin_new,
        "grid_size": grid_size_new,
        "steps": steps_new,
        "atoms": (
            atom_Z.tolist(),
            atom_q.tolist(),
            atom_coords_t[:, 0].tolist(),
            atom_coords_t[:, 1].tolist(),
            atom_coords_t[:, 2].tolist(),
        ),
        "coord_unit": coord_unit,
    }

    # Build comments
    comment1 = "Cube file generated by tdcouplingmodel rotation"
    if mu_2 is not None and mu_1 is not None:
        comment2 = (
            f"Rigid-body transform (coord units: {coord_unit}); "
            f"dipole alignment used; "
            f"trans: {trans[0]:.6f} {trans[1]:.6f} {trans[2]:.6f} ({input_unit})"
        )
    else:
        a, b, c = angles
        comment2 = (
            f"Rigid-body transform (coord units: {coord_unit}); "
            f"angles: {a:.6f} {b:.6f} {c:.6f} deg (ZYX); "
            f"trans: {trans[0]:.6f} {trans[1]:.6f} {trans[2]:.6f} ({input_unit})"
        )

    # --- Write output cube
    write_cube_file(cube_out, header_out, dens3d_out, comment1, comment2)

    if verbose:
        print("Output cube:", cube_out)
        print("Done.")

    return {
        "angles": angles,
        "translation": trans,
        "center": center,
        "input_unit": input_unit,
        "coord_unit": coord_unit,
        "used_dipoles": used_dipoles,
    }
