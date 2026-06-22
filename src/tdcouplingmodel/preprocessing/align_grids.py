"""
Grid alignment utilities for cube files.

This module provides functions to align multiple cube files onto a common grid
using resampling with scipy interpolation.
"""

import numpy as np
from scipy.ndimage import map_coordinates
from ..io.cube_io import parse_cube_file, density_3d_from_flat, write_cube_file


def compute_charge_and_center(density_3d, origin, steps):
    """
    Calculate total charge and center of charge from 3D density.

    Parameters
    ----------
    density_3d : np.ndarray, shape (NX, NY, NZ)
        Charge density distribution.
    origin : tuple (x0, y0, z0)
        Grid origin.
    steps : tuple (dx, dy, dz)
        Grid spacing.

    Returns
    -------
    Q_total : float
        Total integrated charge.
    center : np.ndarray, shape (3,)
        Center of charge (x, y, z).
    """
    NX, NY, NZ = density_3d.shape
    dx, dy, dz = steps
    dV = dx * dy * dz

    # Total charge
    Q_total = np.sum(density_3d) * dV

    # Coordinate grids (voxel centers)
    x0, y0, z0 = origin
    x = x0 + dx/2 + dx * np.arange(NX)
    y = y0 + dy/2 + dy * np.arange(NY)
    z = z0 + dz/2 + dz * np.arange(NZ)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Center of charge
    if abs(Q_total) > 1e-10:
        cx = np.sum(X * density_3d) * dV / Q_total
        cy = np.sum(Y * density_3d) * dV / Q_total
        cz = np.sum(Z * density_3d) * dV / Q_total
    else:
        cx = cy = cz = 0.0

    return Q_total, np.array([cx, cy, cz])


def verify_charge_conservation(density_orig, origin_orig, steps_orig,
                                 density_resamp, origin_resamp, steps_resamp,
                                 file_label="Density", verbose=True):
    """
    Verify charge conservation after resampling.

    Parameters
    ----------
    density_orig : np.ndarray
        Original density.
    origin_orig : tuple
        Original grid origin.
    steps_orig : tuple
        Original grid spacing.
    density_resamp : np.ndarray
        Resampled density.
    origin_resamp : tuple
        Resampled grid origin.
    steps_resamp : tuple
        Resampled grid spacing.
    file_label : str, optional
        Label for printing. Default: "Density".
    verbose : bool, optional
        Print verification info. Default: True.

    Returns
    -------
    dict
        Dictionary with verification statistics.
    """
    Q_orig, center_orig = compute_charge_and_center(density_orig, origin_orig, steps_orig)
    Q_resamp, center_resamp = compute_charge_and_center(density_resamp, origin_resamp, steps_resamp)

    if abs(Q_orig) > 1e-10:
        charge_error_pct = 100 * abs(Q_orig - Q_resamp) / abs(Q_orig)
    else:
        charge_error_pct = 0.0

    center_shift = np.linalg.norm(center_resamp - center_orig)

    if verbose:
        print(f"\n--- Verification: {file_label} ---")
        print(f"Charge original:  {Q_orig:12.6e}")
        print(f"Charge resampled: {Q_resamp:12.6e}")
        if abs(Q_orig) > 1e-10:
            print(f"Charge difference: {abs(Q_orig - Q_resamp):12.6e} ({charge_error_pct:.2f}%)")
        else:
            print(f"Charge difference: {abs(Q_orig - Q_resamp):12.6e}")
        print(f"Center original:  {center_orig}")
        print(f"Center resampled: {center_resamp}")
        print(f"Center shift: {center_shift:.6f}")

        # Warnings
        if charge_error_pct > 1.0:
            print("WARNING: Charge NOT conserved! Error > 1%")
        if center_shift > 0.1:
            print("WARNING: Center of charge shifted > 0.1!")

    return {
        "Q_orig": Q_orig,
        "Q_resamp": Q_resamp,
        "charge_error_pct": charge_error_pct,
        "center_orig": center_orig,
        "center_resamp": center_resamp,
        "center_shift": center_shift,
    }


def resample_cube_on_grid(density_3d, old_origin, old_steps,
                           new_origin, new_size, new_steps, order=3):
    """
    Resample a 3D density onto a new grid using spline interpolation.

    Parameters
    ----------
    density_3d : np.ndarray, shape (NX_old, NY_old, NZ_old)
        Original density.
    old_origin : tuple (x0, y0, z0)
        Origin of old grid.
    old_steps : tuple (dx, dy, dz)
        Spacing of old grid.
    new_origin : tuple (x0, y0, z0)
        Origin of new grid.
    new_size : tuple (NX, NY, NZ)
        Size of new grid.
    new_steps : tuple (dx, dy, dz)
        Spacing of new grid.
    order : int, optional
        Spline interpolation order (0-5). Default: 3 (cubic).

    Returns
    -------
    np.ndarray, shape (NX, NY, NZ)
        Resampled density on new grid.
    """
    x0_old, y0_old, z0_old = old_origin
    dx_old, dy_old, dz_old = old_steps

    x0_new, y0_new, z0_new = new_origin
    NX, NY, NZ = new_size
    dx, dy, dz = new_steps

    # Create physical coordinates of NEW grid (voxel centers)
    x_new = x0_new + dx/2 + dx * np.arange(NX)
    y_new = y0_new + dy/2 + dy * np.arange(NY)
    z_new = z0_new + dz/2 + dz * np.arange(NZ)
    X_new, Y_new, Z_new = np.meshgrid(x_new, y_new, z_new, indexing='ij')

    # Convert physical coordinates to fractional indices in OLD grid
    i_old = (X_new - x0_old) / dx_old - 0.5
    j_old = (Y_new - y0_old) / dy_old - 0.5
    k_old = (Z_new - z0_old) / dz_old - 0.5

    # Interpolate
    new_density = map_coordinates(
        density_3d,
        [i_old, j_old, k_old],
        order=order,
        mode='constant',
        cval=0.0
    )

    return new_density


def compute_common_grid(cube_data_list, spacing='min'):
    """
    Compute a common grid that contains all input grids.

    Parameters
    ----------
    cube_data_list : list of dict
        List of cube data dictionaries (each with 'origin', 'grid_size', 'steps', 'units').
    spacing : str or float, optional
        Spacing strategy:
        - 'min': use finest spacing among all grids
        - 'max': use coarsest spacing
        - 'mean': use average spacing
        - float: use this fixed value
        Default: 'min'.

    Returns
    -------
    common_origin : np.ndarray, shape (3,)
        Origin of common grid.
    common_size : tuple (NX, NY, NZ)
        Size of common grid.
    common_steps : np.ndarray, shape (3,)
        Spacing of common grid.
    common_units : str
        Units (from first cube).
    """
    all_origins = []
    all_ends = []
    all_spacings = []

    for cube in cube_data_list:
        origin = np.array(cube['origin'])
        steps = np.array(cube['steps'])
        size = np.array(cube['grid_size'])

        # Calculate end point (assuming orthogonal grid)
        end = origin + steps * size

        all_origins.append(origin)
        all_ends.append(end)
        all_spacings.append(steps)

    all_origins = np.array(all_origins)
    all_ends = np.array(all_ends)
    all_spacings = np.array(all_spacings)

    # Common grid: min origin, max end
    common_origin = np.min(all_origins, axis=0)
    common_end = np.max(all_ends, axis=0)

    # Determine spacing
    if spacing == 'min':
        new_spacing = np.min(all_spacings, axis=0)
    elif spacing == 'max':
        new_spacing = np.max(all_spacings, axis=0)
    elif spacing == 'mean':
        new_spacing = np.mean(all_spacings, axis=0)
    else:
        try:
            new_spacing = np.array([float(spacing)] * 3)
        except:
            raise ValueError(
                f"spacing must be 'min', 'max', 'mean' or a numeric value, not '{spacing}'"
            )

    # Calculate new grid size
    common_size = np.ceil((common_end - common_origin) / new_spacing).astype(int)

    # Units: take from first cube
    common_units = cube_data_list[0].get('coord_unit', 'bohr')

    return common_origin, tuple(common_size), new_spacing, common_units


def align_cube_grids(input_files, output_files, spacing='min', order=3, verify=False, verbose=True):
    """
    Align multiple cube files onto a common grid.

    Main high-level function that:
    1. Reads all input cubes
    2. Computes common grid
    3. Resamples all densities onto common grid
    4. Writes aligned output cubes

    Parameters
    ----------
    input_files : list of str
        Input cube file paths.
    output_files : list of str
        Output cube file paths (same length as input_files).
    spacing : str or float, optional
        Spacing mode ('min', 'max', 'mean', or numeric value). Default: 'min'.
    order : int, optional
        Spline interpolation order (0-5). Default: 3.
    verify : bool, optional
        Verify charge conservation for each file. Default: False.
    verbose : bool, optional
        Print progress information. Default: True.

    Returns
    -------
    dict
        Dictionary with alignment information and statistics.
    """
    if len(input_files) != len(output_files):
        raise ValueError(
            f"Number of input files ({len(input_files)}) != "
            f"number of output files ({len(output_files)})"
        )

    n_files = len(input_files)

    if verbose:
        print("=" * 70)
        print("CUBE GRID ALIGNMENT - COMMON GRID")
        print("=" * 70)
        print(f"Number of files: {n_files}")
        print(f"Spacing mode: {spacing}")
        print(f"Interpolation order: {order}")
        print()

    # Parse all files
    if verbose:
        print("Parsing cube files...")

    cube_data_list = []
    densities_3d = []

    for i, fname in enumerate(input_files):
        if verbose:
            print(f"  [{i+1}/{n_files}] {fname}")
        header, density_flat = parse_cube_file(fname)
        density_3d = density_3d_from_flat(density_flat, header["grid_size"])

        cube_data_list.append(header)
        densities_3d.append(density_3d)

    # Print original grid info
    if verbose:
        print("\n" + "=" * 70)
        print("ORIGINAL GRIDS")
        print("=" * 70)
        for i, (fname, cube) in enumerate(zip(input_files, cube_data_list)):
            print(f"\nFile {i+1}: {fname}")
            print(f"  Origin: {cube['origin']}")
            print(f"  Size: {cube['grid_size']}")
            print(f"  Spacing: {cube['steps']}")
            print(f"  Units: {cube.get('coord_unit', 'bohr')}")

            origin = np.array(cube['origin'])
            steps = np.array(cube['steps'])
            size = np.array(cube['grid_size'])
            end = origin + steps * size
            print(f"  Bounds: x=[{origin[0]:.2f}, {end[0]:.2f}], "
                  f"y=[{origin[1]:.2f}, {end[1]:.2f}], "
                  f"z=[{origin[2]:.2f}, {end[2]:.2f}]")

    # Compute common grid
    if verbose:
        print("\n" + "=" * 70)
        print("COMPUTING COMMON GRID")
        print("=" * 70)

    common_origin, common_size, common_steps, common_units = compute_common_grid(
        cube_data_list, spacing
    )

    common_end = common_origin + common_steps * np.array(common_size)

    if verbose:
        print(f"\nCommon grid:")
        print(f"  Origin: {common_origin}")
        print(f"  Size: {common_size}")
        print(f"  Spacing: {common_steps}")
        print(f"  Units: {common_units}")
        print(f"  Bounds: x=[{common_origin[0]:.2f}, {common_end[0]:.2f}], "
              f"y=[{common_origin[1]:.2f}, {common_end[1]:.2f}], "
              f"z=[{common_origin[2]:.2f}, {common_end[2]:.2f}]")
        print(f"  Total points: {common_size[0] * common_size[1] * common_size[2]:,}")

    # Resample all files
    if verbose:
        print("\n" + "=" * 70)
        print("RESAMPLING")
        print("=" * 70)

    resampled_densities = []
    verification_results = []

    for i, (density_3d, cube) in enumerate(zip(densities_3d, cube_data_list)):
        if verbose:
            print(f"\n[{i+1}/{n_files}] Resampling {input_files[i]}...")

        new_density = resample_cube_on_grid(
            density_3d,
            old_origin=cube['origin'],
            old_steps=cube['steps'],
            new_origin=tuple(common_origin),
            new_size=common_size,
            new_steps=tuple(common_steps),
            order=order
        )

        resampled_densities.append(new_density)

        # Verify charge conservation if requested
        if verify:
            ver_result = verify_charge_conservation(
                density_3d, cube['origin'], cube['steps'],
                new_density, tuple(common_origin), tuple(common_steps),
                file_label=f"File {i+1}",
                verbose=verbose
            )
            verification_results.append(ver_result)

    # Write output files
    if verbose:
        print("\n" + "=" * 70)
        print("WRITING OUTPUT FILES")
        print("=" * 70)

    for i, (outfile, cube, new_density) in enumerate(zip(output_files, cube_data_list, resampled_densities)):
        if verbose:
            print(f"[{i+1}/{n_files}] Writing {outfile}...")

        # Create output header
        header_out = {
            "natom": cube["natom"],
            "origin": tuple(common_origin),
            "grid_size": common_size,
            "steps": tuple(common_steps),
            "atoms": cube["atoms"],
            "coord_unit": common_units,
        }

        comment1 = f"Resampled on common grid from {input_files[i]}"
        comment2 = f"Common grid: size={common_size}, spacing={common_steps}"

        write_cube_file(outfile, header_out, new_density, comment1, comment2)

    if verbose:
        print("\n" + "=" * 70)
        print("COMPLETED!")
        print("=" * 70)
        print(f"\nAll {n_files} files have been resampled onto the common grid:")
        print(f"  Origin: {common_origin}")
        print(f"  Size: {common_size}")
        print(f"  Spacing: {common_steps}")
        print("\nYou can now use the aligned files with your main code.")

    return {
        "n_files": n_files,
        "common_origin": common_origin,
        "common_size": common_size,
        "common_steps": common_steps,
        "common_units": common_units,
        "verification_results": verification_results if verify else None,
    }
