"""
Cube file summation utilities.

This module provides functions to sum multiple cube files with optional
scaling factors, combining both volumetric data and atomic coordinates.
"""

import numpy as np
from ..io.cube_io import parse_cube_file, density_3d_from_flat, write_cube_file


def sum_cube_densities(density1, density2, factor1=1.0, factor2=1.0):
    """
    Sum two density arrays with optional scaling factors.

    Parameters
    ----------
    density1 : np.ndarray, shape (NX, NY, NZ)
        First density array.
    density2 : np.ndarray, shape (NX, NY, NZ)
        Second density array.
    factor1 : float, optional
        Scaling factor for density1. Default: 1.0.
    factor2 : float, optional
        Scaling factor for density2. Default: 1.0.

    Returns
    -------
    np.ndarray, shape (NX, NY, NZ)
        Summed density: factor1 * density1 + factor2 * density2.

    Raises
    ------
    ValueError
        If density shapes don't match.
    """
    if density1.shape != density2.shape:
        raise ValueError(
            f"Density shapes don't match! "
            f"density1: {density1.shape}, density2: {density2.shape}"
        )

    return (density1 * factor1) + (density2 * factor2)


def combine_atom_data(header1, header2):
    """
    Combine atomic data from two cube headers.

    Parameters
    ----------
    header1 : dict
        First cube header.
    header2 : dict
        Second cube header.

    Returns
    -------
    tuple
        Combined (atom_Z, atom_q, atom_x, atom_y, atom_z) lists.
    """
    Z1, q1, x1, y1, z1 = header1["atoms"]
    Z2, q2, x2, y2, z2 = header2["atoms"]

    atom_Z = list(Z1) + list(Z2)
    atom_q = list(q1) + list(q2)
    atom_x = list(x1) + list(x2)
    atom_y = list(y1) + list(y2)
    atom_z = list(z1) + list(z2)

    return (atom_Z, atom_q, atom_x, atom_y, atom_z)


def verify_grid_compatibility(header1, header2, strict=True):
    """
    Verify that two cube grids are compatible for summation.

    Parameters
    ----------
    header1 : dict
        First cube header.
    header2 : dict
        Second cube header.
    strict : bool, optional
        If True, raise errors for any incompatibility.
        If False, only print warnings. Default: True.

    Returns
    -------
    bool
        True if grids are compatible.

    Raises
    ------
    ValueError
        If grids are incompatible and strict=True.
    """
    compatible = True

    # Check grid sizes
    if header1["grid_size"] != header2["grid_size"]:
        msg = (f"Grid sizes don't match! "
               f"File 1: {header1['grid_size']}, File 2: {header2['grid_size']}")
        if strict:
            raise ValueError(msg)
        else:
            print(f"WARNING: {msg}")
            compatible = False

    # Check origins
    origin1 = np.array(header1["origin"])
    origin2 = np.array(header2["origin"])
    if not np.allclose(origin1, origin2, rtol=1e-6):
        msg = f"Origins are different! File 1: {origin1}, File 2: {origin2}"
        if strict:
            print(f"WARNING: {msg}")
        else:
            print(f"WARNING: {msg}")

    # Check voxel spacing
    steps1 = np.array(header1["steps"])
    steps2 = np.array(header2["steps"])
    if not np.allclose(steps1, steps2, rtol=1e-6):
        msg = f"Voxel steps are different! File 1: {steps1}, File 2: {steps2}"
        if strict:
            print(f"WARNING: {msg}")
        else:
            print(f"WARNING: {msg}")

    # Check coordinate units
    unit1 = header1.get("coord_unit", "bohr")
    unit2 = header2.get("coord_unit", "bohr")
    if unit1 != unit2:
        msg = f"Coordinate units differ! File 1: {unit1}, File 2: {unit2}"
        if strict:
            raise ValueError(msg)
        else:
            print(f"WARNING: {msg}")
            compatible = False

    return compatible


def sum_cube_files(file1, file2, output_file, factor1=1.0, factor2=1.0, 
                    verify=True, verbose=True):
    """
    Sum two cube files with optional scaling factors.

    This function reads two cube files, sums their volumetric data with
    optional scaling factors, combines atomic coordinates, and writes
    the result to a new cube file.

    Parameters
    ----------
    file1 : str
        Path to first cube file.
    file2 : str
        Path to second cube file.
    output_file : str
        Path to output cube file.
    factor1 : float, optional
        Scaling factor for file1 density. Default: 1.0.
    factor2 : float, optional
        Scaling factor for file2 density. Default: 1.0.
    verify : bool, optional
        Verify result by re-reading output file. Default: True.
    verbose : bool, optional
        Print detailed progress information. Default: True.

    Returns
    -------
    dict
        Statistics about the summation:
        - 'density_min': minimum value in summed density
        - 'density_max': maximum value in summed density
        - 'density_mean': mean value in summed density
        - 'natom_total': total number of atoms
        - 'verified': True if verification passed (if verify=True)

    Raises
    ------
    ValueError
        If grid sizes are incompatible.
    """
    if verbose:
        print(f"Reading {file1}...")
    header1, density1_flat = parse_cube_file(file1)
    density1 = density_3d_from_flat(density1_flat, header1["grid_size"])

    if verbose:
        print(f"Reading {file2}...")
    header2, density2_flat = parse_cube_file(file2)
    density2 = density_3d_from_flat(density2_flat, header2["grid_size"])

    # Verify compatibility
    if verbose:
        print("\nVerifying grid compatibility...")
    verify_grid_compatibility(header1, header2, strict=True)

    # Sum densities
    if verbose:
        print(f"\nScaling density 1 by factor {factor1}...")
        print(f"Scaling density 2 by factor {factor2}...")
        print("Summing densities...")

    density_sum = sum_cube_densities(density1, density2, factor1, factor2)

    # Statistics
    stats = {
        'density_min': float(density_sum.min()),
        'density_max': float(density_sum.max()),
        'density_mean': float(density_sum.mean()),
    }

    if verbose:
        print(f"\nSummed density statistics:")
        print(f"  Min:  {stats['density_min']:.6e}")
        print(f"  Max:  {stats['density_max']:.6e}")
        print(f"  Mean: {stats['density_mean']:.6e}")

    # Combine atoms
    combined_atoms = combine_atom_data(header1, header2)
    natom_total = header1["natom"] + header2["natom"]
    stats['natom_total'] = natom_total

    if verbose:
        print(f"\nCombining atomic coordinates:")
        print(f"  System 1: {header1['natom']} atoms")
        print(f"  System 2: {header2['natom']} atoms")
        print(f"  Total: {natom_total} atoms")

    # Prepare output header
    header_out = {
        "natom": natom_total,
        "origin": header1["origin"],
        "grid_size": header1["grid_size"],
        "steps": header1["steps"],
        "atoms": combined_atoms,
        "coord_unit": header1.get("coord_unit", "bohr"),
    }

    # Comments
    comment1 = f"Sum of {file1} (×{factor1}) and {file2} (×{factor2})"
    comment2 = f"System 1: {header1['natom']} atoms, System 2: {header2['natom']} atoms"

    # Write output
    if verbose:
        print(f"\nWriting {output_file}...")
    write_cube_file(output_file, header_out, density_sum, comment1, comment2)

    # Verification
    if verify:
        if verbose:
            print(f"\nVerifying output (re-reading file)...")
        header_check, density_check_flat = parse_cube_file(output_file)
        density_check = density_3d_from_flat(density_check_flat, header_check["grid_size"])

        max_diff = np.abs(density_sum - density_check).max()

        if verbose:
            print(f"  Min:  {density_check.min():.6e}")
            print(f"  Max:  {density_check.max():.6e}")
            print(f"  Mean: {density_check.mean():.6e}")
            print(f"  Max difference: {max_diff:.6e}")

        verified = np.allclose(density_sum, density_check, rtol=1e-5, atol=1e-10)
        stats['verified'] = verified

        if verified:
            if verbose:
                print("\nVerification passed!")
        else:
            n_diff = np.sum(~np.isclose(density_sum, density_check))
            if verbose:
                print(f"\nWARNING: Discrepancy detected!")
                print(f"  Number of differing points: {n_diff}")

    if verbose:
        print(f"\nCompleted! File saved: {output_file}")
        print(f"  Factor 1: {factor1}")
        print(f"  Factor 2: {factor2}")
        print(f"  Total atoms: {natom_total}")

    return stats
