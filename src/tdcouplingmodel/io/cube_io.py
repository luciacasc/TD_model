"""
Input/Output utilities for Gaussian cube files.

This module provides functions to read and write cube files with automatic
unit detection based on grid vector signs.
"""

import numpy as np
import itertools
import periodictable


# def parse_cube_file_basic(filename):
#     """
#     Minimal cube parser for geometry and units.

#     Auto-detect coordinate units from voxel vectors sign:
#       - positive voxel lengths → Bohr (default)
#       - negative voxel lengths → Angstroms

#     Parameters
#     ----------
#     filename : str
#         Path to the cube file.

#     Returns
#     -------
#     header : dict
#         Dictionary containing:
#         - natom: int - Number of atoms
#         - origin: tuple (x0, y0, z0)
#         - grid_size: tuple (NX, NY, NZ)
#         - steps: tuple (dx, dy, dz) - stored as positive magnitudes
#         - atoms: tuple (Z_list, q_list, x_list, y_list, z_list)
#         - coord_unit: str - "bohr" or "ang"

#     Raises
#     ------
#     ValueError
#         If file is too short or has inconsistent unit signs.
#     """
#     with open(filename, 'r') as f:
#         lines = f.read().splitlines()

#     if len(lines) < 6:
#         raise ValueError("File too short for a valid cube")

#     # Parse header
#     natom_str, x0, y0, z0 = lines[2].split()
#     natom = int(natom_str)
#     x0, y0, z0 = float(x0), float(y0), float(z0)

#     NX, dx, _, _ = lines[3].split()
#     NY, _, dy, _ = lines[4].split()
#     NZ, _, _, dz = lines[5].split()
#     NX, NY, NZ = int(NX), int(NY), int(NZ)
#     dx, dy, dz = float(dx), float(dy), float(dz)

#     # Determine units from sign of voxel lengths
#     if dx > 0 and dy > 0 and dz > 0:
#         coord_unit = "bohr"
#     elif dx < 0 and dy < 0 and dz < 0:
#         coord_unit = "ang"
#         dx, dy, dz = -dx, -dy, -dz
#     else:
#         raise ValueError(
#             "Inconsistent sign pattern in grid vectors; cannot determine units. "
#             "Expected all positive (Bohr) or all negative (Angstrom)."
#         )

#     # Parse atomic coordinates
#     coords = [L.split() for L in lines[6:6 + abs(natom)]]
#     atom_Z = [int(float(c[0])) for c in coords]
#     atom_q = [float(c[1]) for c in coords]
#     atom_x = [float(c[2]) for c in coords]
#     atom_y = [float(c[3]) for c in coords]
#     atom_z = [float(c[4]) for c in coords]

#     header = {
#         "natom": natom,
#         "origin": (x0, y0, z0),
#         "grid_size": (NX, NY, NZ),
#         "steps": (dx, dy, dz),
#         "atoms": (atom_Z, atom_q, atom_x, atom_y, atom_z),
#         "coord_unit": coord_unit,
#     }

#     return header


def parse_cube_file(filename):
    """
    Full cube file parser including volumetric data.

    Auto-detect coordinate units from voxel vectors sign:
      - positive voxel lengths → Bohr (default)
      - negative voxel lengths → Angstroms

    Parameters
    ----------
    filename : str
        Path to the cube file.

    Returns
    -------
    header : dict
        Dictionary containing:
        - natom: int - Number of atoms
        - origin: tuple (x0, y0, z0)
        - grid_size: tuple (NX, NY, NZ)
        - steps: tuple (dx, dy, dz) - stored as positive magnitudes
        - atoms: tuple (Z_list, q_list, x_list, y_list, z_list)
        - coord_unit: str - "bohr" or "ang"
        - raw_lines: list - All lines from the file (for reference)
    density_flat : np.ndarray
        1D array of length NX*NY*NZ with volumetric data.

    Raises
    ------
    ValueError
        If file is too short, has inconsistent units, or density size mismatch.
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()

    if len(lines) < 6:
        raise ValueError("File too short for a valid cube")

    # Parse header (same as basic version)
    natom_str, x0, y0, z0 = lines[2].split()
    natom = int(natom_str)
    x0, y0, z0 = float(x0), float(y0), float(z0)

    NX, dx, _, _ = lines[3].split()
    NY, _, dy, _ = lines[4].split()
    NZ, _, _, dz = lines[5].split()
    NX_raw, NY_raw, NZ_raw = int(NX), int(NY), int(NZ)
    dx, dy, dz = float(dx), float(dy), float(dz)

    # Determine units from sign
    if NX_raw > 0 and NY_raw > 0 and NZ_raw > 0:
        coord_unit = "bohr"
    elif NX_raw < 0 and NY_raw < 0 and NZ_raw < 0:
        coord_unit = "ang"
    else:
        raise ValueError(
            "Inconsistent sign pattern in grid vectors; cannot determine units."
        )

    NX, NY, NZ = abs(NX_raw), abs(NY_raw), abs(NZ_raw)

    # Parse atomic coordinates
    coords = [L.split() for L in lines[6:6 + abs(natom)]]
    atom_Z = [int(float(c[0])) for c in coords]
    atom_q = [float(c[1]) for c in coords]
    atom_x = [float(c[2]) for c in coords]
    atom_y = [float(c[3]) for c in coords]
    atom_z = [float(c[4]) for c in coords]

    # Parse volumetric data
    density_tokens = itertools.chain.from_iterable(
        L.split() for L in lines[6 + abs(natom):]
    )
    density_flat = np.array([float(v) for v in density_tokens], dtype=float)

    if density_flat.size != NX * NY * NZ:
        raise ValueError(
            f"Density size mismatch: expected {NX*NY*NZ}, got {density_flat.size}"
        )

    header = {
        "natom": natom,
        "origin": (x0, y0, z0),
        "grid_size": (NX, NY, NZ),
        "steps": (abs(dx), abs(dy), abs(dz)),  # Store as positive
        "atoms": (atom_Z, atom_q, atom_x, atom_y, atom_z),
        "coord_unit": coord_unit,
        "raw_lines": lines,
    }

    return header, density_flat


def density_3d_from_flat(density_flat, grid_size):
    """
    Reshape flat density array into 3D grid.

    Parameters
    ----------
    density_flat : np.ndarray
        1D array of volumetric data.
    grid_size : tuple (NX, NY, NZ)
        Grid dimensions.

    Returns
    -------
    np.ndarray
        3D array with shape (NX, NY, NZ) in C-order (x-outside, z-inside).
    """
    NX, NY, NZ = grid_size
    return density_flat.reshape((NX, NY, NZ), order='C')


def write_cube_file(filename, header, density_3d, comment_line1=None, comment_line2=None):
    """
    Write a cube file.

    Parameters
    ----------
    filename : str
        Output cube file path.
    header : dict
        Header dictionary (from parse_cube_file).
    density_3d : np.ndarray, shape (NX, NY, NZ)
        Volumetric data.
    comment_line1 : str, optional
        First comment line. If None, uses generic comment.
    comment_line2 : str, optional
        Second comment line. If None, uses generic comment.
    """
    natom = header["natom"]
    origin = header["origin"]
    grid_size = header["grid_size"]
    steps = header["steps"]
    coord_unit = header.get("coord_unit", "bohr")
    atom_Z, atom_q, atom_x, atom_y, atom_z = header["atoms"]

    NX, NY, NZ = grid_size
    dx, dy, dz = steps

    # Sign convention: positive = bohr, negative = angstrom
    if coord_unit == "bohr":
        NX_sign, NY_sign, NZ_sign = NX, NY, NZ
        dx_sign, dy_sign, dz_sign = dx, dy, dz
    elif coord_unit == "ang":
        NX_sign, NY_sign, NZ_sign = -NX, -NY, -NZ
        dx_sign, dy_sign, dz_sign = -dx, -dy, -dz
    else:
        raise ValueError(f"Unsupported coord_unit '{coord_unit}'")

    with open(filename, 'w') as f:
        # Comment lines
        if comment_line1 is None:
            comment_line1 = "Cube file written by tdcouplingmodel"
        if comment_line2 is None:
            comment_line2 = f"Coordinate units: {coord_unit}"

        f.write(f"{comment_line1}\n")
        f.write(f"{comment_line2}\n")

        # Header
        f.write(f"{natom:5d} {origin[0]:12.6f} {origin[1]:12.6f} {origin[2]:12.6f}\n")
        f.write(f"{NX_sign:5d} {dx_sign:12.6f}  0.000000  0.000000\n")
        f.write(f"{NY_sign:5d}  0.000000 {dy_sign:12.6f}  0.000000\n")
        f.write(f"{NZ_sign:5d}  0.000000  0.000000 {dz_sign:12.6f}\n")

        # Atoms
        for Z, q, x, y, z in zip(atom_Z, atom_q, atom_x, atom_y, atom_z):
            f.write(f"{int(Z):5d} {q:12.6f} {x:12.6f} {y:12.6f} {z:12.6f}\n")

        # Volumetric data
        density_flat = density_3d.ravel(order='C')
        values_per_line = 6
        for i in range(0, density_flat.size, values_per_line):
            chunk = density_flat[i:i + values_per_line]
            line = " ".join(f"{v:13.5e}" for v in chunk)
            f.write(line + "\n")


# def z_to_symbol(Z):
#     """
#     Convert atomic number to element symbol.

#     Parameters
#     ----------
#     Z : int
#         Atomic number.

#     Returns
#     -------
#     str
#         Element symbol (e.g., "H", "C", "O").
#         Returns "X{Z}" for unknown elements.
#     """
#     periodic = {
#         1: "H", 2: "He",
#         3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
#         11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar",
#         19: "K", 20: "Ca",
#         29: "Cu", 30: "Zn",
#         47: "Ag", 48: "Cd",
#         79: "Au", 80: "Hg",
#     }
#     return periodic.get(int(Z), f"X{int(Z)}")




def z_to_symbol(Z):
    """
    Convert atomic number to element symbol.

    Parameters
    ----------
    Z : int
        Atomic number.

    Returns
    -------
    str
        Element symbol (e.g., "H", "C", "O").
        Returns "X{Z}" for unknown elements.
    """
    try:
        element = periodictable.elements[int(Z)]
        return element.symbol
    except (IndexError, ValueError):
        return f"X{int(Z)}"
