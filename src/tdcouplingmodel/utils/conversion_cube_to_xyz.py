"""
Conversion utilities for cube files.

This module provides functions to convert Gaussian cube files to XYZ format,
with automatic detection of coordinate units from grid vector signs.
"""

import numpy as np
from ..io.cube_io import parse_cube_file, z_to_symbol
from .constants import BOHR_TO_ANG




def write_xyz(filename, atom_Z, coords_ang, comment="generated from cube"):
    """
    Write a simple XYZ file.

    Parameters
    ----------
    filename : str
        Output XYZ filename.
    atom_Z : list or array
        Atomic numbers.
    coords_ang : array-like, shape (natom, 3)
        Atomic coordinates in Angstrom.
    comment : str, optional
        Comment line for XYZ file.
    """
    natom = len(atom_Z)
    with open(filename, 'w') as f:
        f.write(f"{natom}\n")
        f.write(f"{comment}\n")
        for Z, (x, y, z) in zip(atom_Z, coords_ang):
            sym = z_to_symbol(Z)
            f.write(f"{sym:2s}  {x:15.8f} {y:15.8f} {z:15.8f}\n")


def cube_to_xyz(cube_in, xyz_out, verbose = True):
    """
    Convert a Gaussian cube file to XYZ format.

    Automatically detects coordinate units (Bohr or Angstrom) from the cube file
    and writes atomic coordinates in Angstrom to the XYZ file.

    Parameters
    ----------
    cube_in : str
        Input cube file path.
    xyz_out : str
        Output XYZ file path.
    verbose : bool, optional
        Print conversion information (default: True).

    Returns
    -------
    dict
        Header information from the cube file.
    """

    header,_ = parse_cube_file(cube_in)
    coord_unit = header["coord_unit"]
    atom_Z, atom_q, ax, ay, az = header["atoms"]
    coords = np.vstack((ax, ay, az)).T  # internal units

    if coord_unit == "bohr":
        coords_ang = coords * BOHR_TO_ANG
    elif coord_unit == "ang":
        coords_ang = coords  # already in Angstrom
    else:
        raise ValueError(f"Unsupported coord_unit '{coord_unit}'")


    print(f"Input cube: {cube_in}")
    print(f"Detected coordinate unit: {coord_unit}")
    print(f"Writing XYZ (Angstrom) to: {xyz_out}")

    write_xyz(xyz_out, atom_Z, coords_ang, comment=f"from {cube_in} (units: {coord_unit})")


