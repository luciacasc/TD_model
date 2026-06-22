
"""
Command-line interface for cube rotation and translation.
"""

import argparse
import sys
from ..utils.roto_traslation import rotate_translate_cube, rotation_matrix_from_dipoles, rotation_matrix_from_euler
import numpy as np


def main():
    """Main entry point for cube rotation/translation command line tool."""

    parser = argparse.ArgumentParser(
        description=(
            "Rigid-body rotate/translate a cube (atoms + density) and resample on a new grid.\n"
            "Coordinate units are auto-detected from the grid vectors sign:\n"
            "  positive voxel lengths → Bohr (default), negative voxel lengths → Angstroms."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("cube_in", help="Input cube file")
    parser.add_argument("cube_out", help="Output cube file")

    parser.add_argument(
        "--angles",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("ALPHA", "BETA", "GAMMA"),
        help="Euler angles in degrees (order ZYX). Default: 0 0 0."
    )
    parser.add_argument(
        "--trans",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("TX", "TY", "TZ"),
        help="Translation vector components (in input_unit; default: 0 0 0)."
    )

    parser.add_argument(
        "--dipole_1",
        nargs=3,
        type=float,
        default=None,
        metavar=("D1X", "D1Y", "D1Z"),
        help="Target plasmonic dipole vector mu_1 (in Debye, only relative direction matters)."
    )

    parser.add_argument(
        "--dipole_2",
        nargs=3,
        type=float,
        default=None,
        metavar=("D2X", "D2Y", "D2Z"),
        help="Source molecular dipole vector mu_2 (in Debye, to be rotated onto mu_1)."
    )



    parser.add_argument(
        "--center",
        nargs=3,
        type=float,
        default=None,
        metavar=("CX", "CY", "CZ"),
        help="Rotation center (in input_unit). If omitted, the center of the original grid is used."
    )
    parser.add_argument(
        "--input_unit",
        choices=["bohr", "ang"],
        default="ang",
        help="Unit of TX,TY,TZ and center provided on the command line (default: ang)."
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output messages"
    )

    args = parser.parse_args()


    try:
        rotate_translate_cube(
            cube_in=args.cube_in,
            cube_out=args.cube_out,
            angles=tuple(args.angles),
            trans=tuple(args.trans),
            center=tuple(args.center) if args.center is not None else None,
            input_unit=args.input_unit,
            mu_1=args.dipole_1,
            mu_2=args.dipole_2,
            verbose=not args.quiet,
        )
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
