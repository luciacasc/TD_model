"""
Command-line interface for cube to XYZ conversion.
"""

import argparse
import sys
from ..utils.conversion_cube_to_xyz import cube_to_xyz



def main():
    """Main entry point for cube-to-xyz command line tool."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Gaussian cube file to XYZ format.\n"
            "Coordinate units are auto-detected from the grid vectors sign:\n"
            "  positive voxel lengths → Bohr (default)\n"
            "  negative voxel lengths → Angstroms\n"
            "Output XYZ is always in Angstrom."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "cube_in",
        help="Input cube file"
    )

    parser.add_argument(
        "xyz_out",
        help="Output XYZ file"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output messages"
    )

    args = parser.parse_args()

    try:
        cube_to_xyz(args.cube_in, args.xyz_out, verbose=not args.quiet)
        if not args.quiet:
            print("Conversion completed successfully!")
        return 0
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
