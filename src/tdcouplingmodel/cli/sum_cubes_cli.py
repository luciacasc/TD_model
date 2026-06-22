"""
Command-line interface for summing cube files.
"""

import argparse
import sys
from ..utils.sum_cubes import sum_cube_files


def main():
    """Main entry point for cube summation command line tool."""
    parser = argparse.ArgumentParser(
        description="Sum two cube files with optional scaling factors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple sum
  tdcoupling-sum file1.cube file2.cube output.cube

  # Sum with scaling factors
  tdcoupling-sum file1.cube file2.cube output.cube -f1 0.8 -f2 0.6

  # Sum transition densities (often need negative factors)
  tdcoupling-sum mol.cube plas.cube hybrid.cube -f1 -0.8 -f2 -0.58
"""
    )

    parser.add_argument(
        "file1",
        help="First input cube file"
    )

    parser.add_argument(
        "file2",
        help="Second input cube file"
    )

    parser.add_argument(
        "output",
        help="Output cube file"
    )

    parser.add_argument(
        "-f1", "--factor1",
        type=float,
        default=1.0,
        help="Scaling factor for file1 (default: 1.0)"
    )

    parser.add_argument(
        "-f2", "--factor2",
        type=float,
        default=1.0,
        help="Scaling factor for file2 (default: 1.0)"
    )

    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip output verification (faster)"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output messages"
    )

    args = parser.parse_args()

    try:
        stats = sum_cube_files(
            file1=args.file1,
            file2=args.file2,
            output_file=args.output,
            factor1=args.factor1,
            factor2=args.factor2,
            verify=not args.no_verify,
            verbose=not args.quiet
        )

        if not args.quiet:
            print(f"\n{'='*60}")
            print("SUMMARY")
            print('='*60)
            print(f"Output: {args.output}")
            print(f"Total atoms: {stats['natom_total']}")
            print(f"Density range: [{stats['density_min']:.6e}, {stats['density_max']:.6e}]")
            if 'verified' in stats:
                print(f"Verification: {'PASSED' if stats['verified'] else 'FAILED'}")

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
