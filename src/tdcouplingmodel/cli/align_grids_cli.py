"""
Command-line interface for cube grid alignment.
"""

import argparse
import sys
from ..preprocessing.align_grids import align_cube_grids


def main():
    """Main entry point for cube grid alignment command line tool."""
    parser = argparse.ArgumentParser(
        description="Align cube file grids by creating a common grid and resampling all files onto it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Align 3 files with minimum spacing
  tdcoupling-align -i f1.cube f2.cube f3.cube -o out1.cube out2.cube out3.cube --spacing min

  # Use spacing of 0.2 bohr/angstrom
  tdcoupling-align -i f1.cube f2.cube -o o1.cube o2.cube --spacing 0.2

  # Verify charge conservation
  tdcoupling-align -i f1.cube f2.cube -o o1.cube o2.cube --verify
"""
    )

    parser.add_argument(
        "-i", "--input",
        nargs="+",
        required=True,
        help="Input cube files (at least 1)"
    )

    parser.add_argument(
        "-o", "--output",
        nargs="+",
        required=True,
        help="Output cube files (same number as input)"
    )

    parser.add_argument(
        "--spacing",
        default='min',
        help="Spacing of common grid: 'min', 'max', 'mean', or numeric value (default: min)"
    )

    parser.add_argument(
        "--order",
        type=int,
        default=3,
        choices=[0, 1, 2, 3, 4, 5],
        help="Spline interpolation order (0-5, default: 3)"
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify charge conservation for each file"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output messages"
    )

    args = parser.parse_args()

    # Validation
    if len(args.input) != len(args.output):
        parser.error(
            f"Number of input files ({len(args.input)}) != "
            f"number of output files ({len(args.output)})"
        )

    try:
        result = align_cube_grids(
            input_files=args.input,
            output_files=args.output,
            spacing=args.spacing,
            order=args.order,
            verify=args.verify,
            verbose=not args.quiet
        )

        if not args.quiet:
            print(f"\nAlignment completed successfully!")
            if args.verify and result['verification_results']:
                print("\nCharge conservation summary:")
                for i, ver in enumerate(result['verification_results']):
                    print(f"  File {i+1}: {ver['charge_error_pct']:.4f}% error")

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
