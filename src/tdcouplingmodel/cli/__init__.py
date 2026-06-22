"""
Command-line interface modules.

This module contains all CLI entry points for the package.
Each CLI module can be run as:
  python -m tdcouplingmodel.cli.module_name

Or through the installed console scripts:
  tdcoupling-[command]
"""

# CLI modules are not imported here to avoid loading argparse unnecessarily
# They are accessed directly by entry points

__all__ = [
    'conversion_cube_to_xyz_cli',
    'roto_translation_cli',
    'align_grids_cli',
    'coupling_cli',
    'sum_cubes_cli',
]

