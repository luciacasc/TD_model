"""
TD Coupling Model - Transition Density Coupling Calculations

A Python package for calculating electromagnetic coupling between molecular
transition densities and plasmonic excitations using FFT-based methods.

Main modules:
- io: Input/output utilities for cube files
- utils: Utility functions (conversion, rotation, summation)
- preprocessing: Grid alignment and preprocessing tools
- core: Core coupling calculations
- cli: Command-line interfaces

Author: Lucia Cascino
License: GNU GENERAL PUBLIC LICENSE
"""

__version__ = "0.1.0"

# Import main functions for easy access
from .io import parse_cube_file, write_cube_file, density_3d_from_flat
from .utils import cube_to_xyz
from .utils.constants import BOHR_TO_ANG, ANG_TO_BOHR, E_CHARGE
from .core.coupling import compute_g_fft_coulomb, build_hamiltonian, diagonalize_hamiltonian
from .preprocessing.align_grids import align_cube_grids
from .utils.roto_traslation import rotate_translate_cube
from .utils.sum_cubes import sum_cube_files

__all__ = [
    # Version
    '__version__',
    
    # IO functions
    'parse_cube_file',
    'write_cube_file',
    'density_3d_from_flat',
    
    # Utilities
    'cube_to_xyz',
    'rotate_translate_cube',
    'sum_cube_files',
    
    # Constants
    'BOHR_TO_ANG',
    'ANG_TO_BOHR',
    'E_CHARGE',
    
    # Preprocessing
    'align_cube_grids',
    
    # Core calculations
    'compute_g_fft_coulomb',
    'build_hamiltonian',
    'diagonalize_hamiltonian',
]
