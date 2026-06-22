"""
Core coupling calculation functions.

This module implements the main algorithms for calculating
electromagnetic coupling between molecular and plasmonic excitations.
"""

from .coupling import (
    compute_g_fft_coulomb,
    build_hamiltonian,
    diagonalize_hamiltonian,
    analyze_polariton_character,
    set_verbose,
)

__all__ = [
    'compute_g_fft_coulomb',
    'build_hamiltonian',
    'diagonalize_hamiltonian',
    'analyze_polariton_character',
    'set_verbose',
]
