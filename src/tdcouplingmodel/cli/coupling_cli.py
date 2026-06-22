"""
Command-line interface for molecule-plasmon coupling calculations.
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from ..core.coupling import (
    compute_g_fft_coulomb,
    build_hamiltonian,
    diagonalize_hamiltonian,
    analyze_polariton_character,
    set_verbose,
)
from ..io.cube_io import parse_cube_file, density_3d_from_flat


def main():
    """Main entry point for coupling calculation command line tool."""
    parser = argparse.ArgumentParser(
        description="Calculate molecule-plasmon polariton coupling and energies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  tdcoupling-calc --mol-cubes mol1_trans.cube mol2_trans.cube \
                  --mol-energies 2.5 2.6 \
                  --plas-cubes plas1.cube plas2.cube \
                  --plas-energies 2.4 2.5 \
                  --output results.txt
"""
    )

    # Molecular inputs
    parser.add_argument(
        "--mol-cubes",
        nargs="+",
        required=True,
        help="Molecular transition density cube files"
    )

    parser.add_argument(
        "--mol-energies",
        nargs="+",
        type=float,
        required=True,
        help="Molecular excitation energies (eV)"
    )

    # Plasmon inputs
    parser.add_argument(
        "--plas-cubes",
        nargs="+",
        required=True,
        help="Plasmon transition density cube files"
    )

    parser.add_argument(
        "--plas-energies",
        nargs="+",
        type=float,
        required=True,
        help="Plasmon excitation energies (eV)"
    )

    parser.add_argument(
        "--energy-units",
        choices=["ev", "nm"],
        default="ev",
        help="Units for energy values: 'ev' (energies) or 'nm' (wavelengths). Default: ev"
    )
    

    # Output
    parser.add_argument(
        "--output",
        "--out-prefix",
        default="coupling_results",
        help="Output file prefix (default: coupling_results)"
    )

    # Coupling calculation options
    parser.add_argument(
        "--eps-eff",
        type=float,
        default=1.0,
        help="Effective dielectric constant (default: 1.0)"
    )

    parser.add_argument(
        "--pad",
        type=float,
        default=1.0,
        help="FFT padding factor (default: 1.0)"
    )

    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Extra FFT margin in bohr/angstrom (default: 0.0)"
    )

    parser.add_argument(
        "--use-crop",
        action="store_true",
        help="Enable density cropping to bounding box"
    )

    parser.add_argument(
        "--crop-threshold",
        type=float,
        default=1e-6,
        help="Density threshold for cropping (default: 1e-6)"
    )

    parser.add_argument(
        "--crop-margin",
        type=float,
        default=2.0,
        help="Physical margin for cropping in bohr/Å (default: 2.0)"
    )

    # Disable specific couplings
    parser.add_argument(
        "--disable-mol-mol",
        action="store_true",
        help="Disable molecule-molecule couplings (V = 0)"
    )

    parser.add_argument(
        "--disable-plas-plas",
        action="store_true",
        help="Disable plasmon-plasmon couplings (J = 0)"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )

    args = parser.parse_args()

    # Set verbose mode
    set_verbose(not args.quiet)

    # Validation
    N_m = len(args.mol_cubes)
    N_p = len(args.plas_cubes)

    if len(args.mol_energies) != N_m:
        parser.error(f"Number of molecular energies ({len(args.mol_energies)}) != "
                     f"number of molecular cubes ({N_m})")

    if len(args.plas_energies) != N_p:
        parser.error(f"Number of plasmon energies ({len(args.plas_energies)}) != "
                     f"number of plasmon cubes ({N_p})")

    
    try:
        if not args.quiet:
            print("="*80)
            print("MOLECULE-PLASMON POLARITON CALCULATION")
            print("="*80)
            print(f"Molecules: {N_m}")
            print(f"Plasmons: {N_p}")
            print(f"Total states: {N_m + N_p}")
            print()

        # Load molecular densities
        if not args.quiet:
            print("Loading molecular transition densities...")

        mol_densities = []
        mol_headers = []
        for i, fname in enumerate(args.mol_cubes):
            if not args.quiet:
                print(f"  [{i+1}/{N_m}] {fname}")
            header, density_flat = parse_cube_file(fname)
            density_3d = density_3d_from_flat(density_flat, header["grid_size"])
            mol_densities.append(density_3d)
            mol_headers.append(header)

        # Load plasmon densities
        if not args.quiet:
            print("\nLoading plasmon transition densities...")

        plas_densities = []
        plas_headers = []
        for i, fname in enumerate(args.plas_cubes):
            if not args.quiet:
                print(f"  [{i+1}/{N_p}] {fname}")
            header, density_flat = parse_cube_file(fname)
            density_3d = density_3d_from_flat(density_flat, header["grid_size"])
            plas_densities.append(density_3d)
            plas_headers.append(header)

        # Get reference grid info (from first plasmon)
        ref_header = plas_headers[0]
        steps_ref = ref_header["steps"]
        units_len = ref_header.get("coord_unit", "bohr")

        # Calculate molecule-plasmon couplings
        if not args.quiet:
            print("\n" + "="*80)
            print("CALCULATING MOLECULE-PLASMON COUPLINGS")
            print("="*80)

        g_matrix = np.zeros((N_m, N_p), dtype=float)

        for m in range(N_m):
            for p in range(N_p):
                if not args.quiet:
                    print(f"\nCalculating g(mol{m+1}, plas{p+1})...")

                g_mp = compute_g_fft_coulomb(
                    rho_A=mol_densities[m],
                    rho_B=plas_densities[p],
                    steps=steps_ref,
                    units_len=units_len,
                    eps_eff=args.eps_eff,
                    pad=args.pad,
                    fft_margin_bohr=args.margin,
                    use_crop=args.use_crop,
                    crop_threshold_bohr3=args.crop_threshold,
                    crop_margin_bohr=args.crop_margin
                )

                g_matrix[m, p] = g_mp

                if not args.quiet:
                    print(f"  g(mol{m+1}, plas{p+1}) = {g_mp:.6f} eV")

        # Calculate molecule-molecule couplings
        if not args.quiet:
            print("\n" + "="*80)
            print("CALCULATING MOLECULE-MOLECULE COUPLINGS")
            print("="*80)

        V_matrix = np.zeros((N_m, N_m), dtype=float)

        if not args.disable_mol_mol:
            for i in range(N_m):
                for j in range(i + 1, N_m):
                    if not args.quiet:
                        print(f"\nCalculating V(mol{i+1}, mol{j+1})...")

                    V_ij = compute_g_fft_coulomb(
                        rho_A=mol_densities[i],
                        rho_B=mol_densities[j],
                        steps=steps_ref,
                        units_len=units_len,
                        eps_eff=args.eps_eff,
                        pad=args.pad,
                        fft_margin_bohr=args.margin,
                        use_crop=args.use_crop,
                        crop_threshold_bohr3=args.crop_threshold,
                        crop_margin_bohr=args.crop_margin
                    )

                    V_matrix[i, j] = V_ij
                    V_matrix[j, i] = V_ij

                    if not args.quiet:
                        print(f"  V(mol{i+1}, mol{j+1}) = {V_ij:.6f} eV")
        else:
            if not args.quiet:
                print("  Molecule-molecule couplings disabled (V = 0)")

        # Calculate plasmon-plasmon couplings
        if not args.quiet:
            print("\n" + "="*80)
            print("CALCULATING PLASMON-PLASMON COUPLINGS")
            print("="*80)

        J_matrix = np.zeros((N_p, N_p), dtype=float)

        if not args.disable_plas_plas:
            for i in range(N_p):
                for j in range(i + 1, N_p):
                    if not args.quiet:
                        print(f"\nCalculating J(plas{i+1}, plas{j+1})...")

                    J_ij = compute_g_fft_coulomb(
                        rho_A=plas_densities[i],
                        rho_B=plas_densities[j],
                        steps=plas_headers[i]["steps"],
                        units_len=plas_headers[i].get("coord_unit", "bohr"),
                        eps_eff=args.eps_eff,
                        pad=args.pad,
                        fft_margin_bohr=args.margin,
                        use_crop=args.use_crop,
                        crop_threshold_bohr3=args.crop_threshold,
                        crop_margin_bohr=args.crop_margin
                    )

                    J_matrix[i, j] = J_ij
                    J_matrix[j, i] = J_ij

                    if not args.quiet:
                        print(f"  J(plas{i+1}, plas{j+1}) = {J_ij:.6f} eV")
        else:
            if not args.quiet:
                print("  Plasmon-plasmon couplings disabled (J = 0)")

        # Build and diagonalize Hamiltonian
        if not args.quiet:
            print("\n" + "="*80)
            print("BUILDING HAMILTONIAN")
            print("="*80)


        # Convert omegas to eV
        if args.energy_units == "ev":
            omega_m = np.array(args.mol_energies, dtype=float)
            omega_p = np.array(args.plas_energies, dtype=float)
        else:  # "nm"
            omega_m = np.array([1239.8 / lam for lam in args.mol_energies], dtype=float)
            omega_p = np.array([1239.8 / lam for lam in args.plas_energies], dtype=float)


        H = build_hamiltonian(omega_m, omega_p, g_matrix, V_matrix, J_matrix)

        if not args.quiet:
            print(f"Hamiltonian dimension: {H.shape}")
            print(f"Diagonal energies (eV): {H.diagonal()}")

        # Diagonalize
        if not args.quiet:
            print("\nDiagonalizing...")

        energies, states = diagonalize_hamiltonian(H)
        energies_nm = 1239.8 / energies  # eV to nm

        # Analyze character
        character_analysis = analyze_polariton_character(states, N_m, N_p)

        # Print results
        if not args.quiet:
            print("\n" + "="*80)
            print(f"POLARITON ENERGIES ({N_m} molecules + {N_p} plasmons)")
            print("="*80)
            for idx, (e, e_nm) in enumerate(zip(energies, energies_nm)):
                print(f"State {idx+1:2d}: {e:10.6f} eV ({e_nm:7.1f} nm)")

            print("\n" + "="*80)
            print("CHARACTER ANALYSIS")
            print("="*80)
            labels = [f"Mol{i+1}" for i in range(N_m)] + [f"Plas{i+1}" for i in range(N_p)]

            for idx in range(len(energies)):
                analysis = character_analysis[idx]
                print(f"\nState {idx+1:2d} (E = {energies[idx]:.6f} eV, {energies_nm[idx]:.1f} nm):")
                for label, w in zip(labels, analysis['weights']):
                    perc = 100 * w
                    bar = "█" * int(perc / 5)
                    print(f"  {label:<10}: {perc:6.1f}% {bar}")
                print(f"\nEigvec: {analysis['eigvec']}")
                
        # Save results to file
        output_file = f"{args.output}_results.txt" if not args.output.endswith('.txt') else args.output

        with open(output_file, 'w', encoding="utf-8") as f:
            f.write("# Molecule-Plasmon Polariton Calculation Results\n")
            f.write(f"# N_molecules = {N_m}\n")
            f.write(f"# N_plasmons = {N_p}\n")
            f.write("# All energies in eV unless stated otherwise\n\n")

            # Input energies
            f.write("## Input Energies (eV)\n")
            for m in range(N_m):
                f.write(f"Molecule_{m+1}: {omega_m[m]:.8f}\n")
            for p in range(N_p):
                f.write(f"Plasmon_{p+1}: {omega_p[p]:.8f}\n")
            f.write("\n")

            # Coupling matrices
            f.write("## Molecule-Plasmon Couplings g[m,p] (eV)\n")
            for m in range(N_m):
                row = " ".join(f"{g_matrix[m, p]: .8e}" for p in range(N_p))
                f.write(f"g_mol{m+1}: {row}\n")
            f.write("\n")

            f.write("## Molecule-Molecule Couplings V[i,j] (eV)\n")
            for i in range(N_m):
                row = " ".join(f"{V_matrix[i, j]: .8e}" for j in range(N_m))
                f.write(f"V_mol{i+1}: {row}\n")
            f.write("\n")

            f.write("## Plasmon-Plasmon Couplings J[i,j] (eV)\n")
            for i in range(N_p):
                row = " ".join(f"{J_matrix[i, j]: .8e}" for j in range(N_p))
                f.write(f"J_plas{i+1}: {row}\n")
            f.write("\n")

            # Polariton energies
            f.write("## Polariton Energies\n")
            f.write("# State    E(eV)        λ(nm)\n")
            for idx, (e, e_nm) in enumerate(zip(energies, energies_nm)):
                f.write(f"{idx+1:5d}  {e:10.6f}  {e_nm:10.2f}\n")
            f.write("\n")

            # Hopfield coefficients
            f.write("## Hopfield Coefficients |C_i|²\n")
            header_line = f"{'State':<6} | {'E(eV)':<9} | " + " | ".join([f"{lbl:<8}" for lbl in labels])
            f.write(header_line + "\n")
            f.write("-" * len(header_line) + "\n")
            for idx in range(len(energies)):
                w = character_analysis[idx]['weights']
                row = f"{idx+1:<6} | {energies[idx]:<9.6f} | " + " | ".join(f"{x:<8.4f}" for x in w)
                f.write(row + "\n")

            # Eigenvectors (loop separato)
            f.write("\n## Eigenvectors (raw coefficients C_i)\n")
            f.write("# Each row = one polariton state; columns = [Mol1, ..., MolN, Plas1, ..., PlasN]\n")
            header_eig = f"{'State':<6} | {'E(eV)':<9} | " + " | ".join([f"{lbl:<10}" for lbl in labels])
            f.write(header_eig + "\n")
            f.write("-" * len(header_eig) + "\n")
            for idx in range(len(energies)):
                eigvec = character_analysis[idx]['eigvec']
                row = f"{idx+1:<6} | {energies[idx]:<9.6f} | " + " | ".join(f"{x:<+10.6f}" for x in eigvec)
                f.write(row + "\n")


        if not args.quiet:
            print(f"\nResults saved to: {output_file}")

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
