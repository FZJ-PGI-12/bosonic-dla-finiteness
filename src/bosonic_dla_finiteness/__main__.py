import argparse
import sys

from bosonic_dla_finiteness.algebra.finiteness_check import (
    DimensionResult,
    check_finiteness,
)
from bosonic_dla_finiteness.algebra.free_hamiltonian import FreeHamiltonian
from bosonic_dla_finiteness.io.loader import load_from_yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether a bosonic DLA is finite-dimensional."
    )
    parser.add_argument(
        "config", help="Path to the YAML system configuration file."
    )
    args = parser.parse_args()

    try:
        config = load_from_yaml(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    generators = [g.to_generator() for g in config.generators]
    result = check_finiteness(
        n=config.n_modes,
        F=[FreeHamiltonian(config.omegas)],
        generators=generators,
    )

    print(f"Result: {result.dimension.value}")
    if result.dimension == DimensionResult.REMAINING:
        print(f"Remaining generators ({len(result.remaining_generators)}):")
        for gen in sorted(result.remaining_generators, key=lambda g: str(g)):
            print(f"  {gen}")


if __name__ == "__main__":
    main()
