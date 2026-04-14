#!/usr/bin/env python3
"""Main execution block for the polynomial solver."""

from __future__ import annotations

from dialogue import ask_continue, input_coefficients, show_result
from solver_functions import solve_equation


def main() -> None:
    while True:
        coeffs = input_coefficients()
        message, roots = solve_equation(coeffs)
        show_result(message, roots)
        if not ask_continue():
            break


if __name__ == "__main__":
    main()
