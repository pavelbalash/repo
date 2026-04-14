#!/usr/bin/env python3
"""Dialogue/input-output helpers for polynomial solver CLI."""

from __future__ import annotations

from typing import List, Optional


def input_coefficients() -> List[float]:
    while True:
        raw = input("Enter 7 coefficients a b c d n m k: ").strip()
        parts = raw.split()
        if len(parts) != 7:
            print("7 coefficients are required!")
            continue
        try:
            return [float(item) for item in parts]
        except ValueError:
            print("coefficients must be numbers")


def show_result(message: Optional[str], roots: Optional[List[float | int]]) -> None:
    if message:
        print(message)
    if roots is not None:
        print(f"Roots: {roots}")


def ask_continue() -> bool:
    answer = input("Continue? (no/something else) ").strip().lower()
    return answer != "no"
