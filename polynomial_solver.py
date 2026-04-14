#!/usr/bin/env python3
"""Interactive polynomial equation solver for degree up to 6 (7 coefficients)."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import List, Optional

EPS = 1e-9


def eval_poly(coeffs: List[float], x: float) -> float:
    """Evaluate polynomial using Horner's rule."""
    value = 0.0
    for coef in coeffs:
        value = value * x + coef
    return value


def derivative_coeffs(coeffs: List[float]) -> List[float]:
    """Build derivative polynomial coefficients."""
    degree = len(coeffs) - 1
    return [coeffs[i] * (degree - i) for i in range(len(coeffs) - 1)]


def linear_method(m: float, k: float) -> List[float]:
    """Solve m*x + k = 0."""
    if abs(m) < EPS:
        return []
    return [-k / m]


def vieta_method(a: float, b: float, c: float) -> List[float]:
    """Solve quadratic equation a*x^2 + b*x + c = 0."""
    if abs(a) < EPS:
        return linear_method(b, c)

    d = b * b - 4 * a * c
    if d < -EPS:
        return []
    if abs(d) <= EPS:
        return [-b / (2 * a)]

    sqrt_d = math.sqrt(max(d, 0.0))
    x1 = (-b + sqrt_d) / (2 * a)
    x2 = (-b - sqrt_d) / (2 * a)
    return [x1, x2]


def cbrt(value: float) -> float:
    """Real cubic root."""
    if value >= 0:
        return value ** (1.0 / 3.0)
    return -((-value) ** (1.0 / 3.0))


def cardano_method(a: float, b: float, c: float, d: float) -> List[float]:
    """Solve cubic equation a*x^3 + b*x^2 + c*x + d = 0 (real roots only)."""
    if abs(a) < EPS:
        return vieta_method(b, c, d)

    # Normalize: x^3 + A*x^2 + B*x + C = 0
    A = b / a
    B = c / a
    C = d / a

    # Depressed cubic t^3 + p*t + q = 0, x = t - A/3
    p = B - A * A / 3.0
    q = 2 * A**3 / 27.0 - A * B / 3.0 + C
    delta = (q / 2.0) ** 2 + (p / 3.0) ** 3

    roots: List[float] = []
    shift = -A / 3.0

    if delta > EPS:
        u = cbrt(-q / 2.0 + math.sqrt(delta))
        v = cbrt(-q / 2.0 - math.sqrt(delta))
        roots.append(u + v + shift)
    elif abs(delta) <= EPS:
        u = cbrt(-q / 2.0)
        roots.append(2 * u + shift)
        roots.append(-u + shift)
    else:
        r = math.sqrt(-(p**3) / 27.0)
        phi = math.acos(max(-1.0, min(1.0, -q / (2.0 * r))))
        t = 2.0 * math.sqrt(-p / 3.0)
        roots.extend(
            [
                t * math.cos(phi / 3.0) + shift,
                t * math.cos((phi + 2.0 * math.pi) / 3.0) + shift,
                t * math.cos((phi + 4.0 * math.pi) / 3.0) + shift,
            ]
        )

    return roots


def horner_method(coeffs: List[float], root: float) -> List[float]:
    """Divide polynomial by (x - root)."""
    if len(coeffs) <= 1:
        return coeffs
    new_coeffs = [coeffs[0]]
    for coef in coeffs[1:]:
        new_coeffs.append(new_coeffs[-1] * root + coef)
    # Last value is remainder.
    return new_coeffs[:-1]


def horner_with_remainder(coeffs: List[float], root: float) -> tuple[List[float], float]:
    """Divide polynomial by (x - root) and return (quotient, remainder)."""
    if len(coeffs) <= 1:
        return coeffs, 0.0
    new_coeffs = [coeffs[0]]
    for coef in coeffs[1:]:
        new_coeffs.append(new_coeffs[-1] * root + coef)
    return new_coeffs[:-1], new_coeffs[-1]


def stabilize_root(coeffs: List[float], root: float) -> Optional[tuple[float, List[float]]]:
    """
    Snap root to stable values (e.g., integers) when possible and verify deflation remainder.
    Returns (accepted_root, quotient) or None if root is not good enough.
    """
    candidates = [root]
    rounded = round(root)
    if abs(root - rounded) < 1e-3:
        candidates.append(float(rounded))
    if abs(root) < 1e-6:
        candidates.append(0.0)

    best_root: Optional[float] = None
    best_q: Optional[List[float]] = None
    best_rem = float("inf")
    scale = max(1.0, max(abs(c) for c in coeffs))

    for cand in candidates:
        quotient, remainder = horner_with_remainder(coeffs, cand)
        rem_abs = abs(remainder)
        if rem_abs < best_rem:
            best_root = cand
            best_q = quotient
            best_rem = rem_abs

    if best_root is None or best_q is None:
        return None
    if best_rem > 1e-5 * scale:
        return None
    return best_root, best_q


def newton_method(coeffs: List[float]) -> Optional[float]:
    """Find one real root with Newton method using multiple starts."""
    dcoeffs = derivative_coeffs(coeffs)
    for start in range(-10, 11):
        x = float(start)
        for _ in range(200):
            fx = eval_poly(coeffs, x)
            dfx = eval_poly(dcoeffs, x)
            if abs(dfx) < EPS:
                break
            xn = x - fx / dfx
            if abs(xn - x) < 1e-10 and abs(eval_poly(coeffs, xn)) < 1e-6:
                return xn
            x = xn
        if abs(eval_poly(coeffs, x)) < 1e-6:
            return x
    return None


def find_sign_change_interval(coeffs: List[float], left: float = -100, right: float = 100, step: float = 0.5) -> Optional[tuple[float, float]]:
    """Find [x1, x2] where polynomial changes sign."""
    x1 = left
    f1 = eval_poly(coeffs, x1)
    x = left + step
    while x <= right:
        f2 = eval_poly(coeffs, x)
        if abs(f1) < EPS:
            return (x1 - step, x1)
        if f1 * f2 <= 0:
            return (x1, x)
        x1, f1 = x, f2
        x += step
    return None


def chord_tangent_method(coeffs: List[float]) -> Optional[float]:
    """Combined chord/tangent method on auto-found sign-change interval."""
    interval = find_sign_change_interval(coeffs)
    if interval is None:
        return None

    a, b = interval
    fa = eval_poly(coeffs, a)
    fb = eval_poly(coeffs, b)
    dcoeffs = derivative_coeffs(coeffs)

    for _ in range(200):
        # Chord (secant) update for one side.
        if abs(fb - fa) < EPS:
            break
        c = (a * fb - b * fa) / (fb - fa)

        # Tangent (Newton) update from b side.
        dfb = eval_poly(dcoeffs, b)
        if abs(dfb) < EPS:
            bn = c
        else:
            bn = b - fb / dfb

        a, b = min(c, bn), max(c, bn)
        fa, fb = eval_poly(coeffs, a), eval_poly(coeffs, b)

        if abs(fa) < 1e-7:
            return a
        if abs(fb) < 1e-7:
            return b
        if abs(b - a) < 1e-10:
            mid = (a + b) / 2.0
            if abs(eval_poly(coeffs, mid)) < 1e-6:
                return mid
            break

    mid = (a + b) / 2.0
    if abs(eval_poly(coeffs, mid)) < 1e-5:
        return mid
    return None


def secant_method(coeffs: List[float]) -> Optional[float]:
    """Find one real root using secant method and several start pairs."""
    starts = [(-10.0, -9.0), (-5.0, -4.0), (-1.0, 1.0), (2.0, 3.0), (8.0, 9.0)]

    for x0, x1 in starts:
        f0 = eval_poly(coeffs, x0)
        f1 = eval_poly(coeffs, x1)
        for _ in range(250):
            if abs(f1 - f0) < EPS:
                break
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            if abs(x2 - x1) < 1e-10 and abs(eval_poly(coeffs, x2)) < 1e-6:
                return x2
            x0, x1 = x1, x2
            f0, f1 = f1, eval_poly(coeffs, x1)
        if abs(eval_poly(coeffs, x1)) < 1e-6:
            return x1

    return None


def integer_divisors(value: int) -> List[int]:
    value = abs(value)
    if value == 0:
        return [0]
    divisors = set()
    i = 1
    while i * i <= value:
        if value % i == 0:
            divisors.add(i)
            divisors.add(value // i)
        i += 1
    return sorted(divisors)


def rational_root_method(coeffs: List[float]) -> Optional[float]:
    """
    Try Rational Root Theorem when coefficients are integer-like.
    Returns one exact rational root if found.
    """
    int_coeffs: List[int] = []
    for c in coeffs:
        rc = round(c)
        if abs(c - rc) > 1e-9:
            return None
        int_coeffs.append(int(rc))

    leading = int_coeffs[0]
    constant = int_coeffs[-1]
    if leading == 0:
        return None
    if constant == 0:
        return 0.0

    p_values = integer_divisors(constant)
    q_values = integer_divisors(leading)

    tried = set()
    for p in p_values:
        for q in q_values:
            if q == 0:
                continue
            for sign in (-1, 1):
                frac = Fraction(sign * p, q)
                if frac in tried:
                    continue
                tried.add(frac)
                x = float(frac)
                if abs(eval_poly(coeffs, x)) < 1e-9:
                    return x
    return None


def round_root(root: float) -> float | int:
    nearest_int = round(root)
    if abs(root - nearest_int) <= 1e-2:
        return int(nearest_int)
    return round(root, 6)


def deduplicate(roots: List[float]) -> List[float]:
    unique: List[float] = []
    for r in sorted(roots):
        if not unique or abs(r - unique[-1]) >= 1e-4:
            unique.append(r)
    return unique


def solve_equation(coeffs: List[float]) -> List[float | int]:
    if all(abs(c) < EPS for c in coeffs):
        print("Any number is a solution")
        return []

    while len(coeffs) > 1 and abs(coeffs[0]) < EPS:
        coeffs.pop(0)

    if len(coeffs) == 1:
        if abs(coeffs[0]) < EPS:
            print("Any number")
        else:
            print("No solutions")
        return []

    result: List[float] = []

    while len(coeffs) > 1:
        degree = len(coeffs) - 1

        if abs(coeffs[-1]) < EPS:
            result.append(0.0)
            coeffs = coeffs[:-1]
            continue

        if degree == 1:
            result.extend(linear_method(coeffs[0], coeffs[1]))
            break

        if degree == 2:
            result.extend(vieta_method(coeffs[0], coeffs[1], coeffs[2]))
            break

        if degree == 3:
            result.extend(cardano_method(coeffs[0], coeffs[1], coeffs[2], coeffs[3]))
            break

        rational_root = rational_root_method(coeffs)
        if rational_root is not None:
            stabilized = stabilize_root(coeffs, rational_root)
            if stabilized is None:
                print("Could not stabilize a rational root for safe degree reduction")
                break
            accepted_root, coeffs = stabilized
            result.append(accepted_root)
            continue

        root = (
            newton_method(coeffs)
            or chord_tangent_method(coeffs)
            or secant_method(coeffs)
        )

        if root is None:
            print("Could not find a root numerically for the current polynomial")
            break

        stabilized = stabilize_root(coeffs, root)
        if stabilized is None:
            print("Could not stabilize a numeric root for safe degree reduction")
            break
        accepted_root, coeffs = stabilized
        result.append(accepted_root)

        while len(coeffs) > 1 and abs(coeffs[0]) < EPS:
            coeffs.pop(0)

    cleaned = deduplicate([float(round_root(r)) for r in result])
    return [round_root(r) for r in cleaned]


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


def main() -> None:
    while True:
        coeffs = input_coefficients()
        roots = solve_equation(coeffs)
        print(f"Roots: {roots}")

        answer = input("Continue? (no/something else) ").strip().lower()
        if answer == "no":
            break


if __name__ == "__main__":
    main()
