from __future__ import annotations

import builtins

import pytest

from dialogue import input_coefficients
from solver_functions import solve_equation


@pytest.mark.parametrize(
    "coeffs, expected",
    [
        ([0, 0, 0, 0, 0, 0, 0], ("Any number is a solution", None)),
        ([0, 0, 0, 0, 0, 0, 5], ("No solutions", None)),
        ([0, 1, 0, 0, 0, 0, -1], (None, [1])),
        ([0, 0, 1, 0, 0, 0, -1], (None, [-1, 1])),
        ([0, 0, 0, 1, 0, -1, 0], (None, [-1, 0, 1])),
        ([1, 2, 3, 4, 5, 6, 7], (None, [])),
        ([1, 0, 0, 0, 0, 0, 0], (None, [0])),
        ([1, 0, 0, 0, 0, 0, 1], (None, [])),
        ([1, -6, 15, -20, 15, -6, 1], (None, [1])),
        ([1, 3, -3, -11, 6, 12, -8], (None, [-2, 1])),
        ([1, 0, 0, -2, 0, 0, 1], (None, [1])),
        ([1, -1, 0, 0, 0, 0, 0], (None, [0, 1])),
        ([1, 0, 0, 0, -1, 0, 0], (None, [-1, 0, 1])),
        ([1, 0, 0, 0, 0, 0, -64], (None, [-2, 2])),
        ([1, 1, 1, 1, 1, 1, 1], (None, [])),
        ([1, -1, -1, 1, 1, -1, -1], (None, [-0.754878, 1.324718])),
    ],
)
def test_solve_equation_cases(coeffs, expected):
    assert solve_equation(coeffs[:]) == expected


def test_input_coefficients_retries_until_valid(monkeypatch, capsys):
    user_inputs = iter(
        [
            "",  # empty string -> wrong count
            "1 2 3 4 5 6",  # too few
            "1 2 3 4 5 6 7 8",  # too many
            "1 2 a 4 5 6 7",  # non numeric
            "1 2 3 4 5 6 7",  # valid
        ]
    )

    monkeypatch.setattr(builtins, "input", lambda _: next(user_inputs))

    coeffs = input_coefficients()

    assert coeffs == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    output = capsys.readouterr().out

    assert output.count("7 coefficients are required!") == 3
    assert "coefficients must be numbers" in output
