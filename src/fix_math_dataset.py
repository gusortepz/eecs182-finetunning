#!/usr/bin/env python3
"""
fix_math_dataset.py
-------------------

Utility script for regenerating correct answers for the simple math datasets.
It parses each question, solves the underlying algebraic equation, and writes
back an updated CSV with the corrected solution column.
"""

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fix_math_dataset")

# Common textual patterns that reveal which variable we should solve for
VARIABLE_PATTERNS: Sequence[re.Pattern] = [
    re.compile(r"solve\s+for\s+([a-z])", re.IGNORECASE),
    re.compile(r"find\s+([a-z])", re.IGNORECASE),
    re.compile(r"what\s+is\s+the\s+value\s+of\s+([a-z])", re.IGNORECASE),
    re.compile(r"value\s+of\s+([a-z])", re.IGNORECASE),
    re.compile(r"what\s+is\s+([a-z])", re.IGNORECASE),
    re.compile(r"for\s+([a-z])", re.IGNORECASE),
]

# Unicode superscript replacements so we can interpret the expressions
UNICODE_POW_REPLACEMENTS = {
    "²": "^2",
    "³": "^3",
    "⁴": "^4",
    "⁵": "^5",
    "⁶": "^6",
    "⁷": "^7",
    "⁸": "^8",
    "⁹": "^9",
}


@dataclass
class FixedRow:
    """Container for corrected dataset rows."""

    question: str
    solution: str


def replace_unicode_powers(expr: str) -> str:
    """Normalize unicode superscripts to Python power syntax."""
    for source, target in UNICODE_POW_REPLACEMENTS.items():
        expr = expr.replace(source, target)
    return expr

MATH_PREFIX_PATTERN = re.compile(r"[0-9a-z\+\-\*/\^²³⁴⁵⁶⁷⁸⁹\.\s]+")


def truncate_to_math(expr: str) -> str:
    """Trim the expression to the leading math portion (before extra text)."""
    match = MATH_PREFIX_PATTERN.match(expr)
    return match.group(0) if match else expr


def clean_expression(expr: str) -> str:
    """Trim punctuation and normalize power notation."""
    expr = expr.strip().rstrip(".,?")
    expr = truncate_to_math(expr)
    expr = replace_unicode_powers(expr)
    expr = re.sub(r"\s+", "", expr)  # remove whitespace for easier parsing
    return expr


def extract_equation(question: str, variable: Optional[str] = None) -> tuple[str, str]:
    """Extract the algebraic equation from the question text."""
    equals_index = question.find("=")
    if equals_index == -1:
        raise ValueError(f"No '=' symbol found in question: {question}")

    left_part = question[:equals_index]
    right_part = question[equals_index + 1 :]

    base_charset = r"0-9abxyz\+\-\*/\^²³⁴⁵⁶⁷⁸⁹\.\s"
    if variable:
        base_charset_with_var = base_charset + re.escape(variable)
    else:
        base_charset_with_var = base_charset

    left_digit_pattern = re.compile(
        rf"(?i)(?:^|[^a-z])([{base_charset}]*[0-9][{base_charset}]*)$"
    )
    right_digit_pattern = re.compile(
        rf"(?i)^([{base_charset}]*[0-9][{base_charset}]*)"
    )

    left_match = left_digit_pattern.search(left_part)
    right_match = right_digit_pattern.search(right_part)

    if variable:
        left_var_pattern = re.compile(
            rf"(?i)(?:^|[^a-z])([{base_charset_with_var}]*{re.escape(variable)}[{base_charset_with_var}]*)$"
        )
        right_var_pattern = re.compile(
            rf"(?i)^([{base_charset_with_var}]*{re.escape(variable)}[{base_charset_with_var}]*)"
        )
        if left_match is None:
            left_match = left_var_pattern.search(left_part)
        if right_match is None:
            right_match = right_var_pattern.search(right_part)

    if left_match is None or right_match is None:
        raise ValueError(f"Unable to isolate equation from question: {question}")

    left_expr = clean_expression(left_match.group(1))
    right_expr = clean_expression(right_match.group(1))

    return left_expr, right_expr


def detect_variable(question: str, equation_text: Optional[str] = None) -> str:
    """Detect which variable the problem wants solved for."""
    for pattern in VARIABLE_PATTERNS:
        match = pattern.search(question)
        if match:
            return match.group(1)

    # Fallback: inspect the equation itself to infer the symbol (letters in the expression)
    if equation_text:
        candidates = sorted({char for char in equation_text if char.isalpha()})
        if len(candidates) == 1:
            return candidates[0]

        adjacency_matches = re.findall(r"(?:(?:\d|\))\s*)([a-z])", equation_text)
        unique_adjacent = sorted(set(adjacency_matches))
        if len(unique_adjacent) == 1:
            return unique_adjacent[0]

    raise ValueError(f"Unable to infer variable for question: {question}")


def expression_to_coefficients(expr: str, variable: str) -> Dict[int, float]:
    """Convert a polynomial expression into a mapping of exponent -> coefficient."""
    if not expr:
        return {}

    # Ensure every term has an explicit sign to simplify splitting
    if expr[0] not in "+-":
        expr = f"+{expr}"

    terms = re.findall(r"[+-][^+-]+", expr)
    coefficients: Dict[int, float] = {}

    for term in terms:
        sign = 1.0 if term[0] == "+" else -1.0
        body = term[1:]
        if not body:
            continue

        if variable not in body:
            # Constant term
            coeff = float(body)
            exponent = 0
        else:
            var_index = body.index(variable)
            coeff_part = body[:var_index]
            if coeff_part in ("", "+"):
                coeff = 1.0
            elif coeff_part == "-":
                coeff = -1.0
            else:
                coeff = float(coeff_part)

            exponent = 1
            remaining = body[var_index + len(variable) :]
            if remaining.startswith("^"):
                exponent_part = remaining[1:]
                if not exponent_part:
                    raise ValueError(f"Missing exponent in term: {term}")
                exponent = int(exponent_part)

        coefficients[exponent] = coefficients.get(exponent, 0.0) + sign * coeff

    return {exp: coef for exp, coef in coefficients.items() if abs(coef) > 1e-9}


def combine_polynomials(
    left_coeffs: Dict[int, float],
    right_coeffs: Dict[int, float],
) -> Dict[int, float]:
    """Subtract right-hand coefficients from left-hand coefficients (left - right)."""
    combined = dict(left_coeffs)
    for exponent, coefficient in right_coeffs.items():
        combined[exponent] = combined.get(exponent, 0.0) - coefficient
    return {exp: coef for exp, coef in combined.items() if abs(coef) > 1e-9}


def coefficients_to_polynomial(coefficients: Dict[int, float]) -> List[float]:
    """Convert exponent -> coefficient dict into a list suitable for numpy.roots."""
    if not coefficients:
        raise ValueError("Polynomial has no coefficients after simplification.")

    max_degree = max(coefficients.keys())
    if max_degree == 0:
        raise ValueError("Equation does not contain the target variable.")

    poly = [coefficients.get(exp, 0.0) for exp in range(max_degree, -1, -1)]

    # Remove leading zeros if numeric issues left them in
    while len(poly) > 1 and abs(poly[0]) < 1e-9:
        poly.pop(0)

    return poly


def solve_polynomial(poly: Sequence[float]) -> List[float]:
    """Solve a polynomial and return sorted real roots."""
    roots = np.roots(poly)
    real_roots = [
        root.real for root in roots if isinstance(root, (float, np.floating)) or abs(root.imag) < 1e-8
    ]

    if not real_roots:
        raise ValueError("No real roots found.")

    # Deduplicate near-equal roots
    unique_roots: List[float] = []
    for value in real_roots:
        if not any(abs(value - existing) < 1e-6 for existing in unique_roots):
            unique_roots.append(value)

    return sorted(unique_roots)


def pick_canonical_solution(solutions: Sequence[float]) -> float:
    """Select the canonical solution (smallest real root)."""
    if not solutions:
        raise ValueError("No solutions available to select from.")
    return solutions[0]


def format_solution(value: float, precision: int = 6) -> str:
    """Format a floating-point solution as a clean numeric string."""
    if np.isclose(value, round(value), atol=1e-8):
        formatted = str(int(round(value)))
    else:
        formatted = f"{value:.{precision}f}".rstrip("0").rstrip(".")

    if formatted in {"-0", "-0.0"}:
        formatted = "0"

    return formatted


def fix_dataset(input_path: Path, output_path: Optional[Path] = None) -> None:
    """Load, correct, and write the dataset."""
    if output_path is None:
        output_path = input_path

    logger.info("Loading dataset from %s", input_path)
    df = pd.read_csv(input_path)

    fixed_rows: List[FixedRow] = []
    failures: List[str] = []

    for idx, row in df.iterrows():
        question = str(row["question"]).strip()

        try:
            try:
                variable_name = detect_variable(question)
            except ValueError:
                variable_name = None

            left, right = extract_equation(question, variable_name)
            equation_text = f"{left}={right}"

            if variable_name is None:
                try:
                    variable_name = detect_variable(question, equation_text)
                except ValueError:
                    letters = sorted({char for char in equation_text if char.isalpha()})
                    if len(letters) == 1:
                        variable_name = letters[0]
                    else:
                        raise

            left_coeffs = expression_to_coefficients(left, variable_name)
            right_coeffs = expression_to_coefficients(right, variable_name)
            combined_coeffs = combine_polynomials(left_coeffs, right_coeffs)
            poly = coefficients_to_polynomial(combined_coeffs)
            solutions = solve_polynomial(poly)
            canonical_solution = pick_canonical_solution(solutions)
            formatted_solution = format_solution(canonical_solution)
            fixed_rows.append(FixedRow(question=question, solution=formatted_solution))
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to process row %s: %s", idx, exc)
            failures.append(question)

    if failures:
        raise RuntimeError(
            f"Failed to correct {len(failures)} rows. "
            "Fix the issues and re-run the script."
        )

    logger.info("All %s rows processed successfully", len(fixed_rows))

    output_df = pd.DataFrame([row.__dict__ for row in fixed_rows])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    logger.info("Corrected dataset saved to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate correct answers for the simple math dataset."
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        required=True,
        help="Path to the existing CSV file with potentially incorrect solutions.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Path to write the corrected CSV (defaults to overwriting the input).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fix_dataset(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()

