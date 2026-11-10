"""CNPJ validation for estabelecimentos: pure check-digit functions plus a
vectorized chunk splitter.

The estabelecimentos triple (cnpj_basico, cnpj_ordem, cnpj_dv) is validated
as one 14-digit CNPJ: the three columns are concatenated, punctuation is
stripped, and the result is normalized before the mod-11 check digits are
verified. socios.cnpj_cpf_do_socio is out of scope — it may be a CPF and
must never go through this logic.

Reject reason codes:
    dv_mismatch     — 14 digits (or padded 13) whose check digits don't match
    bad_length      — wrong length, or non-digit content after punctuation strip
    repeated_digits — all 14 digits identical (passes mod 11 but is not real)
    zero_padded     — 13 digits recovered by left-padding zero; the row is
                      loaded with the padded value but the original is still
                      written to the sidecar for audit
"""

import polars as pl

# Mod-11 check-digit weights: the first DV cycles 5..2,9..2 over the 12-digit
# base, the second cycles 6..2,9..2 over base + first DV. Remainder < 2 → 0.
WEIGHTS_DV1 = (5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2)
WEIGHTS_DV2 = (6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2)

REASON_DV_MISMATCH = "dv_mismatch"
REASON_BAD_LENGTH = "bad_length"
REASON_REPEATED_DIGITS = "repeated_digits"
REASON_ZERO_PADDED = "zero_padded"

CNPJ_COLUMNS = ("cnpj_basico", "cnpj_ordem", "cnpj_dv")

REJECT_COLUMNS = ("chunk_id", "row_number", "cnpj_basico", "cnpj_ordem", "cnpj_dv", "reason")

# Punctuation seen in dirty extracts: thousand separators, the CNPJ slash and
# dash, and stray whitespace. Stripped before classification.
_PUNCTUATION_PATTERN = r"[./\-\s]"
_STRIP_TABLE = {ord(c): None for c in "./- \t\r\n"}

# Helper column names for the vectorized path; prefixed so they can't collide
# with real estabelecimentos columns.
_ROW_COL = "__cnpj_row"
_JOINED_COL = "__cnpj_joined"
_CLEANED_COL = "__cnpj_cleaned"
_LEN_COL = "__cnpj_len"
_DIGITS_COL = "__cnpj_digits"
_NORM_COL = "__cnpj_normalized"
_REPEATED_COL = "__cnpj_repeated"
_DV_OK_COL = "__cnpj_dv_ok"
_REASON_COL = "__cnpj_reason"
_HELPER_COLS = (
    _ROW_COL,
    _JOINED_COL,
    _CLEANED_COL,
    _LEN_COL,
    _DIGITS_COL,
    _NORM_COL,
    _REPEATED_COL,
    _DV_OK_COL,
    _REASON_COL,
)


def compute_check_digit(digits: str, weights: tuple[int, ...]) -> int:
    """Mod-11 check digit: weighted sum, remainder < 2 maps to 0."""
    total = sum(int(d) * w for d, w in zip(digits, weights, strict=True))
    remainder = total % 11
    return 0 if remainder < 2 else 11 - remainder


def dv_matches(digits: str) -> bool:
    """Both mod-11 check digits must match a 14-digit all-numeric CNPJ.

    Note: all-same-digit strings (e.g. 00000000000000) pass this check;
    callers must apply the repeated_digits rule separately.
    """
    if len(digits) != 14 or not digits.isdigit():
        return False
    return (
        int(digits[12]) == compute_check_digit(digits[:12], WEIGHTS_DV1)
        and int(digits[13]) == compute_check_digit(digits[:13], WEIGHTS_DV2)
    )


def normalize_cnpj(basico: str | None, ordem: str | None, dv: str | None) -> tuple[str | None, str | None]:
    """Classify one raw CNPJ triple. Scalar reference for the vectorized path.

    Returns (normalized_14_digits, reason). reason is None for a clean
    14-digit pass-through, REASON_ZERO_PADDED for a recovered 13-digit value
    (normalized holds the padded form), or a reject reason with
    normalized=None.
    """
    raw = f"{basico or ''}{ordem or ''}{dv or ''}".translate(_STRIP_TABLE)
    if not raw.isdigit():
        return None, REASON_BAD_LENGTH
    if len(raw) == 13:
        candidate = "0" + raw
        pass_reason: str | None = REASON_ZERO_PADDED
    elif len(raw) == 14:
        candidate = raw
        pass_reason = None
    else:
        return None, REASON_BAD_LENGTH
    if candidate == candidate[0] * 14:
        return None, REASON_REPEATED_DIGITS
    if not dv_matches(candidate):
        return None, REASON_DV_MISMATCH
    return candidate, pass_reason


# --- Vectorized chunk path -------------------------------------------------
# Expressions are built once at import time and reused for every chunk, so
# per-chunk work is columnar compute only — no per-row regex, no repeated
# expression-graph allocation.


def _digit(column: str, position: int) -> pl.Expr:
    return pl.col(column).str.slice(position, 1).cast(pl.Int32, strict=False).fill_null(0)


def _check_digit(column: str, weights: tuple[int, ...]) -> pl.Expr:
    total = sum((_digit(column, i) * w for i, w in enumerate(weights)), start=pl.lit(0))
    remainder = total % 11
    return pl.when(remainder < 2).then(pl.lit(0)).otherwise(11 - remainder)


_JOINED = pl.concat_str([pl.col(c).fill_null("") for c in CNPJ_COLUMNS]).alias(_JOINED_COL)
_CLEANED = pl.col(_JOINED_COL).str.replace_all(_PUNCTUATION_PATTERN, "").alias(_CLEANED_COL)
_LEN = pl.col(_CLEANED_COL).str.len_chars().alias(_LEN_COL)
_DIGITS = pl.col(_CLEANED_COL).str.contains(r"^\d+$").alias(_DIGITS_COL)
_NORMALIZED = (
    pl.when(pl.col(_LEN_COL) == 13)
    .then(pl.lit("0") + pl.col(_CLEANED_COL))
    .otherwise(pl.col(_CLEANED_COL))
    .alias(_NORM_COL)
)
# All-same-digit check without a regex backreference (unsupported by the
# regex engine polars uses): every position must equal the first digit.
# Only consulted for 14-digit normalized values; other lengths may compute
# garbage here, which the reason cascade never reads.
_REPEATED = pl.all_horizontal(
    pl.col(_NORM_COL).str.slice(i, 1) == pl.col(_NORM_COL).str.slice(0, 1) for i in range(1, 14)
).alias(_REPEATED_COL)
_DV_OK = (
    (_check_digit(_NORM_COL, WEIGHTS_DV1) == _digit(_NORM_COL, 12))
    & (_check_digit(_NORM_COL, WEIGHTS_DV2) == _digit(_NORM_COL, 13))
).alias(_DV_OK_COL)
_REASON = (
    pl.when(~pl.col(_DIGITS_COL) | ~pl.col(_LEN_COL).is_in([13, 14]))
    .then(pl.lit(REASON_BAD_LENGTH))
    .when(pl.col(_REPEATED_COL))
    .then(pl.lit(REASON_REPEATED_DIGITS))
    .when(~pl.col(_DV_OK_COL))
    .then(pl.lit(REASON_DV_MISMATCH))
    .when(pl.col(_LEN_COL) == 13)
    .then(pl.lit(REASON_ZERO_PADDED))
    .otherwise(None)
    .alias(_REASON_COL)
)


def split_valid_rows(df: pl.DataFrame, chunk_id: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split one estabelecimentos chunk into loadable rows and sidecar rejects.

    Returns (valid, rejects). valid keeps the input row order and carries the
    normalized CNPJ triple (punctuation stripped, 13-digit values left-padded
    with zero). rejects has REJECT_COLUMNS: chunk_id, the row's index within
    the chunk, the original untouched triple, and the reason code. Recovered
    zero_padded rows appear in BOTH frames — loaded normalized, audited raw.
    """
    missing = [c for c in CNPJ_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"split_valid_rows requires {CNPJ_COLUMNS}, missing: {missing}")

    classified = (
        df.with_row_index(_ROW_COL)
        .with_columns(_JOINED)
        .with_columns(_CLEANED)
        .with_columns(_LEN, _DIGITS)
        .with_columns(_NORMALIZED)
        .with_columns(_REPEATED, _DV_OK)
        .with_columns(_REASON)
    )

    loadable = pl.col(_REASON_COL).is_null() | (pl.col(_REASON_COL) == REASON_ZERO_PADDED)
    valid = (
        classified.filter(loadable)
        .with_columns(
            pl.col(_NORM_COL).str.slice(0, 8).alias("cnpj_basico"),
            pl.col(_NORM_COL).str.slice(8, 4).alias("cnpj_ordem"),
            pl.col(_NORM_COL).str.slice(12, 2).alias("cnpj_dv"),
        )
        .drop(*_HELPER_COLS)
    )

    rejects = classified.filter(pl.col(_REASON_COL).is_not_null()).select(
        pl.lit(chunk_id).cast(pl.Int64).alias("chunk_id"),
        pl.col(_ROW_COL).cast(pl.Int64).alias("row_number"),
        pl.col("cnpj_basico"),
        pl.col("cnpj_ordem"),
        pl.col("cnpj_dv"),
        pl.col(_REASON_COL).alias("reason"),
    )

    return valid, rejects
