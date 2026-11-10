"""Tests for the estabelecimentos CNPJ gate (cnpj_validation + processor wiring).

Covers the pure check-digit functions, the vectorized chunk splitter, and the
chunk-consistency contract: valid rows keep their order, rejects land in the
sidecar CSV with chunk_id/row_number/raw triple, and a sidecar write failure
prevents the chunk from being committed.
"""

import csv
from collections import Counter
from pathlib import Path

import polars as pl
import pytest

import processor
from cnpj_validation import (
    REASON_BAD_LENGTH,
    REASON_DV_MISMATCH,
    REASON_REPEATED_DIGITS,
    REASON_ZERO_PADDED,
    WEIGHTS_DV1,
    WEIGHTS_DV2,
    compute_check_digit,
    dv_matches,
    normalize_cnpj,
    split_valid_rows,
)
from config import Config
from file_types import COLUMNS
from main import load_postgres_csv
from processor import process_file

# Anchors verified against the mod-11 spec by hand:
#   11.222.333/0001-81, 12.345.678/0001-95, 04.183.008/0001-91
VALID_CNPJ = "11222333000181"
VALID_CNPJ_ALT = "12345678000195"
PADDED_CNPJ = "04183008000191"  # valid CNPJ whose leading zero was eaten


def _dv(base12: str) -> str:
    """Check digits for a 12-digit base, via the module under test."""
    d1 = compute_check_digit(base12, WEIGHTS_DV1)
    return f"{d1}{compute_check_digit(base12 + str(d1), WEIGHTS_DV2)}"


class TestComputeCheckDigit:
    def test_remainder_zero_maps_to_zero(self) -> None:
        assert compute_check_digit("000000000000", WEIGHTS_DV1) == 0

    def test_remainder_one_maps_to_zero(self) -> None:
        # weighted sum 12 → 12 % 11 == 1 → digit 0
        assert compute_check_digit("200000000001", WEIGHTS_DV1) == 0

    def test_remainder_two_is_not_below_two(self) -> None:
        # weighted sum 2 → 2 % 11 == 2 → 11 - 2 == 9 (boundary: < 2, not <= 2)
        assert compute_check_digit("000100000000", WEIGHTS_DV1) == 9

    def test_known_cnpj_digits(self) -> None:
        assert compute_check_digit("112223330001", WEIGHTS_DV1) == 8
        assert compute_check_digit("1122233300018", WEIGHTS_DV2) == 1


class TestDvMatches:
    def test_valid_cnpj(self) -> None:
        assert dv_matches(VALID_CNPJ)
        assert dv_matches(VALID_CNPJ_ALT)
        assert dv_matches(PADDED_CNPJ)

    def test_wrong_second_digit(self) -> None:
        assert not dv_matches("11222333000182")

    def test_all_zeros_passes_mod11(self) -> None:
        # This is exactly why the repeated_digits rule exists: 00000000000000
        # satisfies mod 11 and must be caught separately.
        assert dv_matches("00000000000000")

    def test_rejects_bad_shape(self) -> None:
        assert not dv_matches("123")
        assert not dv_matches("1122233300018A")


class TestNormalizeCnpj:
    @pytest.mark.parametrize(
        ("basico", "ordem", "dv", "expected"),
        [
            ("11222333", "0001", "81", (VALID_CNPJ, None)),
            ("12345678", "0001", "95", (VALID_CNPJ_ALT, None)),
            # 13 digits, leading zero eaten: recovers only because both DVs check out
            ("4183008", "0001", "91", (PADDED_CNPJ, REASON_ZERO_PADDED)),
            # 13 digits but DV does not verify after padding → reject
            ("4183008", "0001", "92", (None, REASON_DV_MISMATCH)),
            ("11222333", "0001", "82", (None, REASON_DV_MISMATCH)),
            # all-same-digit passes mod 11 but is not a real CNPJ
            ("00000000", "0000", "00", (None, REASON_REPEATED_DIGITS)),
            ("11111111", "1111", "11", (None, REASON_REPEATED_DIGITS)),
            # 15 digits, 12 digits, letters, nulls → bad_length
            ("11222333", "0001", "811", (None, REASON_BAD_LENGTH)),
            ("112223", "0001", "81", (None, REASON_BAD_LENGTH)),
            ("1122334A", "0001", "81", (None, REASON_BAD_LENGTH)),
            (None, None, None, (None, REASON_BAD_LENGTH)),
            # punctuation is stripped before classification
            ("11.222.333", "0001", "81", (VALID_CNPJ, None)),
        ],
    )
    def test_classify(self, basico: str | None, ordem: str | None, dv: str | None, expected: tuple) -> None:
        assert normalize_cnpj(basico, ordem, dv) == expected


# (basico, ordem, dv, nome_fantasia) — one row per dirty-value category.
SAMPLE_ROWS = [
    ("11222333", "0001", "81", "A"),  # valid
    ("12345678", "0001", "95", "B"),  # valid
    ("11222333", "0001", "82", "C"),  # dv_mismatch
    ("4183008", "0001", "91", "D"),  # 13 digits, recoverable → zero_padded
    ("4183008", "0001", "92", "E"),  # 13 digits, DV fails → dv_mismatch
    ("11222333", "0001", "811", "F"),  # 15 digits → bad_length
    ("00000000", "0000", "00", "G"),  # passes mod 11 → repeated_digits
    ("11.222.333", "0001", "81", "H"),  # punctuation → valid after strip
    ("1122334A", "0001", "81", "I"),  # non-digit → bad_length
]


def _sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "cnpj_basico": [r[0] for r in SAMPLE_ROWS],
            "cnpj_ordem": [r[1] for r in SAMPLE_ROWS],
            "cnpj_dv": [r[2] for r in SAMPLE_ROWS],
            "nome_fantasia": [r[3] for r in SAMPLE_ROWS],
        }
    )


class TestSplitValidRows:
    def test_valid_rows_keep_order_and_normalized_triple(self) -> None:
        valid, _ = split_valid_rows(_sample_df(), chunk_id=0)

        # Original order preserved; the 13-digit row is loaded left-padded and
        # the punctuation row is loaded with its digits-only triple.
        assert valid["cnpj_basico"].to_list() == ["11222333", "12345678", "04183008", "11222333"]
        assert valid["cnpj_ordem"].to_list() == ["0001", "0001", "0001", "0001"]
        assert valid["cnpj_dv"].to_list() == ["81", "95", "91", "81"]
        assert valid["nome_fantasia"].to_list() == ["A", "B", "D", "H"]

    def test_no_helper_columns_leak(self) -> None:
        valid, _ = split_valid_rows(_sample_df(), chunk_id=0)
        assert valid.columns == _sample_df().columns

    def test_rejects_carry_chunk_row_raw_triple_and_reason(self) -> None:
        _, rejects = split_valid_rows(_sample_df(), chunk_id=7)

        assert rejects.columns == ["chunk_id", "row_number", "cnpj_basico", "cnpj_ordem", "cnpj_dv", "reason"]
        assert rejects["chunk_id"].to_list() == [7] * 6
        assert rejects["row_number"].to_list() == [2, 3, 4, 5, 6, 8]
        assert rejects["reason"].to_list() == [
            REASON_DV_MISMATCH,
            REASON_ZERO_PADDED,
            REASON_DV_MISMATCH,
            REASON_BAD_LENGTH,
            REASON_REPEATED_DIGITS,
            REASON_BAD_LENGTH,
        ]
        # The recovered row is audited with its ORIGINAL (unpadded) triple.
        padded = rejects.row(1, named=True)
        assert padded["reason"] == REASON_ZERO_PADDED
        assert padded["cnpj_basico"] == "4183008"

    def test_missing_cnpj_columns_raise(self) -> None:
        with pytest.raises(ValueError, match="missing"):
            split_valid_rows(pl.DataFrame({"cnpj_basico": ["11222333"]}), chunk_id=0)


def _estabele_csv(tmp_path: Path, rows: list[tuple[str, str, str]]) -> Path:
    """Write a headerless ESTABELE source with the given CNPJ triples."""
    width = len(COLUMNS["ESTABELE"])
    path = tmp_path / "K3241.K03200Y0.D51113.ESTABELE"
    lines = [";".join([b, o, d] + [""] * (width - 3)) for b, o, d in rows]
    path.write_text("\n".join(lines) + "\n", encoding="ISO-8859-1")
    return path


def _read_sidecar(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None
        return reader.fieldnames, list(reader)


class TestProcessFileCnpjGate:
    def test_valid_order_and_reject_distribution(self, tmp_path: Path) -> None:
        csv_path = _estabele_csv(tmp_path, [(b, o, d) for b, o, d, _ in SAMPLE_ROWS])
        reject_dir = tmp_path / "rejects"

        results = list(process_file(csv_path, batch_size=50000, reject_dir=reject_dir))

        assert results, "expected at least one chunk"
        assert all(table == "estabelecimentos" for _, table, _ in results)
        basicos = [b for df, _, _ in results for b in df["cnpj_basico"].to_list()]
        assert basicos == ["11222333", "12345678", "04183008", "11222333"]

        header, sidecar_rows = _read_sidecar(reject_dir / f"{csv_path.name}.rejects.csv")
        assert header == ["chunk_id", "row_number", "cnpj_basico", "cnpj_ordem", "cnpj_dv", "reason"]
        assert Counter(r["reason"] for r in sidecar_rows) == {
            REASON_DV_MISMATCH: 2,
            REASON_BAD_LENGTH: 2,
            REASON_REPEATED_DIGITS: 1,
            REASON_ZERO_PADDED: 1,
        }
        # Single chunk for 9 rows: row numbers are the source row indices.
        assert {r["chunk_id"] for r in sidecar_rows} == {"0"}
        assert [int(r["row_number"]) for r in sidecar_rows] == [2, 3, 4, 5, 6, 8]
        # Raw triples are preserved untouched in the sidecar.
        by_row = {int(r["row_number"]): r for r in sidecar_rows}
        assert by_row[3]["cnpj_basico"] == "4183008"  # unpadded original
        assert by_row[8]["cnpj_basico"] == "1122334A"  # non-digit as received

    def test_clean_file_writes_no_sidecar(self, tmp_path: Path) -> None:
        csv_path = _estabele_csv(tmp_path, [("11222333", "0001", "81"), ("12345678", "0001", "95")])
        reject_dir = tmp_path / "rejects"

        results = list(process_file(csv_path, batch_size=50000, reject_dir=reject_dir))

        assert sum(df.height for df, _, _ in results) == 2
        assert not reject_dir.exists()

    def test_socios_cpf_not_treated_as_cnpj(self, tmp_path: Path) -> None:
        """socios.cnpj_cpf_do_socio may be an 11-digit CPF; the CNPJ gate must
        not touch SOCIOCSV at all — no filtering, no sidecar."""
        socios = tmp_path / "SOCIOCSV.csv"
        socios.write_text(
            '"12345678";"2";"ALICE SILVA";"12345678901";"22";"20200101";"";"";"";"";"0"\n',
            encoding="ISO-8859-1",
        )
        reject_dir = tmp_path / "rejects"

        results = list(process_file(socios, batch_size=50000, reject_dir=reject_dir))

        assert len(results) == 1
        df, table_name, _ = results[0]
        assert table_name == "socios"
        assert df["cnpj_cpf_do_socio"].to_list() == ["12345678901"]
        assert not reject_dir.exists()


class _FakeDB:
    """Collects committed chunks instead of writing to PostgreSQL."""

    def __init__(self) -> None:
        self.loaded: list[pl.DataFrame] = []

    def bulk_upsert(self, df: pl.DataFrame, table_name: str, columns: list[str]) -> None:
        self.loaded.append(df)


class TestSidecarFailure:
    def test_sidecar_write_failure_blocks_chunk_commit(self, tmp_path: Path) -> None:
        """If the sidecar cannot be written, the chunk must not be committed:
        the main store never gets rows whose audit trail is missing."""
        csv_path = _estabele_csv(tmp_path, [(b, o, d) for b, o, d, _ in SAMPLE_ROWS])
        blocker = tmp_path / "blocker"
        blocker.write_text("occupied")  # a FILE: reject_dir below it cannot be created
        cfg = Config(database_url="", reject_dir=str(blocker / "rejects"))
        db = _FakeDB()

        with pytest.raises(OSError):
            load_postgres_csv(csv_path, db, cfg)

        assert db.loaded == []

    def test_sidecar_failure_only_failed_chunk_uncommitted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Chunks before the failure stay committed; the failing chunk and
        everything after it must not reach the database."""
        rows: list[tuple[str, str, str]] = []
        for i in range(800):
            if i % 50 == 0:
                rows.append(("11222333", "0001", "82"))  # dv_mismatch → every chunk flushes
            else:
                basico = f"{i:08d}"
                rows.append((basico, "0001", _dv(f"{basico}0001")))
        csv_path = _estabele_csv(tmp_path, rows)
        cfg = Config(database_url="", reject_dir=str(tmp_path / "rejects"))
        db = _FakeDB()

        real_flush = processor._flush_rejects
        calls = 0

        def flaky_flush(rejects: pl.DataFrame, path: Path) -> None:
            nonlocal calls
            calls += 1
            if calls >= 2:
                raise OSError("simulated sidecar disk failure")
            real_flush(rejects, path)

        monkeypatch.setattr(processor, "_flush_rejects", flaky_flush)

        with pytest.raises(OSError, match="simulated sidecar"):
            load_postgres_csv(csv_path, db, cfg)

        # Exactly one chunk committed, and it is the valid prefix of the file:
        # no rows from the failed chunk (or later) leaked into the store.
        assert len(db.loaded) == 1
        committed = [b for df in db.loaded for b in df["cnpj_basico"].to_list()]
        valid_basicos = [b for i, (b, _, _) in enumerate(rows) if i % 50 != 0]
        assert 0 < len(committed) < len(valid_basicos)
        assert committed == valid_basicos[: len(committed)]
