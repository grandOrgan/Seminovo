import pytest

from seminovo.metrics import peptide_matches, peptide_precision_auc


def test_mass_equivalent_residues_match():
    assert peptide_matches("PEPTIDE", "PEPTLDE")
    assert peptide_matches("N(+.98)", "D")


def test_precision_and_auc_only():
    metrics = peptide_precision_auc(
        ["PEPTIDE", "AAAA"],
        ["PEPTIDE", "AAAG"],
        [0.9, 0.1],
    )

    assert set(metrics) == {"precision", "auc"}
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["auc"] == pytest.approx(0.0)
