"""NovoBench peptide precision and precision-coverage AUC."""

import re

import numpy as np
from sklearn.metrics import auc
from spectrum_utils.utils import mass_diff


RESIDUE_MASS = {
    "G": 57.02146372057,
    "A": 71.03711378471,
    "S": 87.03202840427001,
    "P": 97.05276384885,
    "V": 99.06841391299,
    "T": 101.04767846841,
    "C": 103.00918478471,
    "L": 113.08406397713001,
    "I": 113.08406397713001,
    "J": 113.08406397713001,
    "N": 114.04292744114001,
    "D": 115.02694302383001,
    "Q": 128.05857750527997,
    "K": 128.09496301399997,
    "E": 129.04259308796998,
    "M": 131.04048491299,
    "H": 137.05891185845002,
    "F": 147.06841391298997,
    "U": 150.95363508471,
    "R": 156.10111102359997,
    "Y": 163.06332853254997,
    "W": 186.07931294985997,
    "O": 237.14772686284996,
    "M(+15.99)": 147.0354,
    "M(ox)": 147.0354,
    "Q(+.98)": 129.0426,
    "Q(Deamidation)": 129.0426,
    "N(+.98)": 115.02695,
    "N(Deamidation)": 115.02695,
    "C(+57.02)": 160.03065,
}


def split_peptide(peptide):
    """Tokenize a modified peptide using the NovoBench residue vocabulary."""
    pattern = "|".join(
        map(re.escape, sorted(RESIDUE_MASS, key=len, reverse=True))
    )
    tokens = re.findall(pattern, peptide)
    if "".join(tokens) != peptide:
        raise ValueError(f"Peptide contains unsupported residues: {peptide}")
    return tokens


def _prefix_matches(
    truth,
    prediction,
    cumulative_tolerance=0.5,
    residue_tolerance=0.1,
):
    matches = np.zeros(max(len(truth), len(prediction)), dtype=np.bool_)
    truth_index = prediction_index = 0
    truth_mass = prediction_mass = 0.0
    while truth_index < len(truth) and prediction_index < len(prediction):
        truth_residue_mass = RESIDUE_MASS[truth[truth_index]]
        prediction_residue_mass = RESIDUE_MASS[prediction[prediction_index]]
        if (
            abs(
                mass_diff(
                    truth_mass + truth_residue_mass,
                    prediction_mass + prediction_residue_mass,
                    True,
                )
            )
            < cumulative_tolerance
        ):
            match_index = max(truth_index, prediction_index)
            matches[match_index] = (
                abs(mass_diff(truth_residue_mass, prediction_residue_mass, True))
                < residue_tolerance
            )
            truth_index += 1
            prediction_index += 1
            truth_mass += truth_residue_mass
            prediction_mass += prediction_residue_mass
        elif prediction_mass + prediction_residue_mass > truth_mass + truth_residue_mass:
            truth_index += 1
            truth_mass += truth_residue_mass
        else:
            prediction_index += 1
            prediction_mass += prediction_residue_mass
    return matches


def peptide_matches(truth, prediction):
    """Return whether every residue matches under the NovoBench mass rule."""
    truth = split_peptide(truth) if isinstance(truth, str) else truth
    prediction = split_peptide(prediction) if isinstance(prediction, str) else prediction
    if not prediction:
        return False

    matches = _prefix_matches(truth, prediction)
    if matches.all():
        return True

    truth_index = len(truth) - 1
    prediction_index = len(prediction) - 1
    stop_index = int(np.flatnonzero(~matches)[0])
    truth_mass = prediction_mass = 0.0
    while truth_index >= stop_index and prediction_index >= stop_index:
        truth_residue_mass = RESIDUE_MASS[truth[truth_index]]
        prediction_residue_mass = RESIDUE_MASS[prediction[prediction_index]]
        if (
            abs(
                mass_diff(
                    truth_mass + truth_residue_mass,
                    prediction_mass + prediction_residue_mass,
                    True,
                )
            )
            < 0.5
        ):
            match_index = max(truth_index, prediction_index)
            matches[match_index] = (
                abs(mass_diff(truth_residue_mass, prediction_residue_mass, True))
                < 0.1
            )
            truth_index -= 1
            prediction_index -= 1
            truth_mass += truth_residue_mass
            prediction_mass += prediction_residue_mass
        elif prediction_mass + prediction_residue_mass > truth_mass + truth_residue_mass:
            truth_index -= 1
            truth_mass += truth_residue_mass
        else:
            prediction_index -= 1
            prediction_mass += prediction_residue_mass
    return bool(matches.all())


def peptide_precision_auc(truths, predictions, scores):
    """Calculate peptide precision and precision-coverage AUC."""
    correct = np.asarray(
        [
            peptide_matches(truth, prediction)
            for truth, prediction in zip(truths, predictions)
        ],
        dtype=np.float64,
    )
    if correct.size == 0:
        return {"precision": 0.0, "auc": 0.0}

    ranked = sorted(zip(scores, correct), key=lambda item: item[0], reverse=True)
    ranked_correct = np.asarray([item[1] for item in ranked], dtype=np.float64)
    cumulative_correct = np.cumsum(ranked_correct)
    precision_curve = cumulative_correct / np.arange(1, correct.size + 1)
    coverage_curve = cumulative_correct / correct.size
    return {
        "precision": float(correct.sum() / (correct.size + 1e-8)),
        "auc": float(auc(coverage_curve, precision_curve)),
    }
