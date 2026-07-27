"""SemiNovo model construction and checkpoint loading."""

from __future__ import annotations

import torch

from seminovo.models.seminovo import FlashSpec2Pep


def build_model(config) -> FlashSpec2Pep:
    """Build the architecture used for the reported SemiNovo experiments."""
    return FlashSpec2Pep(
        dim_model=config.dim_model,
        n_head=config.n_head,
        dim_feedforward=config.dim_feedforward,
        n_layers=config.n_layers,
        dropout=float(config.dropout),
        residues=config.residues,
        max_charge=config.max_charge,
        max_length=config.max_length,
        train_label_smoothing=float(config.train_label_smoothing),
        warmup_iters=config.warmup_iters,
        max_iters=config.max_iters,
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        gated_attention=bool(config.gated_attention),
        use_rope=False,
        mass_conditioning=False,
        prenorm=True,
        optimizer_name=str(config.optimizer_name),
        ema_decay=float(config.ema_decay),
        peak_dropout=0.05,
        intensity_jitter=0.02,
        precursor_mass_tol=float(config.precursor_mass_tol),
        isotope_error_range=config.isotope_error_range,
        min_peptide_len=config.min_peptide_len,
        n_beams=config.n_beams,
        top_match=config.top_match,
        mass_pruning=True,
        peak_embedding=str(config.peak_embedding),
    )


def load_model(checkpoint_path) -> FlashSpec2Pep:
    """Load a supervised or semi-supervised SemiNovo checkpoint."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    hyper_parameters = dict(checkpoint["hyper_parameters"])
    model_class = FlashSpec2Pep
    if "confidence_threshold" in hyper_parameters:
        from seminovo.semi import SemiSupervisedNovo

        model_class = SemiSupervisedNovo

    model = model_class(**hyper_parameters)
    state = dict(checkpoint["state_dict"])
    state.update(checkpoint.get("ema_state_dict", {}))
    model.load_state_dict(state, strict=True)
    return model
