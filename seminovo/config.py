"""Configuration loading for SemiNovo training and evaluation."""

from __future__ import annotations

from pathlib import Path

import yaml


class Config:
    """Load the public YAML configuration into a validated attribute object."""

    REQUIRED_SECTIONS = ("data", "model", "optimization", "trainer", "decoding")

    def __init__(self, path):
        path = Path(path)
        values = yaml.safe_load(path.read_text())
        missing = [section for section in self.REQUIRED_SECTIONS if section not in values]
        if missing:
            raise ValueError(f"Missing configuration sections: {missing}")

        self.random_seed = int(values.get("seed", 42))
        for section in self.REQUIRED_SECTIONS:
            for key, value in values[section].items():
                setattr(self, key, value)

        self.n_peaks = int(self.n_peaks)
        self.max_charge = int(self.max_charge)
        self.dim_model = int(self.dim_model)
        self.n_head = int(self.n_head)
        self.dim_feedforward = int(self.dim_feedforward)
        self.n_layers = int(self.n_layers)
        self.max_length = int(self.max_length)
        self.min_peptide_len = int(self.min_peptide_len)
        self.warmup_iters = int(self.warmup_iters)
        self.max_iters = int(self.max_iters)
        self.max_epochs = int(self.max_epochs)
        self.train_batch_size = int(self.train_batch_size)
        self.eval_batch_size = int(self.eval_batch_size)
        self.num_workers = int(self.num_workers)
        self.num_sanity_val_steps = int(self.num_sanity_val_steps)
        self.check_val_every_n_epoch = int(self.check_val_every_n_epoch)
        self.val_check_interval = (
            None
            if self.val_check_interval is None
            else int(self.val_check_interval)
        )
        self.n_beams = int(self.n_beams)
        self.top_match = int(self.top_match)
        self.isotope_error_range = tuple(int(x) for x in self.isotope_error_range)
        self.residues = {
            str(residue): float(mass)
            for residue, mass in self.residues.items()
        }

        if self.dim_model % self.n_head:
            raise ValueError("model.dim_model must be divisible by model.n_head")
        if self.max_iters <= self.warmup_iters:
            raise ValueError("optimization.max_iters must exceed warmup_iters")
        if not 0 <= float(self.train_label_smoothing) < 1:
            raise ValueError("train_label_smoothing must be in [0, 1)")
