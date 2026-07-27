"""NovoBench-compatible parquet loading and spectrum preprocessing."""

from __future__ import annotations

import re
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import polars as plr
import torch
from torch.utils.data import DataLoader, Dataset

from novobench.data import SpectrumData
from novobench.transforms import (
    FilterIntensity,
    RemovePrecursorPeak,
    ScaleIntensity,
    SetRangeMZ,
)
from novobench.transforms.misc import Compose


PROTON_MASS = 1.007276466812


def canonicalize_modified_sequence(sequence: str) -> str:
    """Normalize sequence aliases to the SemiNovo residue vocabulary."""
    sequence = sequence.replace("M(ox)", "M(+15.99)")
    return re.sub(r"C(?=[A-Z]|$)", "C(+57.02)", sequence)


def preprocessing_pipeline(config) -> Compose:
    """Build the spectrum preprocessing pipeline used by NovoBench."""
    return Compose(
        SetRangeMZ(config.min_mz, config.max_mz),
        RemovePrecursorPeak(config.remove_precursor_tol),
        FilterIntensity(config.min_intensity, config.n_peaks),
        ScaleIntensity(),
    )


class SpectrumDataset(Dataset):
    """Expose a preprocessed :class:`SpectrumData` object to PyTorch."""

    def __init__(self, data: SpectrumData):
        self.frame = data.df

    def __len__(self) -> int:
        return self.frame.height

    def __getitem__(self, index: int):
        mz = torch.tensor(
            self.frame[index, "mz_array"].to_list(),
            dtype=torch.float32,
        )
        intensity = torch.tensor(
            self.frame[index, "intensity_array"].to_list(),
            dtype=torch.float32,
        )
        precursor_mz = float(self.frame[index, "precursor_mz"])
        precursor_charge = int(self.frame[index, "precursor_charge"])
        sequence = (
            self.frame[index, "modified_sequence"]
            if "modified_sequence" in self.frame.columns
            else ""
        )
        return (
            torch.stack((mz, intensity), dim=1),
            precursor_mz,
            precursor_charge,
            sequence,
        )


def collate_batch(batch):
    """Pad spectra and construct neutral-mass precursor features."""
    spectra, precursor_mz, precursor_charge, sequences = zip(*batch)
    spectra = torch.nn.utils.rnn.pad_sequence(spectra, batch_first=True)
    precursor_mz = torch.tensor(precursor_mz, dtype=torch.float32)
    precursor_charge = torch.tensor(precursor_charge, dtype=torch.float32)
    neutral_mass = (precursor_mz - PROTON_MASS) * precursor_charge
    precursors = torch.stack(
        (neutral_mass, precursor_charge, precursor_mz),
        dim=1,
    )
    return spectra, precursors, np.asarray(sequences)


def make_loader(data, batch_size, num_workers, shuffle):
    """Create one pinned-memory spectrum data loader."""
    return DataLoader(
        SpectrumDataset(data),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate_batch,
    )


class NovoBenchDataModule(pl.LightningDataModule):
    """Load NovoBench train, validation, and test parquet splits."""

    def __init__(
        self,
        data_dir,
        config,
        batch_size=None,
        eval_batch_size=None,
        num_workers=None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.config = config
        self.batch_size = batch_size or config.train_batch_size
        self.eval_batch_size = eval_batch_size or config.eval_batch_size
        self.num_workers = (
            config.num_workers if num_workers is None else num_workers
        )
        self.train_data = None
        self.valid_data = None
        self.test_data = None

    def _load_split(self, split):
        path = self.data_dir / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing NovoBench split: {path}")
        frame = plr.read_parquet(path)
        if "modified_sequence" in frame.columns:
            frame = frame.with_columns(
                plr.col("modified_sequence")
                .map_elements(
                    canonicalize_modified_sequence,
                    return_dtype=plr.Utf8,
                )
                .alias("modified_sequence")
            )
        data = SpectrumData(frame)
        if not (self.data_dir / "preprocessed.json").exists():
            data = preprocessing_pipeline(self.config)(data)
        return data

    def setup(self, stage=None):
        if stage in (None, "fit") and self.train_data is None:
            self.train_data = self._load_split("train")
        if stage in (None, "fit", "validate") and self.valid_data is None:
            self.valid_data = self._load_split("valid")
        if stage in ("test", "predict") and self.test_data is None:
            self.test_data = self._load_split("test")

    def train_dataloader(self):
        return make_loader(
            self.train_data,
            self.batch_size,
            self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return make_loader(
            self.valid_data,
            self.eval_batch_size,
            self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return make_loader(
            self.test_data,
            self.eval_batch_size,
            self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
