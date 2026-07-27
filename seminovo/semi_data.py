"""NovoBench labeled data paired with the curated PRIDE array store."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from seminovo.data import NovoBenchDataModule, SpectrumDataset, collate_batch


PROTON_MASS = 1.007276466812


def _load_array(path):
    """Prefer mmap, with a filesystem compatibility fallback."""
    try:
        return np.load(path, mmap_mode="r")
    except OSError:
        return np.load(path)


class UnlabeledArrayDataset(Dataset):
    """Memory-map the canonical PRIDE arrays produced by the curation job."""

    def __init__(self, root):
        root = Path(root)
        manifest_path = root / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text())
            if manifest_path.exists() and manifest_path.stat().st_size
            else {}
        )
        self.segments = None
        if manifest.get("kind") == "indexed_composite_unlabeled_array_store":
            self._load_composite(root, manifest)
            return
        self.spectra = _load_array(root / "spectra.npy")
        self.precursor = _load_array(root / "precursor.npy")
        self._validate_arrays(self.spectra, self.precursor)

    @staticmethod
    def _validate_arrays(spectra, precursor):
        if spectra.ndim != 3 or spectra.shape[-1] != 2:
            raise ValueError(f"Invalid spectra array shape: {spectra.shape}")
        if precursor.shape != (len(spectra), 2):
            raise ValueError(f"Invalid precursor array shape: {precursor.shape}")

    def _load_composite(self, root, manifest):
        self.segments = []
        self.segment_ends = []
        derived_indices = {}
        index_spec = manifest.get("index_spec")
        if index_spec:
            from seminovo.build_common3_indexed import load_target_indices
            from seminovo.common3_multiscale import select_nested_global_indices

            local_sources = root / "index_sources"
            ranking_path = local_sources / "ranking.json"
            if not ranking_path.exists():
                ranking_path = Path(index_spec["ranking_source"])
            selection_dir = local_sources / "selection"
            if not selection_dir.exists():
                selection_dir = Path(index_spec["selection_source"])
            local_target_audit = root / "bases" / "target" / "replacement_audit.jsonl"
            target_audit = (
                local_target_audit
                if local_target_audit.exists()
                else Path(index_spec["target_audit_source"])
            )
            ranking = json.loads(ranking_path.read_text())["ranking"]
            quota = int(index_spec["pride_quota"])
            derived_indices["pride"] = select_nested_global_indices(
                ranking,
                selection_dir,
                quotas=(quota,),
                seed=int(index_spec["seed"]),
            )[quota]
            derived_indices["target"], _ = load_target_indices(
                target_audit,
                expected_rows=int(index_spec["expected_target_rows"]),
            )
        total = 0
        for segment in manifest.get("segments", []):
            local_base = root / "bases" / segment["name"]
            base = local_base if local_base.exists() else Path(segment["source"])
            spectra = _load_array(base / "spectra.npy")
            precursor = _load_array(base / "precursor.npy")
            self._validate_arrays(spectra, precursor)
            indices = (
                _load_array(root / segment["indices"])
                if "indices" in segment
                else derived_indices[segment["name"]]
            )
            if indices.ndim != 1:
                raise ValueError("Composite indexes must be one-dimensional")
            if len(indices) and int(indices.max()) >= len(spectra):
                raise ValueError(f"Composite index exceeds source store: {segment['name']}")
            self.segments.append((spectra, precursor, indices))
            total += len(indices)
            self.segment_ends.append(total)
        if not self.segments or total != int(manifest.get("rows", -1)):
            raise ValueError("Composite manifest row count does not match its indexes")

    def __len__(self):
        return self.segment_ends[-1] if self.segments is not None else len(self.spectra)

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self.segments is None:
            spectra = self.spectra
            precursor_values = self.precursor
            source_index = index
        else:
            segment_index = int(np.searchsorted(self.segment_ends, index, side="right"))
            start = 0 if segment_index == 0 else self.segment_ends[segment_index - 1]
            spectra, precursor_values, indices = self.segments[segment_index]
            source_index = int(indices[index - start])
        spectrum = torch.from_numpy(np.array(spectra[source_index], copy=True))
        # The original curation store used L2 scaling, while NovoBench's
        # Casanovo pipeline retains base-peak scaling. This global rescale
        # exactly restores the expected intensity convention.
        intensity_max = spectrum[:, 1].max().clamp_min(torch.finfo(spectrum.dtype).tiny)
        spectrum[:, 1].div_(intensity_max)
        mz, charge = map(float, precursor_values[source_index])
        precursor = torch.tensor(
            [(mz - PROTON_MASS) * charge, charge, mz],
            dtype=torch.float32,
        )
        return spectrum, precursor


class MultiSourceUnlabeledDataset(ConcatDataset):
    """Concatenate complete unlabeled stores into one shuffled epoch."""

    def __init__(self, roots):
        roots = [Path(root) for root in roots]
        super().__init__([UnlabeledArrayDataset(root) for root in roots])


class PairedSpectrumDataset(Dataset):
    """Pair every unlabeled item with a cycling labeled item."""

    def __init__(self, supervised, unlabeled):
        self.supervised = supervised
        self.unlabeled = unlabeled

    def __len__(self):
        return max(len(self.supervised), len(self.unlabeled))

    def __getitem__(self, index):
        return (
            self.supervised[index % len(self.supervised)],
            self.unlabeled[index % len(self.unlabeled)],
        )


def paired_collate(items):
    supervised, unlabeled = zip(*items)
    return {
        "supervised": collate_batch(supervised),
        "unlabeled": default_collate(unlabeled),
    }


class SemiNovoDataModule(NovoBenchDataModule):
    """Use the complete unlabeled corpus while cycling labeled examples."""

    def __init__(
        self,
        *args,
        unlabeled_dir,
        unlabeled_batch_size,
        extra_unlabeled_dirs=(),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.unlabeled_dirs = [
            Path(unlabeled_dir),
            *(Path(path) for path in extra_unlabeled_dirs),
        ]
        self.unlabeled_batch_size = int(unlabeled_batch_size)

    def setup(self, stage=None):
        super().setup(stage)
        if stage in (None, "fit") and not hasattr(self, "unlabeled_data"):
            self.unlabeled_data = MultiSourceUnlabeledDataset(self.unlabeled_dirs)

    def train_dataloader(self):
        paired = PairedSpectrumDataset(
            SpectrumDataset(self.train_data), self.unlabeled_data
        )
        loader_kwargs = {
            "batch_size": self.unlabeled_batch_size,
            "shuffle": True,
            "num_workers": self.num_workers,
            "pin_memory": True,
            "drop_last": True,
        }
        if self.num_workers > 0:
            loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
        return DataLoader(paired, collate_fn=paired_collate, **loader_kwargs)
