"""Multi-scale continuous peak embeddings for mass spectra."""

import torch

from depthcharge.components.encoders import FloatEncoder


class MultiScalePeakEmbedding(torch.nn.Module):
    """Fuse independent Fourier representations of m/z and intensity."""

    def __init__(
        self,
        d_model,
        min_mz_wavelength=0.001,
        max_mz_wavelength=10000.0,
        min_intensity_wavelength=1e-6,
        max_intensity_wavelength=1.0,
        dropout=0.0,
    ):
        super().__init__()
        self.mz_encoder = FloatEncoder(
            dim_model=d_model,
            min_wavelength=min_mz_wavelength,
            max_wavelength=max_mz_wavelength,
        )
        self.intensity_encoder = FloatEncoder(
            dim_model=d_model,
            min_wavelength=min_intensity_wavelength,
            max_wavelength=max_intensity_wavelength,
        )
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(2 * d_model, d_model),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model, d_model),
        )

    def forward(self, spectra):
        """Encode `[batch, peaks, (m/z, intensity)]` to model features."""
        mz = self.mz_encoder(spectra[..., 0])
        intensity = self.intensity_encoder(spectra[..., 1])
        return self.fusion(torch.cat((mz, intensity), dim=-1))
