"""FlashAttention encoder-decoder for de novo peptide sequencing."""

import heapq
import re

import lightning.pytorch as pl
import numpy as np
import torch
from depthcharge.components.encoders import FloatEncoder, PeakEncoder, PositionalEncoder
from depthcharge.masses import PeptideMass
from flash_attn.bert_padding import pad_input, unpad_input

from novobench.models.casanovo.casanovo_modeling import (
    CosineWarmupScheduler,
    Spec2Pep,
    _aa_pep_score,
)
from seminovo.ema import ExponentialMovingAverage
from seminovo.models.attention import make_flash_block
from seminovo.models.peak_embedding import MultiScalePeakEmbedding


def _unpad(hidden, mask):
    """Normalize FlashAttention 2.x unpadding return values across patch releases."""
    packed, indices, cu_seqlens, max_seqlen, *_ = unpad_input(hidden, mask)
    return packed, indices, cu_seqlens, max_seqlen


def teacher_forcing_accuracy(logits, tokens):
    """Return padding-aware token and exact-sequence teacher-forcing accuracy."""
    predicted = logits[:, :-1].argmax(dim=-1)
    valid = tokens.ne(0)
    correct = predicted.eq(tokens) & valid
    token_count = valid.sum()
    token_accuracy = correct.sum().float() / token_count.clamp_min(1)
    sequence_accuracy = (correct | ~valid).all(dim=1).float().mean()
    return token_accuracy, sequence_accuracy, token_count


def mass_feasible_token_mask(
    tokens,
    precursors,
    token_masses,
    stop_token,
    min_peptide_len,
    precursor_mass_tol,
    isotope_error_range,
):
    """Return feasible next-token candidates for each autoregressive prefix."""
    token_masses = token_masses.to(device=tokens.device, dtype=torch.float64)
    prefix_mass = token_masses[tokens].sum(dim=1)
    candidate_mass = prefix_mass[:, None] + token_masses[None, :]

    charge = precursors[:, 1].to(torch.float64)
    observed_mz = precursors[:, 2].to(torch.float64)
    isotopes = torch.arange(
        isotope_error_range[0],
        isotope_error_range[1] + 1,
        dtype=torch.float64,
        device=tokens.device,
    )
    target_neutral = (
        observed_mz[:, None] - isotopes[None, :] * 1.00335 / charge[:, None]
    ) * charge[:, None] - PeptideMass.proton * charge[:, None]
    target_residue = target_neutral - PeptideMass.h2o
    tolerance_da = (
        observed_mz[:, None] * charge[:, None] * float(precursor_mass_tol) * 1e-6
    )
    remaining = target_residue[:, None, :] - candidate_mass[:, :, None]
    positive_masses = token_masses[1:stop_token]
    minimum_residue = positive_masses[positive_masses > 0].min()
    residue_feasible = (
        (remaining >= -tolerance_da[:, None, :])
        & (
            (remaining.abs() <= tolerance_da[:, None, :])
            | (remaining >= minimum_residue - tolerance_da[:, None, :])
        )
    ).any(dim=-1)

    prefix_remaining = target_residue - prefix_mass[:, None]
    stop_feasible = (prefix_remaining.abs() <= tolerance_da).any(dim=-1)
    lengths = tokens.ne(0).sum(dim=1)
    stop_feasible &= lengths >= int(min_peptide_len)

    mask = residue_feasible
    mask[:, 0] = False
    mask[:, stop_token] = stop_feasible
    no_candidate = ~mask.any(dim=-1)
    mask[:, stop_token] |= no_candidate
    return mask


class PeptideTokenizer:
    """Use the exact residue indexing and C-to-N direction of Casanovo."""

    def __init__(self, residues):
        amino_acids = list(PeptideMass(residues=residues).masses.keys()) + ["$"]
        self.idx_to_aa = {index + 1: aa for index, aa in enumerate(amino_acids)}
        self.aa_to_idx = {aa: index for index, aa in self.idx_to_aa.items()}
        self.stop_token = self.aa_to_idx["$"]

    @property
    def vocab_size(self):
        return len(self.aa_to_idx)

    def encode(self, sequence):
        sequence = sequence.replace("I", "L")
        residues = re.split(r"(?<=.)(?=[A-Z])", sequence)
        residues = list(reversed(residues)) + ["$"]
        return torch.tensor([self.aa_to_idx[aa] for aa in residues], dtype=torch.long)

    def batch_encode(self, sequences, device):
        if sequences is None:
            return torch.empty(0, 0, dtype=torch.long, device=device)
        if isinstance(sequences, torch.Tensor):
            return sequences.to(device=device, dtype=torch.long)
        encoded = [self.encode(sequence) for sequence in sequences]
        tokens = torch.nn.utils.rnn.pad_sequence(encoded, batch_first=True)
        return tokens.to(device=device, non_blocking=True)

    def decode(self, tokens):
        sequence = [self.idx_to_aa.get(int(token), "") for token in tokens]
        if "$" in sequence:
            sequence = sequence[: sequence.index("$") + 1]
        return list(reversed(sequence))


class FlashStack(torch.nn.Module):
    """Run official FlashAttention blocks on packed variable-length sequences."""

    def __init__(self, layers, dim, prenorm):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.prenorm = bool(prenorm)
        self.norm = torch.nn.LayerNorm(dim) if self.prenorm else torch.nn.Identity()

    def forward(self, hidden, mixer_kwargs):
        if not self.prenorm:
            for layer in self.layers:
                hidden = layer(hidden, mixer_kwargs=mixer_kwargs)
            return hidden
        residual = None
        for layer in self.layers:
            hidden, residual = layer(hidden, residual, mixer_kwargs=mixer_kwargs)
        return self.norm((hidden + residual).to(dtype=self.norm.weight.dtype))


class FlashSpectrumEncoder(torch.nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        hidden_dim,
        num_layers,
        dropout,
        prenorm,
        peak_embedding="multiscale",
    ):
        super().__init__()
        if peak_embedding == "multiscale":
            self.peak_encoder = MultiScalePeakEmbedding(dim, dropout=dropout)
        elif peak_embedding == "legacy":
            self.peak_encoder = PeakEncoder(dim)
        else:
            raise ValueError(f"Unknown peak embedding: {peak_embedding}")
        self.latent_spectrum = torch.nn.Parameter(torch.randn(1, 1, dim))
        self.stack = FlashStack(
            [
                make_flash_block(
                    dim=dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    causal=False,
                    gated=False,
                    rotary=False,
                    dropout=dropout,
                    layer_idx=index,
                    prenorm=prenorm,
                )
                for index in range(num_layers)
            ],
            dim,
            prenorm,
        )

    @property
    def device(self):
        return self.latent_spectrum.device

    def forward_packed(self, spectra):
        batch = spectra.shape[0]
        peak_mask = spectra[..., 0] != 0
        hidden = self.peak_encoder(spectra)
        latent = self.latent_spectrum.expand(batch, -1, -1)
        hidden = torch.cat((latent, hidden), dim=1)
        mask = torch.cat(
            (torch.ones(batch, 1, dtype=torch.bool, device=spectra.device), peak_mask),
            dim=1,
        )
        packed, indices, cu_seqlens, max_seqlen = _unpad(hidden, mask)
        packed = self.stack(
            packed,
            {"cu_seqlens": cu_seqlens, "max_seqlen": max_seqlen},
        )
        return packed, indices, cu_seqlens, max_seqlen, hidden.shape[1]

    def forward(self, spectra, packed=False):
        memory = self.forward_packed(spectra)
        if packed:
            return memory
        packed, indices, _, _, padded_length = memory
        batch = spectra.shape[0]
        padded = pad_input(packed, indices, batch, padded_length)
        valid_mask = torch.cat(
            (
                torch.ones(batch, 1, dtype=torch.bool, device=spectra.device),
                spectra[..., 0] != 0,
            ),
            dim=1,
        )
        return padded, ~valid_mask


class FlashPeptideDecoder(torch.nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        hidden_dim,
        num_layers,
        dropout,
        gated,
        residues,
        max_charge,
        max_length,
        use_rope,
        mass_conditioning=True,
        prenorm=True,
    ):
        super().__init__()
        self.tokenizer = PeptideTokenizer(residues)
        self.aa_encoder = torch.nn.Embedding(
            self.tokenizer.vocab_size + 1,
            dim,
            padding_idx=0,
        )
        self.mass_encoder = FloatEncoder(dim)
        self.charge_encoder = torch.nn.Embedding(max_charge, dim)
        self.mass_conditioning = bool(mass_conditioning)
        self.prenorm = bool(prenorm)
        inference_token_masses = torch.zeros(self.tokenizer.vocab_size + 1)
        for token, index in self.tokenizer.aa_to_idx.items():
            if token != "$":
                inference_token_masses[index] = residues[token]
        self.register_buffer(
            "_inference_token_masses",
            inference_token_masses,
            persistent=False,
        )
        self.mass_pruning = False
        self.precursor_mass_tol = 50.0
        self.isotope_error_range = (0, 1)
        self.min_peptide_len = 6
        if self.mass_conditioning:
            self.remaining_mass_encoder = FloatEncoder(dim)
            self.remaining_ratio_encoder = FloatEncoder(dim)
            self.register_buffer(
                "token_masses",
                inference_token_masses.clone(),
                persistent=True,
            )
        if not use_rope:
            self.position_encoder = PositionalEncoder(dim)
        self.self_blocks = torch.nn.ModuleList()
        self.cross_blocks = torch.nn.ModuleList()
        for index in range(num_layers):
            self.self_blocks.append(
                make_flash_block(
                    dim=dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    causal=True,
                    gated=gated,
                    rotary=use_rope,
                    dropout=dropout,
                    layer_idx=2 * index,
                    with_mlp=False,
                    prenorm=prenorm,
                )
            )
            self.cross_blocks.append(
                make_flash_block(
                    dim=dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    causal=False,
                    gated=False,
                    dropout=dropout,
                    layer_idx=2 * index + 1,
                    cross_attn=True,
                    rotary=False,
                    prenorm=prenorm,
                )
            )
        self.norm = torch.nn.LayerNorm(dim) if prenorm else torch.nn.Identity()
        self.output = torch.nn.Linear(dim, self.tokenizer.vocab_size + 1)

    @property
    def device(self):
        return self.output.weight.device

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def _aa2idx(self):
        return self.tokenizer.aa_to_idx

    @property
    def reverse(self):
        return True

    def detokenize(self, tokens):
        return self.tokenizer.decode(tokens)

    def configure_mass_pruning(
        self,
        enabled,
        precursor_mass_tol,
        isotope_error_range,
        min_peptide_len,
    ):
        """Configure inference-only precursor-mass token pruning."""
        self.mass_pruning = bool(enabled)
        self.precursor_mass_tol = float(precursor_mass_tol)
        self.isotope_error_range = tuple(isotope_error_range)
        self.min_peptide_len = int(min_peptide_len)

    def remaining_residue_mass(self, tokens, neutral_mass):
        """Return residue mass left before each autoregressive prediction."""
        total = (neutral_mass - PeptideMass.h2o).clamp_min(0.0)
        consumed = self.token_masses[tokens].cumsum(dim=1)
        consumed_before = torch.cat((torch.zeros_like(total[:, None]), consumed), dim=1)
        return (total[:, None] - consumed_before).clamp_min(0.0)

    def forward(
        self,
        sequences,
        precursors,
        memory,
        memory_padding_mask=None,
        apply_mass_pruning=None,
    ):
        if sequences is None:
            tokens = torch.empty(
                precursors.shape[0], 0, dtype=torch.long, device=precursors.device
            )
        else:
            tokens = self.tokenizer.batch_encode(sequences, precursors.device)
        masses = self.mass_encoder(precursors[:, None, 0])
        charges = self.charge_encoder(precursors[:, 1].long() - 1)[:, None, :]
        precursor_query = masses + charges
        hidden = torch.cat((precursor_query, self.aa_encoder(tokens)), dim=1)
        if hasattr(self, "position_encoder"):
            hidden = self.position_encoder(hidden)
        if self.mass_conditioning:
            remaining = self.remaining_residue_mass(tokens, precursors[:, 0])
            denominator = (precursors[:, 0] - PeptideMass.h2o).clamp_min(1.0)[:, None]
            hidden = hidden + self.remaining_mass_encoder(remaining)
            hidden = hidden + self.remaining_ratio_encoder(remaining / denominator)
        query_mask = torch.cat(
            (
                torch.ones(tokens.shape[0], 1, dtype=torch.bool, device=tokens.device),
                tokens != 0,
            ),
            dim=1,
        )
        packed, indices, cu_q, max_q = _unpad(hidden, query_mask)
        if isinstance(memory, torch.Tensor):
            if memory_padding_mask is None:
                memory_valid_mask = torch.ones(
                    memory.shape[:2], dtype=torch.bool, device=memory.device
                )
            else:
                memory_valid_mask = ~memory_padding_mask
            memory_packed, memory_indices, cu_k, max_k = _unpad(
                memory, memory_valid_mask
            )
            memory = (
                memory_packed,
                memory_indices,
                cu_k,
                max_k,
                memory.shape[1],
            )
        memory_packed, _, cu_k, max_k, _ = memory
        self_kwargs = {"cu_seqlens": cu_q, "max_seqlen": max_q}
        cross_kwargs = {
            "x_kv": memory_packed,
            "cu_seqlens": cu_q,
            "max_seqlen": max_q,
            "cu_seqlens_k": cu_k,
            "max_seqlen_k": max_k,
        }
        if self.prenorm:
            residual = None
            for self_block, cross_block in zip(self.self_blocks, self.cross_blocks):
                packed, residual = self_block(
                    packed, residual, mixer_kwargs=self_kwargs
                )
                packed, residual = cross_block(
                    packed, residual, mixer_kwargs=cross_kwargs
                )
            packed = self.norm((packed + residual).to(dtype=self.norm.weight.dtype))
        else:
            for self_block, cross_block in zip(self.self_blocks, self.cross_blocks):
                packed = self_block(packed, mixer_kwargs=self_kwargs)
                packed = cross_block(packed, mixer_kwargs=cross_kwargs)
        logits_packed = self.output(packed)
        logits = pad_input(logits_packed, indices, tokens.shape[0], hidden.shape[1])
        # NovoBench's beam cache is allocated from the FP32 spectrum tensor.
        # Keep training logits in AMP dtype, but expose FP32 scores at inference.
        if not self.training:
            logits = logits.float()
            prune_mass = (
                self.mass_pruning
                if apply_mass_pruning is None
                else bool(apply_mass_pruning)
            )
            autoregressive = sequences is None or isinstance(sequences, torch.Tensor)
            if prune_mass and autoregressive:
                feasible = mass_feasible_token_mask(
                    tokens=tokens,
                    precursors=precursors,
                    token_masses=self._inference_token_masses,
                    stop_token=self.tokenizer.stop_token,
                    min_peptide_len=self.min_peptide_len,
                    precursor_mass_tol=self.precursor_mass_tol,
                    isotope_error_range=self.isotope_error_range,
                )
                logits[:, -1] = logits[:, -1].masked_fill(
                    ~feasible,
                    -torch.inf,
                )
        return logits, tokens


class FlashSpec2Pep(pl.LightningModule):
    """NovoBench-compatible supervised model using official FlashAttention blocks."""

    def __init__(
        self,
        dim_model,
        n_head,
        dim_feedforward,
        n_layers,
        dropout,
        residues,
        max_charge,
        max_length,
        train_label_smoothing,
        warmup_iters,
        max_iters,
        lr,
        weight_decay,
        gated_attention=True,
        use_rope=True,
        mass_conditioning=True,
        prenorm=True,
        optimizer_name="adam",
        ema_decay=None,
        peak_dropout=0.0,
        intensity_jitter=0.0,
        precursor_mass_tol=50,
        isotope_error_range=(0, 1),
        min_peptide_len=6,
        n_beams=5,
        top_match=1,
        mass_pruning=True,
        peak_embedding="multiscale",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = FlashSpectrumEncoder(
            dim_model,
            n_head,
            dim_feedforward,
            n_layers,
            dropout,
            prenorm,
            peak_embedding,
        )
        self.decoder = FlashPeptideDecoder(
            dim_model,
            n_head,
            dim_feedforward,
            n_layers,
            dropout,
            gated_attention,
            residues,
            max_charge,
            max_length,
            use_rope,
            mass_conditioning,
            prenorm,
        )
        self.decoder.configure_mass_pruning(
            enabled=mass_pruning,
            precursor_mass_tol=precursor_mass_tol,
            isotope_error_range=isotope_error_range,
            min_peptide_len=min_peptide_len,
        )
        self.decode_strategy = "mass_pruned" if mass_pruning else "casanovo"
        self.train_loss = torch.nn.CrossEntropyLoss(
            ignore_index=0,
            label_smoothing=train_label_smoothing,
        )
        self.valid_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.peak_dropout = float(peak_dropout)
        self.intensity_jitter = float(intensity_jitter)
        self.ema = (
            ExponentialMovingAverage(ema_decay) if ema_decay is not None else None
        )
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.max_length = max_length
        self.precursor_mass_tol = precursor_mass_tol
        self.isotope_error_range = isotope_error_range
        self.min_peptide_len = min_peptide_len
        self.n_beams = n_beams
        self.top_match = top_match
        self.peptide_mass_calculator = PeptideMass(residues=residues)
        self._fast_finish_beams = all(
            mass >= 0 and not residue.startswith(("+", "-"))
            for residue, mass in self.peptide_mass_calculator.masses.items()
        )
        beam_token_masses = torch.zeros(
            self.decoder.tokenizer.vocab_size + 1, dtype=torch.float64
        )
        for residue, index in self.decoder._aa2idx.items():
            if residue != "$":
                beam_token_masses[index] = self.peptide_mass_calculator.masses[residue]
        self.register_buffer("_beam_token_masses", beam_token_masses, persistent=False)
        self.stop_token = self.decoder.tokenizer.stop_token
        self.softmax = torch.nn.Softmax(2)

    def configure_decode_strategy(self, strategy):
        """Select official Casanovo decoding or next-token mass pruning."""
        strategy = str(strategy).lower()
        if strategy not in {"casanovo", "mass_pruned"}:
            raise ValueError(
                "Unknown decode strategy "
                f"{strategy!r}; expected 'casanovo' or 'mass_pruned'"
            )
        self.decode_strategy = strategy
        self.decoder.mass_pruning = strategy == "mass_pruned"

    def _augment_spectra(self, spectra):
        if not self.training or (self.peak_dropout <= 0 and self.intensity_jitter <= 0):
            return spectra
        view = spectra.clone()
        valid = view[..., 0].ne(0)
        if self.peak_dropout > 0:
            keep = torch.rand_like(view[..., 0]).ge(self.peak_dropout) & valid
            strongest = view[..., 1].masked_fill(~valid, -1).argmax(dim=1)
            keep.scatter_(1, strongest[:, None], True)
            view = view * keep[..., None]
            valid = keep
        if self.intensity_jitter > 0:
            scale = torch.empty_like(view[..., 1]).uniform_(
                1.0 - self.intensity_jitter,
                1.0 + self.intensity_jitter,
            )
            view[..., 1] = torch.where(valid, view[..., 1] * scale, view[..., 1])
        return view

    def _forward_step(self, spectra, precursors, sequences):
        spectra = self._augment_spectra(spectra)
        memory = self.encoder.forward_packed(spectra)
        return self.decoder(sequences, precursors, memory)

    forward = Spec2Pep.forward
    beam_search_decode = Spec2Pep.beam_search_decode
    _get_topk_beams = Spec2Pep._get_topk_beams

    def _cache_finished_beams(
        self,
        tokens,
        scores,
        step,
        beams_to_cache,
        beam_fits_precursor,
        pred_cache,
    ):
        """Cache equal-scoring beams with a deterministic comparable key."""
        for index in torch.where(beams_to_cache)[0].tolist():
            spectrum_index = index // self.n_beams
            predicted_tokens = tokens[index, : step + 1]
            has_stop = predicted_tokens[-1].eq(self.stop_token)
            peptide = predicted_tokens[:-1] if has_stop else predicted_tokens
            if any(
                torch.equal(cached[-1], peptide)
                for cached in pred_cache[spectrum_index]
            ):
                continue

            probabilities = self.softmax(scores[index : index + 1, : step + 1])
            aa_scores = probabilities[
                0,
                torch.arange(len(predicted_tokens), device=tokens.device),
                predicted_tokens,
            ].tolist()
            if not has_stop:
                aa_scores.append(0.0)
            aa_scores, peptide_score = _aa_pep_score(
                np.asarray(aa_scores),
                bool(beam_fits_precursor[index]),
            )
            aa_scores = aa_scores[:-1]
            tie_breaker = tuple(int(token) for token in peptide.tolist())
            entry = (
                float(peptide_score),
                tie_breaker,
                aa_scores,
                peptide.clone(),
            )
            cache = pred_cache[spectrum_index]
            if len(cache) < self.n_beams:
                heapq.heappush(cache, entry)
            else:
                heapq.heappushpop(cache, entry)

    def _get_top_peptide(self, pred_cache):
        """Return NovoBench predictions from the stable four-field cache."""
        for peptides in pred_cache.values():
            if not peptides:
                yield []
                continue
            yield [
                (
                    peptide_score,
                    aa_scores,
                    "".join(self.decoder.detokenize(predicted_tokens)),
                )
                for peptide_score, _, aa_scores, predicted_tokens in heapq.nlargest(
                    self.top_match,
                    peptides,
                )
            ]

    def _finish_beams(self, tokens, precursors, step):
        """Vectorized NovoBench beam finalization for positive-residue vocabularies."""
        if not self._fast_finish_beams:
            return Spec2Pep._finish_beams(self, tokens, precursors, step)

        current = tokens[:, step]
        finished = current.eq(self.stop_token)
        discarded = current.eq(0)
        peptide_length = step + 1 - finished.to(torch.long)
        discarded |= finished & peptide_length.lt(self.min_peptide_len)

        fits = torch.zeros_like(finished)
        candidates = finished & ~discarded
        candidate_tokens = tokens[candidates, : step + 1]
        residue_mass = self._beam_token_masses[candidate_tokens].sum(dim=1)
        charge = precursors[candidates, 1].to(torch.float64)
        observed_mz = precursors[candidates, 2].to(torch.float64)
        calculated_mz = (
            residue_mass + self.peptide_mass_calculator.h2o
        ) / charge + self.peptide_mass_calculator.proton
        isotopes = torch.arange(
            self.isotope_error_range[0],
            self.isotope_error_range[1] + 1,
            dtype=torch.float64,
            device=tokens.device,
        )
        corrected_mz = (
            observed_mz[:, None] - isotopes[None, :] * 1.00335 / charge[:, None]
        )
        delta_ppm = (calculated_mz[:, None] - corrected_mz) / observed_mz[:, None] * 1e6
        fits[candidates] = delta_ppm.abs().lt(self.precursor_mass_tol).any(dim=1)
        return finished, fits, discarded

    def _teacher_forcing_loss(self, logits, tokens, loss_fn=None):
        """Compute the single supervised objective used by every training path."""
        prediction = logits[:, :-1]
        objective = self.train_loss if loss_fn is None else loss_fn
        return objective(
            prediction.reshape(-1, prediction.shape[-1]),
            tokens.flatten(),
        )

    def _shared_step(self, batch, mode):
        logits, tokens = self._forward_step(*batch)
        loss_fn = self.train_loss if mode == "train" else self.valid_loss
        loss = self._teacher_forcing_loss(logits, tokens, loss_fn)
        if mode == "train":
            self.log_training_metrics(loss, batch_size=tokens.shape[0])
        else:
            token_accuracy, sequence_accuracy, token_count = teacher_forcing_accuracy(
                logits, tokens
            )
            self.log(
                "val/loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=tokens.shape[0],
            )
            self.log(
                "val/token_accuracy",
                token_accuracy,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=int(token_count.item()),
            )
            self.log(
                "val/sequence_accuracy",
                sequence_accuracy,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=tokens.shape[0],
            )
        return loss

    def log_training_metrics(self, loss, batch_size, extra=None):
        """Write one compact, aligned CSV row through Lightning's logger."""
        try:
            optimizer = self.optimizers(use_pl_optimizer=False)
            learning_rate = optimizer.param_groups[0]["lr"]
        except (AttributeError, RuntimeError):
            learning_rate = self.lr
        metrics = {
            "train/loss": loss,
            "train/lr": loss.detach().new_tensor(learning_rate),
        }
        if extra:
            metrics.update(extra)
        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "valid")

    def on_fit_start(self):
        if self.ema is not None:
            if self.ema.shadow:
                self.ema.align(self)
            else:
                self.ema.initialize(self)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema is not None:
            self.ema.update(self)

    def on_validation_start(self):
        if self.ema is not None and self.ema.shadow:
            self.ema.apply(self)

    def on_validation_end(self):
        if self.ema is not None and self.ema.backup:
            self.ema.restore(self)

    def on_save_checkpoint(self, checkpoint):
        if self.ema is not None and self.ema.shadow:
            checkpoint["ema_state_dict"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if self.ema is not None and checkpoint.get("ema_state_dict"):
            self.ema.load_state_dict(checkpoint["ema_state_dict"])

    def configure_optimizers(self):
        optimizer_cls = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
        }[self.optimizer_name]
        optimizer = optimizer_cls(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup=self.warmup_iters,
            max_iters=self.max_iters,
        )
        return [optimizer], {"scheduler": scheduler, "interval": "step"}
