"""Fast online pseudo-label training for SemiNovo."""

import torch
import torch.nn.functional as F
from torch.func import functional_call

from seminovo.models.seminovo import (
    FlashSpec2Pep,
    mass_feasible_token_mask,
)


def cumulative_prefix_mask(probabilities, threshold):
    """Accept only the contiguous prefix whose cumulative probability is high."""
    cumulative = probabilities.float().cumprod(dim=1)
    return cumulative >= float(threshold)


def linear_confidence_threshold(step, start, end, anneal_steps):
    """Linearly anneal a cumulative threshold, then clamp at the endpoint."""
    start = float(start)
    end = float(end)
    if start == end:
        return start
    if anneal_steps is None or int(anneal_steps) <= 0:
        raise ValueError("confidence_anneal_steps must be positive for dynamic thresholds")
    progress = min(max(float(step) / int(anneal_steps), 0.0), 1.0)
    return start + (end - start) * progress


def pseudo_length_histogram(accepted):
    """Count accepted prefix lengths in stable paper-facing bins."""
    lengths = accepted.sum(dim=1)
    return torch.stack(
        (
            lengths.eq(0).sum(),
            ((lengths >= 1) & (lengths <= 4)).sum(),
            ((lengths >= 5) & (lengths <= 8)).sum(),
            ((lengths >= 9) & (lengths <= 16)).sum(),
            ((lengths >= 17) & (lengths <= 32)).sum(),
        )
    )


def invert_label_smoothing(probabilities, smoothing):
    """Map a smoothed categorical distribution back to hard-label space."""
    smoothing = float(smoothing)
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("smoothing must be in [0, 1)")
    if smoothing == 0.0:
        return probabilities
    uniform = smoothing / probabilities.shape[-1]
    corrected = (probabilities - uniform).clamp_min(0.0) / (1.0 - smoothing)
    return corrected / corrected.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(corrected.dtype).tiny
    )


def select_feasible_token(logits, feasible, smoothing=0.0):
    """Choose a feasible token while retaining its original model probability."""
    probabilities = invert_label_smoothing(
        logits.float().softmax(dim=-1),
        smoothing,
    )
    token = probabilities.masked_fill(~feasible, -torch.inf).argmax(dim=-1)
    confidence = probabilities.gather(1, token[:, None]).squeeze(1)
    return confidence, token


def masked_hard_pseudo_loss(logits, tokens, accepted):
    """Compute hard-target CE over accepted pseudo-label tokens only."""
    accepted_count = accepted.sum()
    if not bool(accepted_count):
        return logits.sum() * 0.0
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        tokens.flatten(),
        reduction="none",
        ignore_index=0,
        label_smoothing=0.0,
    ).view_as(tokens)
    return (token_loss * accepted).sum() / accepted_count


def masked_soft_pseudo_loss(logits, teacher_probabilities, accepted, temperature=1.0):
    """Compute KL only over accepted pseudo-label positions."""
    accepted_count = accepted.sum()
    if not bool(accepted_count):
        return logits.sum() * 0.0
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("soft_pseudo_temperature must be positive")
    tiny = torch.finfo(teacher_probabilities.dtype).tiny
    teacher = torch.softmax(
        teacher_probabilities.clamp_min(tiny).log() / temperature,
        dim=-1,
    )
    student = torch.log_softmax(logits.float() / temperature, dim=-1)
    token_kl = F.kl_div(student, teacher, reduction="none").sum(dim=-1)
    return (token_kl * accepted).sum() / accepted_count * temperature**2


def pseudo_mass_mask(
    tokens,
    running_mass,
    precursors,
    token_masses,
    stop_token,
    step,
    min_peptide_len,
    h2o,
    proton,
    precursor_mass_tol,
    isotope_error_range,
):
    """Reject impossible prefixes and require a precursor fit at EOS."""
    token_mass = token_masses[tokens].to(torch.float64)
    next_mass = running_mass + token_mass
    precursor_mass = precursors[:, 0].to(torch.float64)
    tolerance_da = precursor_mass.abs() * float(precursor_mass_tol) * 1e-6
    max_isotope = max(isotope_error_range)
    below_upper_bound = (
        next_mass + float(h2o) <= precursor_mass + max_isotope * 1.00335 + tolerance_da
    )

    is_stop = tokens.eq(stop_token)
    peptide_length = step + 1 - is_stop.to(torch.long)
    charge = precursors[:, 1].to(torch.float64)
    observed_mz = precursors[:, 2].to(torch.float64)
    calculated_mz = (next_mass + float(h2o)) / charge + float(proton)
    isotopes = torch.arange(
        isotope_error_range[0],
        isotope_error_range[1] + 1,
        dtype=torch.float64,
        device=tokens.device,
    )
    corrected_mz = observed_mz[:, None] - isotopes[None, :] * 1.00335 / charge[:, None]
    delta_ppm = (calculated_mz[:, None] - corrected_mz) / observed_mz[:, None] * 1e6
    stop_fits = delta_ppm.abs().lt(float(precursor_mass_tol)).any(dim=1)
    stop_valid = peptide_length.ge(min_peptide_len) & stop_fits
    return torch.where(is_stop, stop_valid, below_upper_bound)


def _pad_spectra(left, right):
    length = max(left.shape[1], right.shape[1])
    if left.shape[1] < length:
        left = F.pad(left, (0, 0, 0, length - left.shape[1]))
    if right.shape[1] < length:
        right = F.pad(right, (0, 0, 0, length - right.shape[1]))
    return left, right


def _pad_tokens(tokens, length):
    return F.pad(tokens, (0, length - tokens.shape[1]))


class SemiSupervisedNovo(FlashSpec2Pep):
    """Joint teacher-forcing and confidence-filtered consistency training."""

    def __init__(
        self,
        *args,
        confidence_threshold=0.9,
        confidence_threshold_end=None,
        confidence_anneal_steps=None,
        lambda_u=1.0,
        pseudo_max_length=32,
        peak_dropout=0.05,
        intensity_jitter=0.02,
        enforce_pseudo_mass=True,
        pseudo_confidence_smoothing=0.0,
        soft_pseudo_weight=0.0,
        soft_pseudo_temperature=1.0,
        pseudo_metrics_interval=1000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.confidence_threshold = float(confidence_threshold)
        self.confidence_threshold_end = float(
            confidence_threshold
            if confidence_threshold_end is None
            else confidence_threshold_end
        )
        self.confidence_anneal_steps = confidence_anneal_steps
        self.lambda_u = float(lambda_u)
        self.pseudo_max_length = int(pseudo_max_length)
        self.enforce_pseudo_mass = bool(enforce_pseudo_mass)
        self.pseudo_confidence_smoothing = float(pseudo_confidence_smoothing)
        self.soft_pseudo_weight = float(soft_pseudo_weight)
        self.soft_pseudo_temperature = float(soft_pseudo_temperature)
        self.pseudo_metrics_interval = int(pseudo_metrics_interval)
        if not 0.0 <= self.soft_pseudo_weight <= 1.0:
            raise ValueError("soft_pseudo_weight must be in [0, 1]")
        if self.soft_pseudo_temperature <= 0:
            raise ValueError("soft_pseudo_temperature must be positive")
        if self.pseudo_metrics_interval <= 0:
            raise ValueError("pseudo_metrics_interval must be positive")
        self.register_buffer(
            "_pseudo_metric_window",
            torch.zeros(12, dtype=torch.float64),
            persistent=False,
        )
        self.peak_dropout = float(peak_dropout)
        self.intensity_jitter = float(intensity_jitter)

    def current_confidence_threshold(self):
        """Return the cumulative threshold at the current optimizer step."""
        trainer = getattr(self, "_trainer", None)
        step = 0 if trainer is None else trainer.global_step
        return linear_confidence_threshold(
            step,
            self.confidence_threshold,
            getattr(self, "confidence_threshold_end", self.confidence_threshold),
            getattr(self, "confidence_anneal_steps", None),
        )

    @torch.no_grad()
    def generate_pseudo_prefixes(
        self,
        spectra,
        precursors,
        return_distributions=False,
    ):
        """Generate deterministic prefixes from the EMA teacher."""
        was_training = self.training
        use_ema = self.ema is not None and bool(self.ema.shadow)
        self.eval()
        try:
            if use_ema:
                encoder_state = {
                    name.removeprefix("encoder."): value
                    for name, value in self.ema.shadow.items()
                    if name.startswith("encoder.")
                }
                decoder_state = {
                    name.removeprefix("decoder."): value
                    for name, value in self.ema.shadow.items()
                    if name.startswith("decoder.")
                }
                memory = functional_call(
                    self.encoder,
                    encoder_state,
                    (spectra,),
                    {"packed": True},
                    strict=False,
                )
            else:
                memory = self.encoder.forward_packed(spectra)
            batch_size = spectra.shape[0]
            tokens = torch.empty(batch_size, 0, dtype=torch.long, device=spectra.device)
            accepted = torch.empty(
                batch_size, 0, dtype=torch.bool, device=spectra.device
            )
            generated = torch.empty(
                batch_size, 0, dtype=torch.bool, device=spectra.device
            )
            cumulative = torch.ones(batch_size, device=spectra.device)
            running_mass = torch.zeros(
                batch_size, dtype=torch.float64, device=spectra.device
            )
            active = torch.ones(batch_size, dtype=torch.bool, device=spectra.device)
            confidence_threshold = self.current_confidence_threshold()
            teacher_distributions = []
            for _ in range(self.pseudo_max_length):
                generated = torch.cat((generated, active[:, None]), dim=1)
                decoder_kwargs = {}
                if self.enforce_pseudo_mass:
                    decoder_kwargs["apply_mass_pruning"] = False
                if use_ema:
                    logits, _ = functional_call(
                        self.decoder,
                        decoder_state,
                        (tokens, precursors, memory),
                        decoder_kwargs,
                        strict=False,
                    )
                elif self.enforce_pseudo_mass:
                    logits, _ = self.decoder(
                        tokens,
                        precursors,
                        memory,
                        apply_mass_pruning=False,
                    )
                else:
                    logits, _ = self.decoder(tokens, precursors, memory)
                step_logits = logits[:, -1].float().clone()
                step_logits[:, 0] = -torch.inf
                if self.enforce_pseudo_mass:
                    feasible = mass_feasible_token_mask(
                        tokens=tokens,
                        precursors=precursors,
                        token_masses=self._beam_token_masses,
                        stop_token=self.stop_token,
                        min_peptide_len=self.min_peptide_len,
                        precursor_mass_tol=self.precursor_mass_tol,
                        isotope_error_range=self.isotope_error_range,
                    )
                    probabilities = invert_label_smoothing(
                        step_logits.softmax(-1),
                        self.pseudo_confidence_smoothing,
                    )
                    token = probabilities.masked_fill(~feasible, -torch.inf).argmax(-1)
                    probability = probabilities.gather(1, token[:, None]).squeeze(1)
                    constrained = probabilities.masked_fill(~feasible, 0.0)
                    constrained = constrained / constrained.sum(
                        dim=-1, keepdim=True
                    ).clamp_min(torch.finfo(constrained.dtype).tiny)
                else:
                    probabilities = invert_label_smoothing(
                        step_logits.softmax(-1),
                        self.pseudo_confidence_smoothing,
                    )
                    probability, token = probabilities.max(-1)
                    constrained = probabilities
                if return_distributions:
                    teacher_distributions.append(constrained[:, None])
                cumulative = cumulative * probability
                if self.enforce_pseudo_mass:
                    mass_valid = pseudo_mass_mask(
                        token,
                        running_mass,
                        precursors,
                        self._beam_token_masses,
                        self.stop_token,
                        tokens.shape[1],
                        self.min_peptide_len,
                        self.peptide_mass_calculator.h2o,
                        self.peptide_mass_calculator.proton,
                        self.precursor_mass_tol,
                        self.isotope_error_range,
                    )
                else:
                    mass_valid = torch.ones_like(active)
                step_accepted = (
                    active & (cumulative >= confidence_threshold) & mass_valid
                )
                tokens = torch.cat(
                    (tokens, torch.where(step_accepted, token, 0)[:, None]),
                    dim=1,
                )
                accepted = torch.cat((accepted, step_accepted[:, None]), dim=1)
                if self.enforce_pseudo_mass:
                    running_mass = torch.where(
                        step_accepted & token.ne(self.stop_token),
                        running_mass + self._beam_token_masses[token],
                        running_mass,
                    )
                active = step_accepted & token.ne(self.stop_token)
                if not torch.any(active):
                    break
            outputs = (tokens, accepted, generated)
            if return_distributions:
                outputs += (torch.cat(teacher_distributions, dim=1),)
            return outputs
        finally:
            self.train(was_training)

    def _strong_view(self, spectra):
        view = spectra.clone()
        valid = view[..., 0].ne(0)
        if self.peak_dropout > 0:
            keep = (torch.rand_like(view[..., 0]) >= self.peak_dropout) & valid
            strongest = view[..., 1].masked_fill(~valid, -1).argmax(dim=1)
            keep.scatter_(1, strongest[:, None], True)
            view = view * keep[..., None]
            valid = keep
        if self.intensity_jitter > 0:
            scale = torch.empty_like(view[..., 1]).uniform_(
                1 - self.intensity_jitter,
                1 + self.intensity_jitter,
            )
            view[..., 1] = torch.where(valid, view[..., 1] * scale, view[..., 1])
        return view

    @torch.no_grad()
    def _record_pseudo_window(self, accepted, generated, sequence_acceptance, loss_u):
        """Aggregate stable pseudo-label diagnostics and log every fixed window."""
        histogram = pseudo_length_histogram(accepted).to(self._pseudo_metric_window)
        values = torch.stack(
            (
                accepted.sum(),
                generated.sum(),
                accepted.sum(dim=1).sum(),
                self._pseudo_metric_window.new_tensor(accepted.shape[0]),
                sequence_acceptance.sum(),
                *histogram,
                loss_u.detach(),
                loss_u.new_ones(()),
            )
        ).to(self._pseudo_metric_window)
        self._pseudo_metric_window.add_(values)
        trainer = getattr(self, "_trainer", None)
        step = 1 if trainer is None else trainer.global_step + 1
        if step % self.pseudo_metrics_interval:
            return
        window = self._pseudo_metric_window
        generated_count = window[1].clamp_min(1)
        sample_count = window[3].clamp_min(1)
        update_count = window[11].clamp_min(1)
        metrics = {
            "semi/token_acceptance": window[0] / generated_count,
            "semi/prefix_length_mean": window[2] / sample_count,
            "semi/sequence_acceptance": window[4] / sample_count,
            "semi/length_0": window[5] / sample_count,
            "semi/length_1_4": window[6] / sample_count,
            "semi/length_5_8": window[7] / sample_count,
            "semi/length_9_16": window[8] / sample_count,
            "semi/length_17_32": window[9] / sample_count,
            "semi/loss_u": window[10] / update_count,
            "semi/confidence_threshold": window.new_tensor(
                self.current_confidence_threshold()
            ),
        }
        self.log_dict(metrics, on_step=True, on_epoch=False, sync_dist=True)
        self._pseudo_metric_window.zero_()

    def _forward_preaugmented(self, spectra, precursors, tokens):
        """Forward views that have already received their intended augmentation."""
        memory = self.encoder.forward_packed(spectra)
        return self.decoder(tokens, precursors, memory)

    @staticmethod
    def _acceptance_metrics(accepted, generated):
        generated_count = generated.sum().clamp_min(1)
        ratio = accepted.sum().float() / generated_count
        prefix_length = accepted.sum(1).float().mean()
        return ratio, prefix_length

    def training_step(self, batch, batch_idx):
        spectra_s, precursors_s, sequences_s = batch["supervised"]
        spectra_u, precursors_u = batch["unlabeled"]
        use_soft_pseudo = self.soft_pseudo_weight > 0
        pseudo_outputs = self.generate_pseudo_prefixes(
            spectra_u,
            precursors_u,
            return_distributions=use_soft_pseudo,
        )
        pseudo_tokens, pseudo_mask, generated_mask = pseudo_outputs[:3]
        teacher_probabilities = pseudo_outputs[3] if use_soft_pseudo else None
        accepted_count = pseudo_mask.sum()
        pseudo_sequence_acceptance = (
            (pseudo_mask & pseudo_tokens.eq(self.stop_token)).any(dim=1).float().mean()
        )

        if accepted_count == 0:
            logits_s, tokens_s = self._forward_step(*batch["supervised"])
            loss_s = self._teacher_forcing_loss(logits_s, tokens_s)
            zero = loss_s.detach().new_zeros(())
            self._record_pseudo_window(
                pseudo_mask,
                generated_mask,
                torch.zeros_like(pseudo_sequence_acceptance),
                zero,
            )
            self.log_training_metrics(
                loss_s,
                batch_size=spectra_s.shape[0],
                extra={
                    "train/supervised_loss": loss_s.detach(),
                    "train/unsupervised_loss": zero,
                },
            )
            return loss_s

        supervised_tokens = self.decoder.tokenizer.batch_encode(
            sequences_s, spectra_s.device
        )
        token_length = max(supervised_tokens.shape[1], pseudo_tokens.shape[1])
        supervised_tokens = _pad_tokens(supervised_tokens, token_length)
        pseudo_tokens = _pad_tokens(pseudo_tokens, token_length)
        pseudo_mask = _pad_tokens(pseudo_mask, token_length)
        if teacher_probabilities is not None:
            teacher_probabilities = F.pad(
                teacher_probabilities,
                (0, 0, 0, token_length - teacher_probabilities.shape[1]),
            )
        spectra_s = self._augment_spectra(spectra_s)
        spectra_s, spectra_u = _pad_spectra(spectra_s, self._strong_view(spectra_u))
        spectra = torch.cat((spectra_s, spectra_u), dim=0)
        precursors = torch.cat((precursors_s, precursors_u), dim=0)
        tokens = torch.cat((supervised_tokens, pseudo_tokens), dim=0)
        logits, _ = self._forward_preaugmented(spectra, precursors, tokens)
        prediction = logits[:, :-1]
        supervised_size = spectra_s.shape[0]
        loss_s = self._teacher_forcing_loss(
            logits[:supervised_size],
            supervised_tokens,
        )
        loss_u_hard = masked_hard_pseudo_loss(
            prediction[supervised_size:],
            pseudo_tokens,
            pseudo_mask,
        )
        if teacher_probabilities is None:
            loss_u_soft = loss_u_hard.detach().new_zeros(())
            loss_u = loss_u_hard
        else:
            loss_u_soft = masked_soft_pseudo_loss(
                prediction[supervised_size:],
                teacher_probabilities,
                pseudo_mask,
                temperature=self.soft_pseudo_temperature,
            )
            loss_u = (
                (1.0 - self.soft_pseudo_weight) * loss_u_hard
                + self.soft_pseudo_weight * loss_u_soft
            )
        loss = loss_s + self.lambda_u * loss_u
        sequence_accepted = (pseudo_mask & pseudo_tokens.eq(self.stop_token)).any(dim=1)
        self._record_pseudo_window(
            pseudo_mask,
            generated_mask,
            sequence_accepted,
            loss_u,
        )
        self.log_training_metrics(
            loss,
            batch_size=spectra_s.shape[0],
            extra={
                "train/supervised_loss": loss_s.detach(),
                "train/unsupervised_loss": loss_u.detach(),
            },
        )
        return loss
