"""Train SemiNovo jointly on NovoBench labels and curated PRIDE spectra."""

import argparse
import json
from pathlib import Path

import lightning.pytorch as pl
import torch

from seminovo.config import Config
from seminovo.semi import SemiSupervisedNovo
from seminovo.semi_data import SemiNovoDataModule
from seminovo.train import scaled_schedule_steps
from seminovo.training import build_trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/seminovo.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--unlabeled-dir", required=True)
    parser.add_argument("--extra-unlabeled-dir", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--unlabeled-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--model-dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--confidence-threshold-end", type=float)
    parser.add_argument("--confidence-anneal-steps", type=int)
    parser.add_argument("--lambda-u", type=float, default=1.0)
    parser.add_argument("--teacher-ema-decay", type=float, default=0.999)
    parser.add_argument("--supervised-label-smoothing", type=float, default=0.0)
    parser.add_argument("--pseudo-max-length", type=int, default=32)
    parser.add_argument("--peak-dropout", type=float, default=0.05)
    parser.add_argument("--intensity-jitter", type=float, default=0.02)
    parser.add_argument("--soft-pseudo-weight", type=float, default=0.0)
    parser.add_argument("--soft-pseudo-temperature", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--schedule-max-steps", type=int)
    parser.add_argument("--val-check-interval", type=int, default=1000)
    return parser.parse_args()


def resolve_schedule_steps(
    epochs,
    steps_per_epoch,
    warmup_steps=None,
    schedule_max_steps=None,
):
    """Resolve a complete step schedule from the actual semi-supervised epoch."""
    epochs = int(epochs)
    steps_per_epoch = int(steps_per_epoch)
    if epochs <= 0 or steps_per_epoch <= 0:
        raise ValueError("epochs and steps_per_epoch must be positive")
    total_steps = (
        epochs * steps_per_epoch
        if schedule_max_steps is None
        else int(schedule_max_steps)
    )
    warmup_steps = 2000 if warmup_steps is None else int(warmup_steps)
    if total_steps <= 0:
        raise ValueError("schedule_max_steps must be positive")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must be non-negative and shorter than the run")
    return warmup_steps, total_steps


def validate_confidence_schedule(start, end):
    """Keep cumulative pseudo-label thresholds inside the approved safe range."""
    start = float(start)
    end = start if end is None else float(end)
    if not 0.90 <= start <= 1.0 or not 0.90 <= end <= 1.0:
        raise ValueError("confidence thresholds must stay within [0.90, 1.0]")
    return start, end


def dataset_sample_count(dataset):
    """Return the number of spectra for NovoBench and PyTorch datasets."""
    frame = getattr(dataset, "df", None)
    if frame is not None:
        return int(frame.height)
    return len(dataset)


def initialize_from_checkpoint(model, checkpoint_path):
    """Initialize the student and EMA teacher from checkpoint EMA weights."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ema_state = checkpoint.get("ema_state_dict")
    if not ema_state:
        raise ValueError(
            f"Checkpoint has no EMA weights: {checkpoint_path}"
        )
    online_state = dict(checkpoint["state_dict"])
    online_state.update(ema_state)
    model.load_state_dict(online_state, strict=True)
    if model.ema is None:
        raise ValueError("Semi-supervised training requires an EMA teacher")
    model.ema.load_state_dict(ema_state)


def build_semi_model(config, args):
    """Build the same regularized G1 architecture used by supervised training."""
    schedule_batch_size = getattr(args, "unlabeled_batch_size", args.batch_size)
    return SemiSupervisedNovo(
        dim_model=config.dim_model,
        n_head=config.n_head,
        dim_feedforward=config.dim_feedforward,
        n_layers=config.n_layers,
        dropout=args.model_dropout,
        residues=config.residues,
        max_charge=config.max_charge,
        max_length=config.max_length,
        train_label_smoothing=getattr(args, "supervised_label_smoothing", 0.0),
        warmup_iters=getattr(args, "warmup_steps", None)
        if getattr(args, "warmup_steps", None) is not None
        else scaled_schedule_steps(config.warmup_iters, schedule_batch_size),
        max_iters=getattr(args, "schedule_max_steps", None)
        if getattr(args, "schedule_max_steps", None) is not None
        else scaled_schedule_steps(config.max_iters, schedule_batch_size),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        gated_attention=True,
        use_rope=False,
        mass_conditioning=False,
        prenorm=True,
        optimizer_name="adamw",
        ema_decay=getattr(args, "teacher_ema_decay", 0.999),
        peak_dropout=args.peak_dropout,
        intensity_jitter=args.intensity_jitter,
        precursor_mass_tol=config.precursor_mass_tol,
        isotope_error_range=config.isotope_error_range,
        min_peptide_len=config.min_peptide_len,
        n_beams=config.n_beams,
        top_match=config.top_match,
        confidence_threshold=args.confidence_threshold,
        confidence_threshold_end=args.confidence_threshold_end,
        confidence_anneal_steps=args.confidence_anneal_steps,
        pseudo_confidence_smoothing=getattr(args, "supervised_label_smoothing", 0.0),
        lambda_u=args.lambda_u,
        pseudo_max_length=args.pseudo_max_length,
        soft_pseudo_weight=args.soft_pseudo_weight,
        soft_pseudo_temperature=args.soft_pseudo_temperature,
    )


def main():
    args = parse_args()
    config = Config(args.config)
    config.max_epochs = args.epochs
    seed = config.random_seed if args.seed is None else args.seed
    pl.seed_everything(seed, workers=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = SemiNovoDataModule(
        args.data_dir,
        config,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        unlabeled_dir=args.unlabeled_dir,
        unlabeled_batch_size=args.unlabeled_batch_size,
        extra_unlabeled_dirs=args.extra_unlabeled_dir,
    )
    data.setup("fit")
    steps_per_epoch = len(data.train_dataloader())
    args.warmup_steps, args.schedule_max_steps = resolve_schedule_steps(
        args.epochs,
        steps_per_epoch,
        args.warmup_steps,
        args.schedule_max_steps,
    )
    if args.confidence_threshold_end is not None:
        if args.confidence_anneal_steps is None:
            args.confidence_anneal_steps = args.schedule_max_steps
    args.confidence_threshold, resolved_threshold_end = validate_confidence_schedule(
        args.confidence_threshold,
        args.confidence_threshold_end,
    )
    if args.confidence_threshold_end is not None:
        args.confidence_threshold_end = resolved_threshold_end
    print(
        f"steps_per_epoch={steps_per_epoch} "
        f"warmup_steps={args.warmup_steps} "
        f"schedule_max_steps={args.schedule_max_steps}"
    )
    run_config = {
        **vars(args),
        "seed": seed,
        "steps_per_epoch": steps_per_epoch,
        "supervised_samples": dataset_sample_count(data.train_data),
        "unlabeled_samples": dataset_sample_count(data.unlabeled_data),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True, default=str) + "\n"
    )
    model = build_semi_model(config, args)
    if config.load_checkpoint:
        initialize_from_checkpoint(model, config.load_checkpoint)
        print(f"initialized_from={config.load_checkpoint}")
    trainer = build_trainer(
        config,
        output_dir=output_dir,
        accelerator="gpu",
        devices=1,
        max_steps=args.max_steps,
        val_check_interval=args.val_check_interval,
        checkpoint_metric="val/sequence_accuracy",
        checkpoint_filename="best-val-sequence-accuracy",
        checkpoint_mode="max",
    )
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
