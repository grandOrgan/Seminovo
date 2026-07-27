"""Train SemiNovo on labeled NovoBench parquet splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning.pytorch as pl

from seminovo.checkpoints import build_model
from seminovo.config import Config
from seminovo.data import NovoBenchDataModule
from seminovo.training import build_trainer


def scaled_schedule_steps(steps, batch_size, reference_batch_size=32):
    """Keep scheduler milestones fixed in numbers of spectra seen."""
    return max(1, round(steps * reference_batch_size / batch_size))


def resolve_schedule_steps(
    warmup_steps,
    max_steps,
    batch_size,
    reference_batch_size=32,
    warmup_override=None,
    max_override=None,
):
    """Resolve explicit schedule overrides or scale the reference schedule."""
    warmup = (
        int(warmup_override)
        if warmup_override is not None
        else scaled_schedule_steps(warmup_steps, batch_size, reference_batch_size)
    )
    maximum = (
        int(max_override)
        if max_override is not None
        else scaled_schedule_steps(max_steps, batch_size, reference_batch_size)
    )
    if warmup < 0 or maximum <= warmup:
        raise ValueError("Schedule requires 0 <= warmup_steps < max_steps")
    return warmup, maximum


def resolve_validation_interval(requested_steps, steps_per_epoch):
    """Validate at the requested interval or once per shorter epoch."""
    if requested_steps is None:
        return None
    requested_steps = int(requested_steps)
    steps_per_epoch = int(steps_per_epoch)
    if requested_steps <= 0 or steps_per_epoch <= 0:
        raise ValueError("Validation interval and steps per epoch must be positive")
    return min(requested_steps, steps_per_epoch)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/seminovo.yaml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--label-smoothing", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--schedule-max-steps", type=int)
    parser.add_argument("--val-check-interval", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(args.config)
    if args.epochs is not None:
        config.max_epochs = args.epochs
    if args.label_smoothing is not None:
        config.train_label_smoothing = args.label_smoothing

    pl.seed_everything(config.random_seed, workers=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = NovoBenchDataModule(
        args.data_dir,
        config,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    data.setup("fit")

    batch_size = args.batch_size or config.train_batch_size
    warmup, maximum = resolve_schedule_steps(
        config.warmup_iters,
        config.max_iters,
        batch_size,
        warmup_override=args.warmup_steps,
        max_override=args.schedule_max_steps,
    )
    model = build_model(config)
    model.warmup_iters = warmup
    model.max_iters = maximum
    if args.learning_rate is not None:
        model.lr = args.learning_rate

    requested_interval = (
        config.val_check_interval
        if args.val_check_interval is None
        else args.val_check_interval
    )
    validation_interval = resolve_validation_interval(
        requested_interval,
        len(data.train_dataloader()),
    )
    trainer = build_trainer(
        config,
        output_dir=output_dir,
        accelerator="gpu",
        devices=1,
        max_steps=args.max_steps,
        val_check_interval=validation_interval,
        checkpoint_metric="val/sequence_accuracy",
        checkpoint_filename="best-val-sequence-accuracy",
        checkpoint_mode="max",
    )
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
