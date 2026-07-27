"""Lightning trainer construction for SemiNovo."""

from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger


def build_trainer(
    config,
    output_dir,
    accelerator="auto",
    devices=1,
    max_steps=None,
    val_check_interval=None,
    callbacks=None,
    checkpoint_metric=None,
    checkpoint_filename=None,
    checkpoint_mode=None,
):
    """Build one trainer while preserving NovoBench scheduling semantics."""
    output_dir = Path(output_dir)
    checkpoint_specs = (
        [
            (
                checkpoint_metric,
                checkpoint_filename or "best-validation-metric",
                checkpoint_mode or "max",
            )
        ]
        if checkpoint_metric is not None
        else [
            ("val/loss", "best-val-loss", "min"),
            (
                "val/sequence_accuracy",
                "best-val-sequence-accuracy",
                "max",
            ),
        ]
    )
    checkpoint_callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename=filename,
            auto_insert_metric_name=False,
            monitor=monitor,
            mode=mode,
            save_top_k=1,
            save_last=False,
            save_weights_only=False,
            save_on_train_epoch_end=False,
        )
        for monitor, filename, mode in checkpoint_specs
    ]
    logger = CSVLogger(output_dir, name="csv")
    trainer_callbacks = list(checkpoint_callbacks)
    trainer_callbacks.extend(callbacks or [])
    return pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision="bf16-mixed" if accelerator != "cpu" else "32-true",
        max_epochs=config.max_epochs,
        max_steps=-1 if max_steps is None else max_steps,
        num_sanity_val_steps=config.num_sanity_val_steps,
        val_check_interval=(
            config.val_check_interval
            if val_check_interval is None
            else val_check_interval
        ),
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        callbacks=trainer_callbacks,
        logger=logger,
        log_every_n_steps=10,
        enable_checkpointing=True,
        benchmark=True,
        gradient_clip_val=1.0,
    )
