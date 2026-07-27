"""Evaluate a SemiNovo checkpoint with the NovoBench protocol."""

import argparse
import csv
import json
from pathlib import Path

import torch

from seminovo.checkpoints import load_model
from seminovo.config import Config
from seminovo.data import NovoBenchDataModule
from seminovo.metrics import peptide_precision_auc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/seminovo.yaml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--beams", type=int, default=5)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument(
        "--decode-strategy",
        choices=("casanovo", "mass_pruned"),
        default="mass_pruned",
        help="Use official Casanovo beam expansion or next-token mass pruning.",
    )
    return parser.parse_args()


def fp32_decoder_scores(_module, _inputs, output):
    """Match NovoBench's FP32 beam cache while retaining AMP model execution."""
    logits, tokens = output
    return logits.float(), tokens


def evaluate_checkpoint(args):
    config = Config(args.config)
    data = NovoBenchDataModule(
        args.data_dir,
        config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data.setup("test")
    loader = data.test_dataloader()

    model = load_model(args.checkpoint)
    if hasattr(model, "configure_decode_strategy"):
        model.configure_decode_strategy(args.decode_strategy)
    model.n_beams = args.beams
    model.top_match = 1
    model.eval().cuda()
    model.decoder.register_forward_hook(fp32_decoder_scores)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.csv"
    truths, predictions, scores = [], [], []
    with prediction_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("peptides_true", "peptides_pred", "peptides_score"))
        for batch_index, (spectra, precursors, sequences) in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            spectra = spectra.cuda(non_blocking=True)
            precursors = precursors.cuda(non_blocking=True)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                batch_predictions = model(spectra, precursors)
            for truth, candidates in zip(sequences.tolist(), batch_predictions):
                if candidates:
                    score, _, prediction = candidates[0]
                    score = float(score)
                else:
                    prediction, score = "", float("-inf")
                truth = str(truth)
                writer.writerow((truth, prediction, score))
                truths.append(truth)
                predictions.append(prediction)
                scores.append(score)
            handle.flush()

    metrics = peptide_precision_auc(truths, predictions, scores)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def main():
    evaluate_checkpoint(parse_args())


if __name__ == "__main__":
    main()
