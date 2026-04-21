"""Command line interface for uncs-interp."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="uncs-interp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract-features", help="Compute lesion uncertainty and interpretability features.")
    extract.add_argument("--dataset-dir", type=Path, required=True)
    extract.add_argument("--predictions", type=Path, nargs="+", required=True, help="Member/fold prediction folders.")
    extract.add_argument("--output-csv", type=Path, required=True)
    extract.add_argument("--images-subdir", default="imagesTs")
    extract.add_argument("--labels-subdir", default=None)
    extract.add_argument("--class-index", type=int, default=1)
    extract.add_argument("--ensemble-threshold", type=float, default=0.5)
    extract.add_argument("--member-thresholds", type=Path, default=None)
    extract.add_argument("--min-lesion-size", type=int, default=2)
    extract.add_argument("--detection-threshold", type=float, default=0.1)
    extract.add_argument("--detection-method", choices=["iou_adj", "iou", "non_zero"], default="iou_adj")
    extract.add_argument("--atlas-config", type=Path, default=None)
    extract.add_argument("--transforms-dir", type=Path, default=None)
    extract.add_argument("--transform-pattern", default="{case_id}_to_mni.mat")
    extract.add_argument("--fixed-mni", type=Path, default=None)
    extract.add_argument("--register-missing-transforms", action="store_true")
    extract.add_argument("--mriqc-csv", type=Path, default=None)
    extract.add_argument("--skip-radiomics", action="store_true")
    extract.add_argument("--n-dilation", type=int, default=0)

    novelty = subparsers.add_parser("add-novelty", help="Add PCA/Mahalanobis novelty features.")
    novelty.add_argument("--reference-csv", type=Path, required=True)
    novelty.add_argument("--target-csv", type=Path, required=True)
    novelty.add_argument("--output-csv", type=Path, required=True)
    novelty.add_argument("--target-column", default="ddu")

    train = subparsers.add_parser("train-elasticnet", help="Train ElasticNet uncertainty interpretability model.")
    train.add_argument("--train-csv", type=Path, required=True)
    train.add_argument("--model-out", type=Path, required=True)
    train.add_argument("--target-column", default="ddu")
    train.add_argument("--include-iou-adj", action="store_true")
    train.add_argument("--no-group-by-case", action="store_true")
    train.add_argument("--unweighted", action="store_true")
    train.add_argument("--n-jobs", type=int, default=1)
    train.add_argument("--cv-splits", type=int, default=5)

    predict = subparsers.add_parser("predict", help="Predict uncertainty with a trained model.")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--features-csv", type=Path, required=True)
    predict.add_argument("--output-csv", type=Path, required=True)

    counterfactual = subparsers.add_parser("counterfactual", help="Explain feature changes needed to reduce uncertainty.")
    counterfactual.add_argument("--model", type=Path, required=True)
    counterfactual.add_argument("--features-csv", type=Path, required=True)
    counterfactual.add_argument("--output-csv", type=Path, required=True)
    counterfactual.add_argument("--target-uncertainty", type=float, default=0.0)
    counterfactual.add_argument("--top-k", type=int, default=10)
    counterfactual.add_argument(
        "--immutable-pattern",
        action="append",
        default=[],
        help="Regex for features to exclude from counterfactual suggestions. Can be repeated.",
    )

    mriqc_cmd = subparsers.add_parser("mriqc-command", help="Print a Docker command for MRIQC.")
    mriqc_cmd.add_argument("--bids-dir", type=Path, required=True)
    mriqc_cmd.add_argument("--output-dir", type=Path, required=True)
    mriqc_cmd.add_argument("--participant-label", default=None)

    mriqc_select = subparsers.add_parser("select-mriqc", help="Select MRIQC IQMs used by this analysis.")
    mriqc_select.add_argument("--input-csv", type=Path, required=True)
    mriqc_select.add_argument("--output-csv", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "extract-features":
        from .features import FeatureExtractionConfig, extract_features, parse_thresholds

        thresholds = parse_thresholds(args.member_thresholds, len(args.predictions), default=0.5)
        extract_features(
            FeatureExtractionConfig(
                dataset_dir=args.dataset_dir,
                predictions=args.predictions,
                output_csv=args.output_csv,
                images_subdir=args.images_subdir,
                labels_subdir=args.labels_subdir,
                class_index=args.class_index,
                ensemble_threshold=args.ensemble_threshold,
                member_thresholds=thresholds,
                min_lesion_size=args.min_lesion_size,
                detection_threshold=args.detection_threshold,
                detection_method=args.detection_method,
                atlas_config=args.atlas_config,
                transforms_dir=args.transforms_dir,
                transform_pattern=args.transform_pattern,
                fixed_mni=args.fixed_mni,
                register_missing_transforms=args.register_missing_transforms,
                mriqc_csv=args.mriqc_csv,
                skip_radiomics=args.skip_radiomics,
                n_dilation=args.n_dilation,
            )
        )
    elif args.command == "add-novelty":
        import pandas as pd

        from .novelty import add_novelty_features

        result = add_novelty_features(
            reference=pd.read_csv(args.reference_csv),
            target=pd.read_csv(args.target_csv),
            target_column=args.target_column,
        )
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output_csv, index=False)
    elif args.command == "train-elasticnet":
        from .modeling import TrainConfig, train_elasticnet

        train_elasticnet(
            TrainConfig(
                train_csv=args.train_csv,
                model_out=args.model_out,
                target_column=args.target_column,
                include_iou_adj=args.include_iou_adj,
                group_by_case=not args.no_group_by_case,
                weighted=not args.unweighted,
                n_jobs=args.n_jobs,
                cv_splits=args.cv_splits,
            )
        )
    elif args.command == "predict":
        from .modeling import predict_uncertainty

        predict_uncertainty(args.model, args.features_csv, args.output_csv)
    elif args.command == "counterfactual":
        from .counterfactual import (
            DEFAULT_IMMUTABLE_PATTERNS,
            CounterfactualConfig,
            explain_counterfactuals,
        )

        immutable_patterns = tuple(args.immutable_pattern) if args.immutable_pattern else DEFAULT_IMMUTABLE_PATTERNS
        explain_counterfactuals(
            CounterfactualConfig(
                model_path=args.model,
                features_csv=args.features_csv,
                output_csv=args.output_csv,
                target_uncertainty=args.target_uncertainty,
                top_k=args.top_k,
                immutable_patterns=immutable_patterns,
            )
        )
    elif args.command == "mriqc-command":
        from .mriqc import mriqc_docker_command

        print(mriqc_docker_command(args.bids_dir, args.output_dir, args.participant_label))
    elif args.command == "select-mriqc":
        from .mriqc import select_mriqc_features

        select_mriqc_features(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
