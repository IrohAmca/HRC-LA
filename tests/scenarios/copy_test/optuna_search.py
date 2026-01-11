"""
Optuna hyperparameter optimization for HRC-LA Copy Task.

This module provides Optuna integration for finding the best hyperparameter
combinations for the HRC Transformer model on the copy task.
"""

import argparse
import logging
import os
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

import optuna
import torch
from optuna.trial import Trial
from torch.utils.data import DataLoader

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, project_root)

from tests.models import HRCTransformer
from tests.scenarios.common.config import BenchmarkConfig
from tests.scenarios.common.datasets import CopyTaskDataset
from tests.scenarios.common.trainer import (
    count_parameters,
    get_lr_scheduler,
    train_epoch,
    validate,
)

logger = logging.getLogger(__name__)

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not installed. Install with: pip install wandb")


class OptunaHyperparameterSearch:
    """
    Optuna-based hyperparameter search for HRC-LA Copy Task.

    Attributes:
        base_config: Base configuration to use for non-optimized parameters
        search_space: Dictionary defining the search space for each parameter
        n_trials: Number of optimization trials
        study_name: Name of the Optuna study
        storage: Optuna storage URL (SQLite, PostgreSQL, etc.)
        direction: Optimization direction ('maximize' for accuracy, 'minimize' for loss)
        metric: Metric to optimize ('val_acc' or 'val_loss')
    """

    DEFAULT_SEARCH_SPACE = {
        "seq_len": {"type": "int", "low": 64, "high": 1024, "step": 64},
        "d_model": {"type": "categorical", "values": [32, 64, 128, 256]},
        "num_heads": {"type": "categorical", "values": [2, 4, 8]},
        "num_layers": {"type": "int", "low": 1, "high": 6},
        "d_ff": {"type": "categorical", "values": [64, 128, 256, 512]},
        "m_features": {"type": "categorical", "values": [32, 64, 128, 256]},
        "dropout": {"type": "float", "low": 0.0, "high": 0.3, "step": 0.05},
        "batch_size": {"type": "categorical", "values": [32, 64, 128]},
        "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
        "warmup_steps": {"type": "int", "low": 50, "high": 500, "step": 50},
        "learnable_omega_penalty": {"type": "loguniform", "low": 1e-6, "high": 1e-2},
    }

    def __init__(
        self,
        base_config: Optional[BenchmarkConfig] = None,
        search_space: Optional[Dict[str, Dict[str, Any]]] = None,
        n_trials: int = 100,
        study_name: str = "hrc_copy_task_optimization",
        storage: Optional[str] = None,
        direction: str = "maximize",
        metric: str = "val_acc",
        pruner: Optional[optuna.pruners.BasePruner] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        use_wandb: bool = True,
    ):
        """
        Initialize the hyperparameter search.

        Args:
            base_config: Base configuration for non-optimized parameters
            search_space: Custom search space (uses DEFAULT_SEARCH_SPACE if None)
            n_trials: Number of optimization trials
            study_name: Name of the Optuna study
            storage: Database URL for study persistence (e.g., 'sqlite:///optuna.db')
            direction: 'maximize' for accuracy, 'minimize' for loss
            metric: Metric to optimize ('val_acc' or 'val_loss')
            pruner: Optuna pruner for early stopping (default: MedianPruner)
            sampler: Optuna sampler (default: TPESampler)
            use_wandb: Whether to log to Weights & Biases
        """
        self.base_config = base_config or BenchmarkConfig()
        self.search_space = search_space or self.DEFAULT_SEARCH_SPACE.copy()
        self.n_trials = n_trials
        self.study_name = study_name
        self.storage = storage
        self.direction = direction
        self.metric = metric
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        self.pruner = pruner or optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1,
        )

        self.sampler = sampler or optuna.samplers.TPESampler(
            n_startup_trials=10,
            multivariate=True,
        )

        self.device = self.base_config.get_device()
        self.study: Optional[optuna.Study] = None

    def _suggest_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters based on the search space.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested hyperparameters
        """
        params = {}

        for param_name, config in self.search_space.items():
            param_type = config["type"]

            if param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name, config["values"]
                )
            elif param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    config["low"],
                    config["high"],
                    step=config.get("step", 1),
                )
            elif param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    config["low"],
                    config["high"],
                    step=config.get("step"),
                )
            elif param_type == "loguniform":
                params[param_name] = trial.suggest_float(
                    param_name,
                    config["low"],
                    config["high"],
                    log=True,
                )

        if "d_model" in params and "num_heads" in params:
            d_model = params["d_model"]
            num_heads = params["num_heads"]
            if d_model % num_heads != 0:
                valid_heads = [h for h in [1, 2, 4, 8, 16] if d_model % h == 0]
                params["num_heads"] = trial.suggest_categorical(
                    "num_heads_adjusted", valid_heads
                )

        return params

    def _create_config(self, params: Dict[str, Any]) -> BenchmarkConfig:
        """
        Create a BenchmarkConfig from suggested parameters.

        Args:
            params: Dictionary of hyperparameters

        Returns:
            BenchmarkConfig with updated parameters
        """
        config_dict = asdict(self.base_config)
        config_dict.update(params)

        if "num_heads_adjusted" in params:
            config_dict["num_heads"] = params["num_heads_adjusted"]
            del config_dict["num_heads_adjusted"]

        return BenchmarkConfig(**config_dict)

    def objective(self, trial: Trial) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Validation metric value
        """
        params = self._suggest_hyperparameters(trial)
        config = self._create_config(params)

        logger.info(f"Trial {trial.number}: {params}")

        if self.use_wandb:
            wandb.init(
                project="hrc-copy-task-optuna",
                name=f"trial-{trial.number}",
                config=params,
                reinit=True,
            )

        try:
            train_dataset = CopyTaskDataset(
                config.vocab_size, config.seq_len, config.num_samples
            )
            val_dataset = CopyTaskDataset(
                config.vocab_size, config.seq_len, config.num_samples // 10
            )

            train_loader = DataLoader(
                train_dataset, batch_size=config.batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=config.batch_size, shuffle=False
            )

            model = HRCTransformer(
                vocab_size=config.vocab_size,
                d_model=config.d_model,
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                d_ff=config.d_ff,
                m_features=config.m_features,
                dropout=config.dropout,
                learnable_omega=True,
                max_len=config.seq_len + 100,
                max_seq_len=config.seq_len + 100,
            ).to(self.device)

            total_params, trainable_params = count_parameters(model)
            trial.set_user_attr("total_params", total_params)
            trial.set_user_attr("trainable_params", trainable_params)

            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            total_steps = len(train_loader) * config.num_epochs
            scheduler = get_lr_scheduler(optimizer, config.warmup_steps, total_steps)

            best_val_metric = 0.0 if self.direction == "maximize" else float("inf")

            for epoch in range(config.num_epochs):
                start_time = time.time()

                train_loss, train_acc = train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    scheduler,
                    self.device,
                    learnable_omega_penalty=config.learnable_omega_penalty,
                )

                val_loss, val_acc = validate(model, val_loader, self.device)

                epoch_time = time.time() - start_time

                current_metric = val_acc if self.metric == "val_acc" else val_loss

                if self.direction == "maximize":
                    best_val_metric = max(best_val_metric, current_metric)
                else:
                    best_val_metric = min(best_val_metric, current_metric)

                trial.report(current_metric, epoch)

                if self.use_wandb:
                    wandb.log(
                        {
                            "train_loss": train_loss,
                            "train_acc": train_acc,
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                            "epoch_time": epoch_time,
                            "epoch": epoch,
                            "best_val_metric": best_val_metric,
                        }
                    )

                logger.info(
                    f"Trial {trial.number} | Epoch {epoch + 1}/{config.num_epochs} | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
                )

                if trial.should_prune():
                    if self.use_wandb:
                        wandb.finish()
                    raise optuna.TrialPruned()

            if self.use_wandb:
                wandb.log({"final_best_metric": best_val_metric})
                wandb.finish()

            return best_val_metric

        except Exception as e:
            logger.error(f"Trial {trial.number} failed with error: {e}")
            if self.use_wandb:
                wandb.finish()
            raise optuna.TrialPruned()

    def optimize(self, show_progress_bar: bool = True) -> optuna.Study:
        """
        Run the hyperparameter optimization.

        Args:
            show_progress_bar: Whether to show progress bar

        Returns:
            Optuna study object with results
        """
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction=self.direction,
            pruner=self.pruner,
            sampler=self.sampler,
            load_if_exists=True,
        )

        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=show_progress_bar,
            gc_after_trial=True,
        )

        return self.study

    def get_best_params(self) -> Dict[str, Any]:
        """Get the best hyperparameters found."""
        if self.study is None:
            raise ValueError("No study available. Run optimize() first.")
        return self.study.best_params

    def get_best_trial(self) -> optuna.trial.FrozenTrial:
        """Get the best trial."""
        if self.study is None:
            raise ValueError("No study available. Run optimize() first.")
        return self.study.best_trial

    def get_best_config(self) -> BenchmarkConfig:
        """Get the BenchmarkConfig with best hyperparameters."""
        return self._create_config(self.get_best_params())

    def print_results(self):
        """Print optimization results."""
        if self.study is None:
            raise ValueError("No study available. Run optimize() first.")

        print("\n" + "=" * 60)
        print("OPTUNA OPTIMIZATION RESULTS")
        print("=" * 60)

        print(f"\nBest trial: {self.study.best_trial.number}")
        print(f"Best value ({self.metric}): {self.study.best_value:.4f}")

        print("\nBest hyperparameters:")
        for key, value in self.study.best_params.items():
            print(f"  {key}: {value}")

        print("\nTrial statistics:")
        print(f"  Completed trials: {len(self.study.trials)}")
        pruned_trials = [
            t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED
        ]
        print(f"  Pruned trials: {len(pruned_trials)}")
        failed_trials = [
            t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL
        ]
        print(f"  Failed trials: {len(failed_trials)}")

        if self.study.best_trial.user_attrs:
            print("\nBest trial attributes:")
            for key, value in self.study.best_trial.user_attrs.items():
                print(f"  {key}: {value}")

        print("=" * 60)

    def save_results(self, filepath: str):
        """Save optimization results to a file."""
        if self.study is None:
            raise ValueError("No study available. Run optimize() first.")

        import json

        results = {
            "study_name": self.study_name,
            "direction": self.direction,
            "metric": self.metric,
            "n_trials": len(self.study.trials),
            "best_trial_number": self.study.best_trial.number,
            "best_value": self.study.best_value,
            "best_params": self.study.best_params,
            "best_trial_user_attrs": self.study.best_trial.user_attrs,
        }

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {filepath}")


def create_search_space_from_args(args) -> Dict[str, Dict[str, Any]]:
    """Create search space from command line arguments."""
    search_space = {}

    if args.optimize_seq_len:
        search_space["seq_len"] = {"type": "int", "low": 64, "high": 1024, "step": 64}

    if args.optimize_d_model:
        search_space["d_model"] = {"type": "categorical", "values": [32, 64, 128, 256]}

    if args.optimize_num_heads:
        search_space["num_heads"] = {"type": "categorical", "values": [2, 4, 8]}

    if args.optimize_num_layers:
        search_space["num_layers"] = {"type": "int", "low": 1, "high": 6}

    if args.optimize_d_ff:
        search_space["d_ff"] = {"type": "categorical", "values": [64, 128, 256, 512]}

    if args.optimize_m_features:
        search_space["m_features"] = {
            "type": "categorical",
            "values": [32, 64, 128, 256],
        }

    if args.optimize_dropout:
        search_space["dropout"] = {
            "type": "float",
            "low": 0.0,
            "high": 0.3,
            "step": 0.05,
        }

    if args.optimize_batch_size:
        search_space["batch_size"] = {"type": "categorical", "values": [32, 64, 128]}

    if args.optimize_learning_rate:
        search_space["learning_rate"] = {
            "type": "loguniform",
            "low": 1e-5,
            "high": 1e-2,
        }

    if args.optimize_warmup_steps:
        search_space["warmup_steps"] = {
            "type": "int",
            "low": 50,
            "high": 500,
            "step": 50,
        }

    if args.optimize_omega_penalty:
        search_space["learnable_omega_penalty"] = {
            "type": "loguniform",
            "low": 1e-6,
            "high": 1e-2,
        }

    return search_space if search_space else None


def main():
    """Main entry point for hyperparameter optimization."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter optimization for HRC-LA Copy Task"
    )

    parser.add_argument(
        "--n_trials", type=int, default=100, help="Number of optimization trials"
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="hrc_copy_task_optimization",
        help="Name of the Optuna study",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Database URL for study persistence (e.g., sqlite:///optuna.db)",
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["maximize", "minimize"],
        default="maximize",
        help="Optimization direction",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["val_acc", "val_loss"],
        default="val_acc",
        help="Metric to optimize",
    )

    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length (used when not optimizing seq_len)")
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="Number of training epochs per trial"
    )
    parser.add_argument(
        "--num_samples", type=int, default=10000, help="Number of training samples"
    )
    parser.add_argument("--vocab_size", type=int, default=64, help="Vocabulary size")

    parser.add_argument(
        "--optimize_seq_len", action="store_true", help="Optimize seq_len parameter"
    )
    parser.add_argument(
        "--optimize_d_model", action="store_true", help="Optimize d_model parameter"
    )
    parser.add_argument(
        "--optimize_num_heads", action="store_true", help="Optimize num_heads parameter"
    )
    parser.add_argument(
        "--optimize_num_layers",
        action="store_true",
        help="Optimize num_layers parameter",
    )
    parser.add_argument(
        "--optimize_d_ff", action="store_true", help="Optimize d_ff parameter"
    )
    parser.add_argument(
        "--optimize_m_features",
        action="store_true",
        help="Optimize m_features parameter",
    )
    parser.add_argument(
        "--optimize_dropout", action="store_true", help="Optimize dropout parameter"
    )
    parser.add_argument(
        "--optimize_batch_size",
        action="store_true",
        help="Optimize batch_size parameter",
    )
    parser.add_argument(
        "--optimize_learning_rate",
        action="store_true",
        help="Optimize learning_rate parameter",
    )
    parser.add_argument(
        "--optimize_warmup_steps",
        action="store_true",
        help="Optimize warmup_steps parameter",
    )
    parser.add_argument(
        "--optimize_omega_penalty",
        action="store_true",
        help="Optimize learnable_omega_penalty parameter",
    )
    parser.add_argument(
        "--optimize_all", action="store_true", help="Optimize all parameters"
    )

    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument(
        "--save_results", type=str, default=None, help="Path to save results JSON"
    )
    parser.add_argument(
        "--no_progress_bar", action="store_true", help="Disable progress bar"
    )

    args = parser.parse_args()

    base_config = BenchmarkConfig(
        seq_len=args.seq_len,
        num_epochs=args.num_epochs,
        num_samples=args.num_samples,
        vocab_size=args.vocab_size,
    )

    if args.optimize_all:
        search_space = None
    else:
        search_space = create_search_space_from_args(args)
        if not search_space:
            logger.warning(
                "No parameters selected for optimization. Using default search space."
            )
            search_space = None

    optimizer = OptunaHyperparameterSearch(
        base_config=base_config,
        search_space=search_space,
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage=args.storage,
        direction=args.direction,
        metric=args.metric,
        use_wandb=not args.no_wandb,
    )

    logger.info("Starting Optuna hyperparameter optimization...")
    logger.info(f"Number of trials: {args.n_trials}")
    logger.info(f"Direction: {args.direction}")
    logger.info(f"Metric: {args.metric}")

    optimizer.optimize(show_progress_bar=not args.no_progress_bar)

    optimizer.print_results()

    if args.save_results:
        optimizer.save_results(args.save_results)

    best_config = optimizer.get_best_config()
    print("\nBest BenchmarkConfig:")
    print("config = BenchmarkConfig(")
    for key, value in asdict(best_config).items():
        if isinstance(value, str):
            print(f"    {key}='{value}',")
        else:
            print(f"    {key}={value},")
    print(")")


if __name__ == "__main__":
    main()
