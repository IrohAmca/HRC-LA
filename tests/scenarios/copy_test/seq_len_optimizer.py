"""
Optuna optimization for finding optimal seq_len with minimal m_features.

Goal: Maximize sequence length while keeping m_features small and maintaining
acceptable model accuracy. Other parameters should remain reasonable.
"""

import argparse
import logging
import os
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

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


class SeqLenOptimizer:
    """
    Multi-objective optimizer to find maximum seq_len with minimal m_features.

    Objectives:
    1. Maximize seq_len
    2. Minimize m_features
    3. Keep total parameters under control
    4. Maintain acceptable accuracy (above threshold)

    Strategy:
    - Uses Optuna's multi-objective optimization (NSGA-II sampler)
    - Returns Pareto-optimal solutions
    """

    DEFAULT_SEARCH_SPACE = {
        # Primary optimization targets
        "seq_len": {"type": "int", "low": 64, "high": 1024, "step": 64},
        "m_features": {"type": "categorical", "values": [16, 32, 48, 64, 128, 256]},
        # Secondary params - kept small/reasonable
        "d_model": {"type": "categorical", "values": [32, 64, 96, 128]},
        "num_heads": {"type": "categorical", "values": [2, 4, 8]},
        "num_layers": {"type": "categorical", "values": [1, 2, 3]},
        "d_ff": {"type": "categorical", "values": [64, 128, 256]},
        # Training params
        "dropout": {"type": "float", "low": 0.0, "high": 0.2, "step": 0.05},
        "learning_rate": {"type": "loguniform", "low": 5e-5, "high": 5e-3},
        "batch_size": {"type": "categorical", "values": [16, 32, 64]},
    }

    def __init__(
        self,
        n_trials: int = 100,
        accuracy_threshold: float = 0.85,
        max_total_params: int = 500_000,
        num_epochs: int = 15,
        num_samples: int = 5000,
        vocab_size: int = 64,
        study_name: str = "seq_len_optimization",
        storage: Optional[str] = None,
        search_space: Optional[Dict[str, Dict[str, Any]]] = None,
        use_wandb: bool = True,
    ):
        """
        Initialize the seq_len optimizer.

        Args:
            n_trials: Number of optimization trials
            accuracy_threshold: Minimum acceptable validation accuracy
            max_total_params: Maximum allowed total parameters
            num_epochs: Training epochs per trial
            num_samples: Training samples
            vocab_size: Vocabulary size
            study_name: Name of the Optuna study
            storage: Database URL for persistence
            search_space: Custom search space
            use_wandb: Whether to log to wandb
        """
        self.n_trials = n_trials
        self.accuracy_threshold = accuracy_threshold
        self.max_total_params = max_total_params
        self.num_epochs = num_epochs
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.study_name = study_name
        self.storage = storage
        self.search_space = search_space or self.DEFAULT_SEARCH_SPACE.copy()
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.study: Optional[optuna.Study] = None

        self.best_results: List[Dict[str, Any]] = []

    def _suggest_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        """Suggest hyperparameters from the search space."""
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
                valid_heads = [h for h in [1, 2, 4, 8] if d_model % h == 0]
                params["num_heads"] = valid_heads[-1]

        return params

    def _train_and_evaluate(
        self, trial: Trial, params: Dict[str, Any]
    ) -> Tuple[float, int]:
        """
        Train model and return validation accuracy and total params.

        Returns:
            Tuple of (best_val_accuracy, total_parameters)
        """
        config = BenchmarkConfig(
            vocab_size=self.vocab_size,
            seq_len=params["seq_len"],
            d_model=params["d_model"],
            num_heads=params["num_heads"],
            num_layers=params["num_layers"],
            d_ff=params["d_ff"],
            m_features=params["m_features"],
            dropout=params["dropout"],
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
            num_epochs=self.num_epochs,
            num_samples=self.num_samples,
        )

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

        if total_params > self.max_total_params:
            logger.info(
                f"Trial {trial.number}: Skipping - {total_params:,} params > "
                f"{self.max_total_params:,} max"
            )
            return 0.0, total_params

        trial.set_user_attr("total_params", total_params)
        trial.set_user_attr("trainable_params", trainable_params)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        total_steps = len(train_loader) * config.num_epochs
        scheduler = get_lr_scheduler(optimizer, config.warmup_steps, total_steps)

        best_val_acc = 0.0

        for epoch in range(config.num_epochs):
            train_loss, train_acc = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                self.device,
                learnable_omega_penalty=config.learnable_omega_penalty,
            )

            val_loss, val_acc = validate(model, val_loader, self.device)
            best_val_acc = max(best_val_acc, val_acc)

            trial.report(val_acc, epoch)

            if self.use_wandb:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "best_val_acc": best_val_acc,
                        "epoch": epoch,
                    }
                )

            logger.info(
                f"Trial {trial.number} | Epoch {epoch + 1}/{config.num_epochs} | "
                f"Val Acc: {val_acc:.4f} | Best: {best_val_acc:.4f}"
            )

            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val_acc, total_params

    def objective(self, trial: Trial) -> Tuple[float, float, float]:
        """
        Multi-objective function.

        Returns:
            Tuple of (seq_len, -m_features, accuracy) for maximization
            Note: m_features is negated because we want to minimize it
        """
        params = self._suggest_hyperparameters(trial)

        logger.info(
            f"\nTrial {trial.number}: seq_len={params['seq_len']}, "
            f"m_features={params['m_features']}, d_model={params['d_model']}, "
            f"num_layers={params['num_layers']}"
        )

        if self.use_wandb:
            wandb.init(
                project="hrc-seqlen-optimization",
                name=f"trial-{trial.number}",
                config=params,
                reinit=True,
            )

        try:
            val_acc, total_params = self._train_and_evaluate(trial, params)

            trial.set_user_attr("val_acc", val_acc)
            trial.set_user_attr("seq_len", params["seq_len"])
            trial.set_user_attr("m_features", params["m_features"])

            if self.use_wandb:
                wandb.log(
                    {
                        "final_val_acc": val_acc,
                        "total_params": total_params,
                        "seq_len": params["seq_len"],
                        "m_features": params["m_features"],
                    }
                )
                wandb.finish()

            return float(params["seq_len"]), float(params["m_features"]), val_acc

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            if self.use_wandb:
                wandb.finish()
            raise optuna.TrialPruned()

    def optimize(self, show_progress_bar: bool = True) -> optuna.Study:
        """
        Run the multi-objective optimization.

        Returns:
            Optuna study with Pareto-optimal solutions
        """
        sampler = optuna.samplers.NSGAIISampler(
            population_size=20,
            mutation_prob=0.1,
            crossover_prob=0.9,
        )

        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            directions=[
                "maximize",
                "minimize",
                "maximize",
            ],
            sampler=sampler,
            load_if_exists=True,
        )

        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=show_progress_bar,
            gc_after_trial=True,
        )

        return self.study

    def get_pareto_solutions(
        self, min_accuracy: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get Pareto-optimal solutions filtered by minimum accuracy.

        Args:
            min_accuracy: Minimum accuracy threshold (uses self.accuracy_threshold if None)

        Returns:
            List of Pareto-optimal configurations
        """
        if self.study is None:
            raise ValueError("No study available. Run optimize() first.")

        min_acc = min_accuracy or self.accuracy_threshold

        solutions = []
        for trial in self.study.best_trials:
            seq_len, m_features, val_acc = trial.values

            if val_acc >= min_acc:
                solutions.append(
                    {
                        "trial_number": trial.number,
                        "seq_len": int(seq_len),
                        "m_features": int(m_features),
                        "val_acc": val_acc,
                        "params": trial.params,
                        "user_attrs": trial.user_attrs,
                    }
                )

        solutions.sort(key=lambda x: (-x["seq_len"], x["m_features"]))

        return solutions

    def get_best_efficiency_solution(
        self, min_accuracy: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get the solution with best seq_len/m_features ratio above accuracy threshold.

        Args:
            min_accuracy: Minimum accuracy threshold

        Returns:
            Best efficiency configuration or None
        """
        solutions = self.get_pareto_solutions(min_accuracy)

        if not solutions:
            return None

        for sol in solutions:
            sol["efficiency"] = sol["seq_len"] / sol["m_features"]

        return max(solutions, key=lambda x: x["efficiency"])

    def print_results(self):
        """Print optimization results."""
        if self.study is None:
            raise ValueError("No study available. Run optimize() first.")

        print("\n" + "=" * 70)
        print("SEQ_LEN OPTIMIZATION RESULTS")
        print("=" * 70)

        print(f"\nTotal trials: {len(self.study.trials)}")
        print(f"Pareto-optimal trials: {len(self.study.best_trials)}")

        solutions = self.get_pareto_solutions()

        if solutions:
            print(f"\n{'=' * 70}")
            print(f"PARETO-OPTIMAL SOLUTIONS (accuracy >= {self.accuracy_threshold})")
            print(f"{'=' * 70}")
            print(
                f"{'Trial':>6} | {'SeqLen':>7} | {'m_feat':>6} | {'Acc':>6} | {'Eff':>8}"
            )
            print("-" * 70)

            for sol in solutions:
                eff = sol["seq_len"] / sol["m_features"]
                print(
                    f"{sol['trial_number']:>6} | {sol['seq_len']:>7} | "
                    f"{sol['m_features']:>6} | {sol['val_acc']:>6.4f} | {eff:>8.2f}"
                )

            best_eff = self.get_best_efficiency_solution()
            if best_eff:
                print(f"\n{'=' * 70}")
                print("BEST EFFICIENCY SOLUTION")
                print(f"{'=' * 70}")
                print(f"  seq_len: {best_eff['seq_len']}")
                print(f"  m_features: {best_eff['m_features']}")
                print(
                    f"  efficiency (seq_len/m_features): {best_eff['efficiency']:.2f}"
                )
                print(f"  val_acc: {best_eff['val_acc']:.4f}")
                print(f"\n  Full params:")
                for k, v in best_eff["params"].items():
                    print(f"    {k}: {v}")
        else:
            print(f"\nNo solutions found with accuracy >= {self.accuracy_threshold}")
            print("Try lowering accuracy_threshold or running more trials")

        print("=" * 70)

    def save_results(self, filepath: str):
        """Save results to JSON file."""
        if self.study is None:
            raise ValueError("No study available. Run optimize() first.")

        import json

        solutions = self.get_pareto_solutions(min_accuracy=0.0)
        best_eff = self.get_best_efficiency_solution()

        results = {
            "study_name": self.study_name,
            "n_trials": len(self.study.trials),
            "accuracy_threshold": self.accuracy_threshold,
            "max_total_params": self.max_total_params,
            "pareto_solutions": solutions,
            "best_efficiency_solution": best_eff,
        }

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {filepath}")


class SingleObjectiveSeqLenOptimizer:
    """
    Single-objective optimizer using a weighted score.

    Score = seq_len * (1 / m_features) * accuracy^2 * param_penalty

    This rewards: longer sequences, smaller m_features, higher accuracy
    """

    DEFAULT_SEARCH_SPACE = {
        "seq_len": {"type": "int", "low": 64, "high": 2048, "step": 64},
        "m_features": {"type": "categorical", "values": [8, 16, 32, 48, 64]},
        "d_model": {"type": "categorical", "values": [32, 64, 96, 128]},
        "num_heads": {"type": "categorical", "values": [2, 4]},
        "num_layers": {"type": "categorical", "values": [1, 2, 3]},
        "d_ff": {"type": "categorical", "values": [64, 128, 256]},
        "dropout": {"type": "float", "low": 0.0, "high": 0.2, "step": 0.05},
        "learning_rate": {"type": "loguniform", "low": 5e-5, "high": 5e-3},
        "batch_size": {"type": "categorical", "values": [16, 32, 64]},
    }

    def __init__(
        self,
        n_trials: int = 100,
        accuracy_threshold: float = 0.85,
        max_total_params: int = 500_000,
        num_epochs: int = 15,
        num_samples: int = 5000,
        vocab_size: int = 64,
        study_name: str = "seq_len_single_objective",
        storage: Optional[str] = None,
        search_space: Optional[Dict[str, Dict[str, Any]]] = None,
        use_wandb: bool = True,
        seq_len_weight: float = 1.0,
        m_features_penalty: float = 1.0,
        accuracy_weight: float = 2.0,
        param_penalty_weight: float = 0.5,
    ):
        """
        Initialize single-objective optimizer.

        Args:
            seq_len_weight: Weight for seq_len in score
            m_features_penalty: How much to penalize larger m_features
            accuracy_weight: Exponent for accuracy (higher = more important)
            param_penalty_weight: How much to penalize total parameters
        """
        self.n_trials = n_trials
        self.accuracy_threshold = accuracy_threshold
        self.max_total_params = max_total_params
        self.num_epochs = num_epochs
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.study_name = study_name
        self.storage = storage
        self.search_space = search_space or self.DEFAULT_SEARCH_SPACE.copy()
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        self.seq_len_weight = seq_len_weight
        self.m_features_penalty = m_features_penalty
        self.accuracy_weight = accuracy_weight
        self.param_penalty_weight = param_penalty_weight

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.study: Optional[optuna.Study] = None

    def _suggest_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        """Suggest hyperparameters from the search space."""
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
                valid_heads = [h for h in [1, 2, 4, 8] if d_model % h == 0]
                params["num_heads"] = valid_heads[-1]

        return params

    def _compute_score(
        self,
        seq_len: int,
        m_features: int,
        val_acc: float,
        total_params: int,
    ) -> float:
        """
        Compute composite optimization score.

        Higher is better.
        """
        seq_len_normalized = seq_len / 2048.0

        m_features_normalized = 1.0 / (m_features / 8.0)

        param_ratio = total_params / self.max_total_params
        param_penalty = max(0.0, 1.0 - param_ratio * self.param_penalty_weight)

        score = (
            (seq_len_normalized**self.seq_len_weight)
            * (m_features_normalized**self.m_features_penalty)
            * (val_acc**self.accuracy_weight)
            * param_penalty
        )

        if val_acc < self.accuracy_threshold:
            score *= 0.1

        return score

    def objective(self, trial: Trial) -> float:
        """Single-objective function returning composite score."""
        params = self._suggest_hyperparameters(trial)

        logger.info(
            f"\nTrial {trial.number}: seq_len={params['seq_len']}, "
            f"m_features={params['m_features']}, d_model={params['d_model']}"
        )

        if self.use_wandb:
            wandb.init(
                project="hrc-seqlen-single-obj",
                name=f"trial-{trial.number}",
                config=params,
                reinit=True,
            )

        try:
            config = BenchmarkConfig(
                vocab_size=self.vocab_size,
                seq_len=params["seq_len"],
                d_model=params["d_model"],
                num_heads=params["num_heads"],
                num_layers=params["num_layers"],
                d_ff=params["d_ff"],
                m_features=params["m_features"],
                dropout=params["dropout"],
                batch_size=params["batch_size"],
                learning_rate=params["learning_rate"],
                num_epochs=self.num_epochs,
                num_samples=self.num_samples,
            )

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

            total_params, _ = count_parameters(model)
            trial.set_user_attr("total_params", total_params)

            if total_params > self.max_total_params:
                logger.info(f"Trial {trial.number}: Too many params ({total_params:,})")
                if self.use_wandb:
                    wandb.finish()
                return 0.0

            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            total_steps = len(train_loader) * config.num_epochs
            scheduler = get_lr_scheduler(optimizer, config.warmup_steps, total_steps)

            best_val_acc = 0.0

            for epoch in range(config.num_epochs):
                train_loss, train_acc = train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    scheduler,
                    self.device,
                    learnable_omega_penalty=config.learnable_omega_penalty,
                )

                val_loss, val_acc = validate(model, val_loader, self.device)
                best_val_acc = max(best_val_acc, val_acc)

                current_score = self._compute_score(
                    params["seq_len"], params["m_features"], best_val_acc, total_params
                )
                trial.report(current_score, epoch)

                if self.use_wandb:
                    wandb.log(
                        {
                            "train_loss": train_loss,
                            "train_acc": train_acc,
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                            "best_val_acc": best_val_acc,
                            "current_score": current_score,
                            "epoch": epoch,
                        }
                    )

                logger.info(
                    f"Trial {trial.number} | Epoch {epoch + 1} | "
                    f"Val Acc: {val_acc:.4f} | Score: {current_score:.6f}"
                )

                if trial.should_prune():
                    raise optuna.TrialPruned()

            final_score = self._compute_score(
                params["seq_len"], params["m_features"], best_val_acc, total_params
            )

            trial.set_user_attr("val_acc", best_val_acc)
            trial.set_user_attr("seq_len", params["seq_len"])
            trial.set_user_attr("m_features", params["m_features"])
            trial.set_user_attr("final_score", final_score)

            if self.use_wandb:
                wandb.log(
                    {
                        "final_score": final_score,
                        "final_val_acc": best_val_acc,
                    }
                )
                wandb.finish()

            logger.info(
                f"Trial {trial.number} COMPLETE | "
                f"seq_len={params['seq_len']}, m_features={params['m_features']}, "
                f"acc={best_val_acc:.4f}, score={final_score:.6f}"
            )

            return final_score

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            if self.use_wandb:
                wandb.finish()
            raise optuna.TrialPruned()

    def optimize(self, show_progress_bar: bool = True) -> optuna.Study:
        """Run optimization."""
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=10,
            multivariate=True,
        )

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
        )

        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=show_progress_bar,
            gc_after_trial=True,
        )

        return self.study

    def print_results(self):
        """Print optimization results."""
        if self.study is None:
            raise ValueError("No study available. Run optimize() first.")

        print("\n" + "=" * 70)
        print("SINGLE-OBJECTIVE SEQ_LEN OPTIMIZATION RESULTS")
        print("=" * 70)

        print(f"\nTotal trials: {len(self.study.trials)}")
        print(f"Best score: {self.study.best_value:.6f}")

        best = self.study.best_trial
        print(f"\nBest trial: {best.number}")
        print(f"  seq_len: {best.user_attrs.get('seq_len', 'N/A')}")
        print(f"  m_features: {best.user_attrs.get('m_features', 'N/A')}")
        print(f"  val_acc: {best.user_attrs.get('val_acc', 'N/A'):.4f}")
        print(f"  total_params: {best.user_attrs.get('total_params', 'N/A'):,}")

        print(f"\nBest hyperparameters:")
        for k, v in best.params.items():
            print(f"  {k}: {v}")

        completed = [
            t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        completed.sort(key=lambda t: t.value, reverse=True)

        print(f"\n{'=' * 70}")
        print("TOP 5 TRIALS")
        print(f"{'=' * 70}")
        print(
            f"{'Trial':>6} | {'SeqLen':>7} | {'m_feat':>6} | {'Acc':>6} | {'Score':>10}"
        )
        print("-" * 70)

        for t in completed[:5]:
            print(
                f"{t.number:>6} | "
                f"{t.user_attrs.get('seq_len', 'N/A'):>7} | "
                f"{t.user_attrs.get('m_features', 'N/A'):>6} | "
                f"{t.user_attrs.get('val_acc', 0):.4f} | "
                f"{t.value:.6f}"
            )

        print("=" * 70)

    def save_results(self, filepath: str):
        """Save results to JSON."""
        if self.study is None:
            raise ValueError("No study available. Run optimize() first.")

        import json

        completed = [
            t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        completed.sort(key=lambda t: t.value, reverse=True)

        results = {
            "study_name": self.study_name,
            "n_trials": len(self.study.trials),
            "best_score": self.study.best_value,
            "best_params": self.study.best_params,
            "best_user_attrs": self.study.best_trial.user_attrs,
            "top_trials": [
                {
                    "trial_number": t.number,
                    "score": t.value,
                    "params": t.params,
                    "user_attrs": t.user_attrs,
                }
                for t in completed[:10]
            ],
        }

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {filepath}")


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Optimize seq_len with minimal m_features for HRC-LA"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["multi", "single"],
        default="single",
        help="Optimization mode: 'multi' for multi-objective, 'single' for weighted score",
    )

    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--vocab_size", type=int, default=64)
    parser.add_argument("--accuracy_threshold", type=float, default=0.85)
    parser.add_argument("--max_total_params", type=int, default=500_000)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--save_results", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--no_progress_bar", action="store_true")

    parser.add_argument("--seq_len_weight", type=float, default=1.0)
    parser.add_argument("--m_features_penalty", type=float, default=1.0)
    parser.add_argument("--accuracy_weight", type=float, default=2.0)

    args = parser.parse_args()

    if args.mode == "multi":
        optimizer = SeqLenOptimizer(
            n_trials=args.n_trials,
            accuracy_threshold=args.accuracy_threshold,
            max_total_params=args.max_total_params,
            num_epochs=args.num_epochs,
            num_samples=args.num_samples,
            vocab_size=args.vocab_size,
            storage=args.storage,
            use_wandb=not args.no_wandb,
        )
    else:
        optimizer = SingleObjectiveSeqLenOptimizer(
            n_trials=args.n_trials,
            accuracy_threshold=args.accuracy_threshold,
            max_total_params=args.max_total_params,
            num_epochs=args.num_epochs,
            num_samples=args.num_samples,
            vocab_size=args.vocab_size,
            storage=args.storage,
            use_wandb=not args.no_wandb,
            seq_len_weight=args.seq_len_weight,
            m_features_penalty=args.m_features_penalty,
            accuracy_weight=args.accuracy_weight,
        )

    logger.info(f"Starting {args.mode}-objective optimization...")
    logger.info(
        f"Goal: Maximize seq_len, minimize m_features, maintain accuracy >= {args.accuracy_threshold}"
    )

    optimizer.optimize(show_progress_bar=not args.no_progress_bar)
    optimizer.print_results()

    if args.save_results:
        optimizer.save_results(args.save_results)


if __name__ == "__main__":
    main()
