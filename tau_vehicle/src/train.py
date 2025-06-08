from datetime import datetime
from pathlib import Path
from typing import List

import git
import hydra
import mlflow
import pytorch_lightning as pl
import torch
from dotenv import load_dotenv
from dvc.api import DVCFileSystem
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger, MLFlowLogger

from tau_vehicle.src.data.module import VehicleDataModule
from tau_vehicle.src.model.huggingface import HuggingFaceModel

###########################
# Supplementary functions #
###########################


def print_abort(message: str):
    """
    Print message and abort.
    """
    print(message)
    print("Aborting...")
    exit(1)


def set_tracking(config: DictConfig):
    """
    Set tracking for MLflow and WandB.
    """
    mlflow_uri = config.mlflow_uri
    debug = config.debug

    # Enter run's name and description
    if not debug:
        run_name = input("Enter run name: ")
        if not run_name:
            print("Run name is empty. Running without tracking.")
            run_description = ""
            debug = True
        else:
            run_description = input("Enter description: ")
    else:
        run_name = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        run_description = ""

    # Set mlflow params
    if not debug:
        # Initialize MLFlow tracking
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(config.experiment)

    # Enable/disable tracking
    tracking = not debug

    return tracking, (run_name, run_description)


def find_next_run_dir(base_dir: str, run_name: str):
    """
    Find directory name for the next run with given name.

    Args:
        base_dir (str): Base directory where results are saved.
        run_name (str): Run name (e.g., 'my_experiment').

    Returns:
         str : Directory name for the next run (e.g., 'my_experiment' or 'my_experiment2').
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        base_path.mkdir(parents=True)
        return run_name

    existing_runs = [d.name for d in base_path.iterdir() if d.is_dir()]
    matching_runs = [d for d in existing_runs if d.startswith(run_name)]
    if not matching_runs:
        return run_name

    if run_name in matching_runs:
        run_numbers = []
        for run in matching_runs:
            try:
                number = int(run.replace(run_name, ""))
                run_numbers.append(number)
            except ValueError:
                continue

        next_run_number = 2
        while next_run_number in run_numbers:
            next_run_number += 1

        return f"{run_name}{next_run_number}"
    else:
        return run_name


def check_uncommitted_changes():
    """
    Check if there are any uncommitted changes in the current git repository.
    Returns True if there are uncommitted changes, False otherwise.
    """
    try:
        # Initialize repository object for current directory
        repo = git.Repo(search_parent_directories=True)

        untracked_files = repo.untracked_files
        modified_files = [item.a_path for item in repo.index.diff(None)]
        staged_files = [item.a_path for item in repo.index.diff("HEAD")]

        # If any of these lists are non-empty, there are uncommitted changes
        has_changes = bool(untracked_files or modified_files or staged_files)

        if has_changes:
            print_abort(
                "There are uncommitted changes in the git repository. "
                "Please commit or stash them before running this script."
            )

    except git.InvalidGitRepositoryError:
        raise RuntimeError("Not a git repository")
    except Exception as e:
        raise RuntimeError(f"Error checking git status: {str(e)}")


def check_if_already_active():
    """
    Check if the run with the given name is already active.
    """
    # Get all active runs
    if mlflow.active_run():
        print(
            f"Run with UUID {mlflow.active_run().info.run_id} is already active. Do you want to stop it? [Y/n]."
        )
        if input().strip() in ["Y", "y"]:
            mlflow.end_run()
        else:
            print_abort(
                "Aborting. You can resume this run with config parameter 'resume=True'."
            )


##################
# Main functions #
##################


def load_model(config: DictConfig):
    if config.model.provider == "huggingface":
        return HuggingFaceModel(config.model, config.hyp)
    else:
        raise ValueError(
            f"Unknown model provider '{config.model.provider}'. Parsing is not implemented"
        )


def load_data(config: DictConfig):
    # Download data
    if not Path(config.data.dir).exists():
        print("- downloading from DVC remote storage")
        fs = DVCFileSystem(".")
        fs.get(config.data.dir, Path(config.data.dir).parent, recursive=True)

    # Create a module
    print("- module initializing")
    return VehicleDataModule(config)


def load_logger(config: DictConfig, run_name: str):
    return MLFlowLogger(
        experiment_name=config.experiment,
        run_name=run_name,
        tracking_uri=config.mlflow_uri,
    )


def load_callbacks(callbacks_config: DictConfig, run_dir: str):
    callbacks = []
    for cb_conf in callbacks_config.values():
        if "dirpath" in cb_conf:
            cb_conf.dirpath = run_dir
        callbacks.append(instantiate(cb_conf))
    return callbacks


def init_trainer(config: DictConfig, logger: Logger, callbacks: List[Callback]):
    use_gpu = torch.cuda.is_available() and config.devices not in ("cpu", None, [])

    trainer = pl.Trainer(
        max_epochs=config.hyp.epochs,
        accelerator="gpu" if use_gpu else "cpu",
        **({"devices": config.devices} if use_gpu else {}),
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
        **config.trainer.args,
    )
    return trainer


@hydra.main(version_base=None, config_path="../../conf", config_name="baseline")
def train(config: DictConfig):
    """
    Train a model.

    Args:
        config : DictConfig - configuration of model, training, logging and overall experiment setup.
    """
    ##################
    # Input run name #
    ##################

    tracking, run_info = set_tracking(config)
    run_name, run_description = run_info

    ############
    # Checking #
    ############

    check_if_already_active()
    # if tracking:
    #    check_uncommitted_changes()

    ##########################
    # Filesystem preparation #
    ##########################

    # Change working directory
    exp_dir = Path(config.store_exp_to) / config.experiment
    new_run_dir_name = find_next_run_dir(exp_dir, run_name)
    new_run_dir = exp_dir / new_run_dir_name
    new_run_dir.mkdir(parents=True, exist_ok=False)

    #############################
    # Experiment initialization #
    #############################

    # Fix seed
    torch.cuda.manual_seed_all(config.seed)
    torch.manual_seed(config.seed)
    pl.seed_everything(config.seed, workers=True)

    # System settings
    torch.set_float32_matmul_precision(config.matmul)
    load_dotenv()

    # Setup
    print("[0/3] Loading model...")
    model = load_model(config)
    print("[1/3] Loading data...")
    data = load_data(config)
    data.setup()

    print("[2/3] Loading trainer...")
    logger = load_logger(config, run_name) if tracking else False
    callbacks = load_callbacks(config.trainer.callbacks, str(new_run_dir))

    trainer = init_trainer(config, logger, callbacks)

    ############
    # Training #
    ############

    print("[3/3] Starting training...")

    run_id = logger.run_id

    try:
        trainer.fit(model, data)
    finally:
        if tracking:
            # Store yolo artifacts
            mlflow.log_artifact(new_run_dir, new_run_dir, run_id=run_id)
            # Stop MLFlow tracking
            trainer.test(model, datamodule=data, ckpt_path="best")


if __name__ == "__main__":
    train()
