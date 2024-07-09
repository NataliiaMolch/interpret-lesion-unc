import os.path
import torch
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
import pandas as pd
import logging, sys


class LoggingCallback:
    def __init__(self, logger, validator, print_name, print_freq: int = 1):
        self.validator = validator
        self.logger = logger
        self.print_freq = print_freq
        self.name = print_name

    def __call__(self, trainer, *args, **kwargs):
        if trainer.state.epoch % self.print_freq == 0:
            msg = self.validator(trainer.network)
            msg.update({'set name': self.name, 'epoch': trainer.state.epoch})
            self.logger.write(msg)


class PlottingCallback:
    def __init__(self, logger, print_freq: int = 1):
        self.log_file = logger.log_file
        self.save_path = os.path.dirname(self.log_file)
        self.print_freq = print_freq

    def __call__(self, trainer, *args, **kwargs):
        if self.log_file is not None and trainer.state.epoch % self.print_freq == 0:
            df = pd.read_csv(self.log_file)
            metrics_names = [c for c in list(df.columns) if c not in ['set name', 'epoch']]
            for metric_name in metrics_names:
                for name in df['set name'].unique():
                    df_mn = df[df['set name'] == name]
                    plt.plot(df_mn['epoch'], df_mn[metric_name], label=name)
                    plt.title(metric_name)
                    plt.xlabel("Epoch")
                    plt.ylabel(metric_name)
                plt.legend()
                plt.savefig(os.path.join(self.save_path, metric_name + '_history.jpg'), dpi=300)
                plt.close()


class SaveModelCallback:
    def __init__(self, save_freq: int, save_path: str, train_log_file: str):
        self.save_freq = save_freq
        self.save_path = save_path
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(train_log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def __call__(self, trainer, *args, **kwargs):
        epoch = trainer.state.epoch if trainer.continue_training else 'last'
        if trainer.state.epoch % self.save_freq == 0 or epoch == 'last':
            torch.save(trainer.network.state_dict(), os.path.join(self.save_path, f"model_epoch_{epoch}.pth"))
            logging.info(f"Saved model at epoch {trainer.state.epoch}")


class PlotLRCallback:
    def __init__(self, logger):
        self.log_file = logger.log_file
        self.filepath = os.path.join(os.path.dirname(self.log_file), "LR_history.jpg")

    def __call__(self, trainer, *args, **kwargs):
        n_iters = trainer.state.epoch * len(trainer.data_loader)
        plt.plot(np.linspace(1, trainer.state.epoch, n_iters), trainer.lrs_history)
        plt.xlabel("Epochs")
        plt.ylabel("Learning rate")
        plt.savefig(self.filepath, dpi=300)
        plt.close()


class GeneralLoggingCallback:
    def __init__(self, logger, validator, print_name: str, patience: int, tolerance: float, min_is_good: bool,
                 metric: str, train_log_file: str):
        """ Includes:
        - Validation on val set
        - Early stopping + Saving of the best model
        """
        self.name = print_name
        self.patience = patience
        self.tolerance = tolerance
        self.validator = validator
        self.silent_epochs = 0
        self.mult_factor = 1 if min_is_good else -1
        self.best_metric = np.inf if min_is_good else -np.inf
        self.es_metric = metric
        self.logger = logger
        self.save_dir = os.path.dirname(self.logger.log_file)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(train_log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def __call__(self, trainer, *args, **kwargs):
        metrics_row: dict = self.validator(trainer.network)
        # log validation metrics
        metrics_row.update({'epoch': trainer.state.epoch, 'set name': self.name})
        self.logger.write(metrics_row)
        # es
        metric_val = metrics_row[self.es_metric]
        if self.mult_factor * metric_val > self.mult_factor * self.best_metric or \
                np.abs(metric_val - self.best_metric) < self.tolerance:
            self.silent_epochs += 1
            if self.silent_epochs > self.patience:
                logging.info(f"Early stopping at epoch {trainer.state.epoch + 1}")
                trainer.continue_training = False
        else:
            self.best_metric = metric_val
            self.silent_epochs = 0
            torch.save(trainer.network.state_dict(),
                       os.path.join(self.save_dir, f"model_epoch_{trainer.state.epoch}.pth"))
            logging.info(f"Saved best model at epoch {trainer.state.epoch}")

        return metric_val



