import logging, sys
import torch
from callbacks import GeneralLoggingCallback
from losses import BlobNormalisedDiceLoss, DetectionLoss


class State:
    def __init__(self):
        self.epoch = 0
        self.loss = 0
        self.iteration = 0
        self.metrics = 0


class Trainer:
    def __init__(self, data_loader, network, optimiser, scheduler, loss_function, max_epochs, min_lr, device,
                 print_freq: int, train_log_file: str,
                 inputs_key="inputs", targets_key="targets", instance_key="instance_mask",
                 callbacks=None, warmup_iters: int = 0):
        """
        :param data_loader: monai DataLoader
        :param network: monai / pytorch model
        :param optimiser: pytorch optimiser
        :param scheduler: pytorch scheduler
        :param loss_function: loss function (inputs, targets)
        :param max_epochs: max number of epochs after which training stops
        :param min_lr: minimum learning rate after which scheduler is not called
        :param inputs_key: input key in the monai style dict-like batch
        :param targets_key: target key in the monai style dict-like batch
        :param callbacks: a list of callbacks called after each epoch
        """
        if callbacks is None:
            callbacks = []
        self.data_loader = data_loader
        self.network = network
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.max_epoch = max_epochs
        self.min_lr = min_lr
        self.inputs_key = inputs_key
        self.targets_key = targets_key
        self.instance_key = instance_key
        self.callbacks = callbacks
        self.continue_training = True
        self.device = device
        self.print_freq = print_freq
        self.warmup_iters = warmup_iters
        if warmup_iters > 0.0:
            self.warmup = torch.optim.lr_scheduler.LambdaLR(self.optimiser, lr_lambda=lambda ep: ep / self.warmup_iters)
            self.warming_up = True
        else:
            self.warming_up = False

        self.state = State()
        self.loss_history = []
        self.lrs_history = []

        self.send_instmask = isinstance(loss_function, (BlobNormalisedDiceLoss, DetectionLoss))
        self.plateau_scheduler = isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(train_log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def run(self):
        """ Run training for the maximum amount of epochs """
        while self.continue_training and self.state.epoch < self.max_epoch:
            self.run_epoch()

    def run_epoch(self):
        """ Run training for one epoch """
        if self.continue_training:
            self.network.train(True)
            self.state.epoch += 1
            self.state.iteration = 0
            logging.info(f"Started epoch: {self.state.epoch}")
            for batch_data in self.data_loader:
                self.lrs_history.append(self.optimiser.param_groups[0]['lr'])
                self.state.iteration += 1
                self.optimiser.zero_grad()
                inputs, targets = batch_data[self.inputs_key].to(self.device), \
                                  batch_data[self.targets_key].to(self.device)
                outputs = self.network(inputs)

                if self.send_instmask:
                    instance_mask = batch_data[self.instance_key].to(self.device)
                    loss = self.loss_function(outputs, targets, instance_mask)
                else:
                    loss = self.loss_function(outputs, targets)
                self.state.loss += loss.item()
                loss.backward()
                self.optimiser.step()

                if self.state.iteration % self.print_freq == 0:
                    logging.info(f"iteration {self.state.iteration} / {len(self.data_loader)}: "
                                 f"{(self.state.loss / self.state.iteration):.4f}")

                if (self.state.epoch - 1) * len(
                        self.data_loader) + self.state.iteration <= self.warmup_iters and self.warming_up:
                    self.warmup.step()
                    logging.info(
                        f"Iter {(self.state.epoch - 1) * len(self.data_loader) + self.state.iteration}, step to {self.optimiser.param_groups[0]['lr']}")
                else:
                    self.warming_up = False

            self.state.loss /= self.state.iteration
            self.loss_history.append(self.state.loss)

            for callback in self.callbacks:
                if isinstance(callback, GeneralLoggingCallback):
                    val_metric = callback(trainer=self)
                else:
                    callback(trainer=self)

            if self.scheduler is not None and self.lrs_history[-1] > self.min_lr and not self.warming_up:
                if self.plateau_scheduler:
                    self.scheduler.step(val_metric)
                else:
                    self.scheduler.step()
                logging.info(f"Changing lr: {self.lrs_history[-1]:.3e} --> {self.optimiser.param_groups[0]['lr']:.3e}")
            elif self.warming_up:
                logging.info(f"Warmup learning rate on epoch {self.state.epoch}: {self.lrs_history[-1]:.3e}")

            return self.state
        else:
            return self.state
