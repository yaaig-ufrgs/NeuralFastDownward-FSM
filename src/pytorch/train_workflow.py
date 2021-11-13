import logging
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from torch.utils.data import DataLoader

from src.pytorch.model import HNN
from src.pytorch.utils.plot import save_y_pred_scatter
from src.pytorch.utils.helpers import prefix_to_h

_log = logging.getLogger(__name__)


class TrainWorkflow:
    def __init__(
        self,
        model: HNN,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        max_epochs: int,
        plot_n_epochs: int,
        dirname: str,
        optimizer: optim.Optimizer,
        loss_fn: nn = nn.MSELoss(),
        restart_no_conv: bool = True,
        patience: int = None,
    ):
        self.model = model
        self.best_epoch_model = None
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.max_epochs = max_epochs
        self.plot_n_epochs = plot_n_epochs
        self.dirname = dirname
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.patience = patience
        self.early_stopped = False
        self.restart_no_conv = restart_no_conv
        self.y_pred_values = {}  # {state: (y, pred)} of the last epoch

    def train_loop(self, t: int, fold_idx: int):
        # size = len(self.train_dataloader.dataset)
        num_batches = len(self.train_dataloader)
        train_loss = 0

        for batch, (X, y) in enumerate(self.train_dataloader):
            # Compute prediction and loss.
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            train_loss += loss.item()

            # Clear gradients for the variables it will update.
            self.optimizer.zero_grad()

            # Compute gradient of the loss.
            loss.backward()

            # Update parameters.
            self.optimizer.step()

            if t % self.plot_n_epochs == 0 and self.plot_n_epochs != -1:
                x_lst = X.tolist()
                for i, _ in enumerate(x_lst):
                    x_int = [int(x) for x in x_lst[i]]
                    x_str = "".join(str(e) for e in x_int)
                    if len(y[i]) > 1:  # Prefix (unary encoding)
                        y_h = prefix_to_h(y[i].tolist())
                        pred_h = prefix_to_h(pred[i].tolist())
                        self.y_pred_values[x_str] = (y_h, pred_h)
                    else:  # Regression
                        self.y_pred_values[x_str] = (int(y[i][0]), int(pred[i][0]))

        if len(self.y_pred_values) > 0:
            save_y_pred_scatter(
                self.y_pred_values, t, fold_idx, f"{self.dirname}/plots"
            )
            self.y_pred_values.clear()

        return train_loss / num_batches

    def val_loop(self):
        # size = len(self.val_dataloader.dataset)
        num_batches = len(self.val_dataloader)
        val_loss = 0
        with torch.no_grad():
            for X, y in self.val_dataloader:
                pred = self.model(X)
                val_loss += self.loss_fn(pred, y).item()

        return val_loss / num_batches

    def dead(self):
        with torch.no_grad():
            for X, _ in self.train_dataloader:
                for p in self.model(X):
                    if float(p) != 0.0:
                        return False
        return True

    def save_traced_model(self, filename: str, model="hnn"):
        """
        Saves a traced model to be used by the C++ backend.
        """

        # What is torch.jit.trace?
        # ------------------------
        # docs: https://pytorch.org/docs/stable/generated/torch.jit.trace.html
        #
        # "Trace a function and return an executable or ScriptFunction that will be optimized using
        # just-in-time compilation. Tracing is ideal for code that operates only on Tensors and lists,
        # dictionaries, and tuples of Tensors."
        #
        # "Using torch.jit.trace and torch.jit.trace_module, you can turn an existing module or Python
        # function into a TorchScript ScriptFunction or ScriptModule. You must provide example inputs,
        # and we run the function, recording the operations performed on all the tensors."
        #
        # In other words, "tracing" a model means transforming your PyTorch code ("eager mode") to
        # TorchScript code ("script mode"). Script mode is focused on production, while eager mode is
        # for prototyping and research. Script mode is performatic (JIT) and portable.
        #
        # TorchScript is a domain-specific language for ML, and it is a subset of Python.

        if model == "resnet":
            example_input = self.train_dataloader.dataset[:10][0]
        elif model == "simple":
            example_input = self.train_dataloader.dataset[0][0]

        model_save = self.model if not self.early_stopped else self.best_epoch_model
        traced_model = torch.jit.trace(model_save, example_input)
        traced_model.save(filename)

    def run(self, fold_idx, train_timer, validation=True):
        loss_first_epoch = 0
        best_val_loss = None
        born_dead = False
        for t in range(self.max_epochs):
            cur_train_loss = self.train_loop(t, fold_idx)
            # Check if born dead
            if t == 0:
                if self.dead():
                    if self.restart_no_conv:
                        _log.warning("All predictions are 0 (born dead). Restarting training with a new seed...")
                        return None, True
                    else:
                        _log.warning("All predictions are 0 (born dead), but restart is disabled.")
                        born_dead = True
            # Check if the network died during training
            elif not born_dead and self.dead():
                _log.warning("All predictions are 0.")

            if validation:
                cur_val_loss = self.val_loop()
                loss_first_epoch = cur_val_loss if t == 0 else loss_first_epoch
                if self.patience != None:
                    if best_val_loss is None or best_val_loss > cur_val_loss:
                        best_val_loss, best_val_epoch = cur_val_loss, t
                        self.best_epoch_model = deepcopy(self.model)

                    if best_val_epoch < t - self.patience:
                        _log.info(
                            f"Early stop. Best epoch: {best_val_epoch}/{t}"
                        )
                        self.early_stopped = True
                        break
                _log.info(
                    f"Epoch {t} | avg_train_loss={cur_train_loss:>7f} "
                    f"| avg_val_loss={cur_val_loss:>7f}"
                )
            else:
                _log.info(f"Epoch {t} | avg_train_loss={cur_train_loss:>7f}")

            # Check once every 10 epochs
            if (t % 10 == 0) and train_timer.check_timeout():
                break
            if t == self.max_epochs - 1:
                _log.info("Done!")

        # Post-training scatter plot.
        self.save_post_scatter_plot(fold_idx)

        return (cur_val_loss if validation else None), False

    def save_post_scatter_plot(self, fold_idx: int):
        with torch.no_grad():
            self.y_pred_values.clear()
            for X, y in self.train_dataloader:
                pred = self.model(X)
                x_lst = X.tolist()
                for i, _ in enumerate(x_lst):
                    x_int = [int(x) for x in x_lst[i]]
                    x_str = "".join(str(e) for e in x_int)
                    if len(y[i]) > 1:  # Prefix (unary encoding)
                        y_h = prefix_to_h(y[i].tolist())
                        pred_h = prefix_to_h(pred[i].tolist())
                        self.y_pred_values[x_str] = (y_h, pred_h)
                    else:  # Regression
                        self.y_pred_values[x_str] = (int(y[i][0]), int(pred[i][0]))

            _log.info(f"Saving post-training scatter plot.")
            save_y_pred_scatter(
                self.y_pred_values, -1, fold_idx, f"{self.dirname}/plots"
            )
