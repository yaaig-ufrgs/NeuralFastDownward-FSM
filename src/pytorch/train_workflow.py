import logging
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from torch.utils.data import DataLoader
from src.pytorch.model import HNN
from src.pytorch.utils.plot import save_y_pred_scatter
from src.pytorch.utils.helpers import prefix_to_h
from src.pytorch.utils.timer import Timer

_log = logging.getLogger(__name__)


class TrainWorkflow:
    def __init__(
        self,
        model: HNN,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        device: torch.device,
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
        self.validation = False if self.val_dataloader == None else True
        self.device = device
        self.max_epochs = max_epochs
        self.plot_n_epochs = plot_n_epochs
        self.dirname = dirname
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.patience = patience
        self.early_stopped = False
        self.restart_no_conv = restart_no_conv
        self.train_y_pred_values = {}  # {state = (y, pred)}
        self.val_y_pred_values = {}
        self.cur_train_loss = 0.0

    def train_loop(self, t: int, fold_idx: int) -> float:
        """
        Network's train loop.
        """
        # size = len(self.train_dataloader.dataset)
        num_batches = len(self.train_dataloader)
        train_loss = 0

        for _batch, (X, y) in enumerate(self.train_dataloader):
            # Compute prediction and loss.
            X, y = X.to(self.device), y.to(self.device)
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
                self.train_y_pred_values = self.fill_y_pred(
                    X, y, pred, self.train_y_pred_values
                )

        if len(self.train_y_pred_values) > 0:
            save_y_pred_scatter(
                self.train_y_pred_values, t, fold_idx, f"{self.dirname}/plots", "train_"
            )
            self.train_y_pred_values.clear()

        return train_loss / num_batches

    def val_loop(self, t: int, fold_idx: int) -> float:
        """
        Network's evaluation loop.
        """
        num_batches = len(self.val_dataloader)
        val_loss = 0
        with torch.no_grad():
            for X, y in self.val_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                val_loss += self.loss_fn(pred, y).item()
                if t % self.plot_n_epochs == 0 and self.plot_n_epochs != -1:
                    self.val_y_pred_values = self.fill_y_pred(
                        X, y, pred, self.val_y_pred_values
                    )
        if len(self.val_y_pred_values) > 0:
            save_y_pred_scatter(
                self.val_y_pred_values, t, fold_idx, f"{self.dirname}/plots", "val_"
            )
            self.val_y_pred_values.clear()
        return val_loss / num_batches

    def val_loop_no_contrasting(self, contrasting_h: int = 501) -> float:
        """
        Evaluation loop without contrasting.
        """
        num_batches = len(self.val_dataloader)
        val_loss = 0
        with torch.no_grad():
            for X, y in self.val_dataloader:
                X = X.to(self.device)
                pred = self.model(X)
                val_loss += self.loss_fn(
                    torch.tensor(
                        [pred_ for i, pred_ in enumerate(pred) if y[i] != contrasting_h]
                    ),
                    torch.tensor([y_ for y_ in y if y_ != contrasting_h]),
                ).item()
        return val_loss / num_batches

    def dead(self) -> bool:
        """
        Checks if the network is dead.
        """
        with torch.no_grad():
            for X, _ in self.train_dataloader:
                X = X.to(self.device)
                for p in self.model(X):
                    if len(p) > 1:  # prefix
                        p = prefix_to_h(p.tolist())
                    if float(p) != 0.0:
                        return False
        return True

    def save_traced_model(self, filename: str, model="hnn"):
        """
        Saves a traced model to be used by the C++ backend.
        """
        if model == "resnet":
            example_input = self.train_dataloader.dataset[:10][0]
        elif model == "simple":
            example_input = self.train_dataloader.dataset[0][0]

        model_save = self.model if not self.early_stopped else self.best_epoch_model
        # To make testing possible (and fair), the model has to be saved while in the CPU,
        # even if training was performed in GPU.
        traced_model = torch.jit.trace(model_save.to("cpu"), example_input)
        traced_model.save(filename)

    def run(
        self, fold_idx: int, train_timer: Timer) -> (float, bool):
        """
        Network train/eval main loop.
        """
        best_val_loss = None
        born_dead = False
        for t in range(self.max_epochs):
            self.cur_train_loss = self.train_loop(t, fold_idx)
            # Check if born dead
            if t == 0:
                if self.dead():
                    if self.restart_no_conv:
                        _log.warning(
                            "All predictions are 0 (born dead). Restarting training with a new seed..."
                        )
                        return None, True
                    else:
                        _log.warning(
                            "All predictions are 0 (born dead), but restart is disabled."
                        )
                        born_dead = True
            # Check if the network died during training
            elif not born_dead and self.dead():
                _log.warning("All predictions are 0.")

            #print(self.cur_train_loss)

            if self.validation:
                cur_val_loss = self.val_loop(t, fold_idx)
                _log.info(
                    f"Epoch {t} | avg_train_loss={self.cur_train_loss:>7f}"
                    f" | avg_val_loss={cur_val_loss:>7f}"
                )
                if self.patience != None:
                    if best_val_loss is None or best_val_loss > cur_val_loss:
                        best_val_loss, best_val_epoch = cur_val_loss, t
                        self.best_epoch_model = deepcopy(self.model)

                    if best_val_epoch < t - self.patience:
                        _log.info(f"Early stop. Best epoch: {best_val_epoch}/{t}")
                        self.early_stopped = True
                        break
            else:
                _log.info(f"Epoch {t} | avg_train_loss={self.cur_train_loss:>7f}")

            # Check once every 10 epochs
            if (t % 10 == 0) and train_timer.check_timeout():
                break
            if t == self.max_epochs - 1:
                _log.info("Done!")

        if self.validation == False:
            # If not using validation data, save the last epoch's model.
            self.best_epoch_model = deepcopy(self.model)

        # Post-training scatter plot.
        self.save_post_scatter_plot(fold_idx)

        return (best_val_loss if self.validation else None), False

    def save_post_scatter_plot(self, fold_idx: int):
        """
        After all epochs are done, perform an extra evaluation to create the final scatter plot.
        """
        with torch.no_grad():
            if self.validation:
                for X, y in self.val_dataloader:
                    X, y = X.to(self.device), y.to(self.device)
                    self.val_y_pred_values = self.fill_y_pred(
                        X, y, self.model(X), self.val_y_pred_values
                    )
            else:
                self.val_y_pred_values = None
            for X, y in self.train_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                self.train_y_pred_values = self.fill_y_pred(
                    X, y, self.model(X), self.train_y_pred_values
                )

            _log.info(f"Saving post-training scatter plot.")
            save_y_pred_scatter(
                self.train_y_pred_values,
                -1,
                fold_idx,
                f"{self.dirname}/plots",
                "train_",
            )
            save_y_pred_scatter(
                self.val_y_pred_values, -1, fold_idx, f"{self.dirname}/plots", "val_"
            )

    def fill_y_pred(
        self, X: torch.Tensor, y: torch.Tensor, pred: torch.Tensor, y_pred_values: dict
    ) -> dict:
        """
        Fill `y_pred_values` dict with sampled and predicted h values for each state.
        state = (sampled heuristic, predicted heuristic)
        """
        x_lst = X.tolist()
        for i, _ in enumerate(x_lst):
            x_int = [int(x) for x in x_lst[i]]
            x_str = "".join(str(e) for e in x_int)
            if len(y[i]) > 1:  # Prefix (unary encoding)
                y_h = prefix_to_h(y[i].tolist())
                pred_h = prefix_to_h(pred[i].tolist())
                y_pred_values[x_str] = (y_h, pred_h)
            else:  # Regression
                y_pred_values[x_str] = (torch.round(y[i][0]), torch.round(pred[i][0]))
        return y_pred_values
