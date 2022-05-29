import logging
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from torch.utils.data import DataLoader
from src.pytorch.model import HNN
from src.pytorch.utils.plot import save_y_pred_scatter
from src.pytorch.utils.helpers import prefix_to_h, get_memory_usage_mb
from src.pytorch.utils.timer import Timer

_log = logging.getLogger(__name__)


class TrainWorkflow:
    def __init__(
        self,
        model: HNN,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        device: torch.device,
        max_epochs: int,
        plot_n_epochs: int,
        save_best: bool,
        dirname: str,
        optimizer: optim.Optimizer,
        scatter_plot: bool,
        check_dead_once: bool,
        loss_fn: nn = nn.MSELoss(),
        is_weighted_loss_fn: bool = False,
        restart_no_conv: bool = True,
        patience: int = None,
    ):
        self.model = model
        self.best_epoch_model = None
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.validation = self.val_dataloader != None
        self.testing = self.test_dataloader != None
        self.device = device
        self.max_epochs = max_epochs
        self.plot_n_epochs = plot_n_epochs
        self.save_best = save_best
        self.dirname = dirname
        self.optimizer = optimizer
        self.scatter_plot = scatter_plot
        self.loss_fn = loss_fn
        self.is_weighted_loss_fn = is_weighted_loss_fn
        self.patience = patience
        self.early_stopped = False
        self.restart_no_conv = restart_no_conv
        self.check_dead_once = check_dead_once
        self.train_y_pred_values = []  # [state, y, pred]
        self.val_y_pred_values = [] # [state, y, pred]

    def train_loop(self, t: int, fold_idx: int) -> float:
        """
        Network's train loop.
        """
        # size = len(self.train_dataloader.dataset)
        num_batches = len(self.train_dataloader)
        train_loss = 0

        for _batch, item in enumerate(self.train_dataloader):
            # Compute prediction and loss.
            if len(item) == 3:
                X, y, w = item[0].to(self.device), item[1].to(self.device), item[2].to(self.device)
                pred = self.model(X.float())
                loss = self.loss_fn(pred, y, w) if self.is_weighted_loss_fn else self.loss_fn(pred, y)
            else:
                X, y = item[0].to(self.device), item[1].to(self.device)
                pred = self.model(X.float())
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

        if len(self.train_y_pred_values) > 0 and self.scatter_plot:
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
            for item in self.val_dataloader:
                if len(item) == 3:
                    X, y, w = item[0].to(self.device), item[1].to(self.device), item[2].to(self.device)
                    pred = self.model(X.float())
                    val_loss += self.loss_fn(pred, y, w).item() if self.is_weighted_loss_fn else self.loss_fn(pred, y).item()
                else:
                    X, y = item[0].to(self.device), item[1].to(self.device)
                    pred = self.model(X.float())
                    val_loss += self.loss_fn(pred, y).item()

                if t % self.plot_n_epochs == 0 and self.plot_n_epochs != -1:
                    self.val_y_pred_values = self.fill_y_pred(
                        X, y, pred, self.val_y_pred_values
                    )
        if len(self.val_y_pred_values) > 0 and self.scatter_plot:
            save_y_pred_scatter(
                self.val_y_pred_values, t, fold_idx, f"{self.dirname}/plots", "val_"
            )
            self.val_y_pred_values.clear()
        return val_loss / num_batches

    def test_loop(self) -> float:
        """
        Network's testing loop.
        """
        num_batches = len(self.test_dataloader)
        test_loss = 0
        with torch.no_grad():
            for item in self.test_dataloader:
                if len(item) == 3:
                    X, y, w = item[0].to(self.device), item[1].to(self.device), item[2].to(self.device)
                    pred = self.model(X.float())
                    test_loss += self.loss_fn(pred, y, w).item() if self.is_weighted_loss_fn else self.loss_fn(pred, y).item()
                else:
                    X, y = item[0].to(self.device), item[1].to(self.device)
                    pred = self.model(X.float())
                    test_loss += self.loss_fn(pred, y).item()

        return test_loss / num_batches

    def val_loop_no_contrasting(self, contrasting_h: int = 501) -> float:
        """
        Evaluation loop without contrasting.
        """
        num_batches = len(self.val_dataloader)
        val_loss = 0
        with torch.no_grad():
            for item in self.val_dataloader:
                X, y = item[0].to(self.device), item[1].to(self.device)
                pred = self.model(X.float())
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
            for item in self.train_dataloader:
                X = item[0] if self.device == "cpu" else item[0].to(self.device)
                for p in self.model(X.float()):
                    p_list = p.tolist()
                    if type(p_list) is float:
                        if p_list != 0.0:
                            return False
                    else:
                        if len(p) > 1:  # prefix
                            p = prefix_to_h(p_list)
                        if float(p) != 0.0:
                            return False
        return True

    def save_traced_model(self, filename: str, model="hnn"):
        """
        Saves a traced model to be used by the C++ backend.
        """
        if model == "resnet" or model == "resnet_rtdl":
            example_input = self.train_dataloader.dataset[:10][0].float()
        elif model == "simple" or model == "hnn":
            example_input = self.train_dataloader.dataset[0][0].float()

        # To make testing possible (and fair), the model has to be saved while in the CPU,
        # even if training was performed in GPU.
        traced_model = torch.jit.trace(self.best_epoch_model.to("cpu"), example_input)
        traced_model.save(filename)

    def run(self, fold_idx: int, train_timer: Timer) -> float:
        """
        Network train/eval main loop.
        """
        best_loss, best_epoch = None, None
        cur_train_loss, cur_val_loss = None, None
        born_dead = False
        check_once = False
        t = 0
        while t < self.max_epochs and not self.early_stopped and not train_timer.check_timeout():
            cur_train_loss = self.train_loop(t, fold_idx)
            # Check if born dead (or died during training)
            if not (t % 10) and not born_dead and not check_once:
                if self.dead():
                    if self.restart_no_conv:
                        _log.warning(
                            "All predictions are 0 (born dead). Restarting training with a new seed..."
                        )
                        return None
                    else:
                        _log.warning(
                            "All predictions are 0 (born dead), but restart is disabled."
                        )
                        born_dead = True
                if self.check_dead_once:
                    check_once = True

            epoch_log = f"Epoch {t} | avg_train_loss={cur_train_loss:>7f}"

            if self.validation:
                cur_val_loss = self.val_loop(t, fold_idx)
                epoch_log += f" | avg_val_loss={cur_val_loss:>7f}"

            cur_loss = cur_val_loss if self.validation else cur_train_loss
            if not best_loss or best_loss > cur_loss:
                best_loss, best_epoch = cur_loss, t
                self.best_epoch_model = deepcopy(self.model)
            if best_epoch < t - self.patience:
                self.early_stopped = True

            if self.testing:
                cur_test_loss = self.test_loop()
                epoch_log += f" | avg_test_loss={cur_test_loss:>7f}"

            _log.info(epoch_log)

            if t % 10 == 0:
                _log.debug(f"Current mem usage: {get_memory_usage_mb()} MB")

            t += 1

        if self.early_stopped:
            _log.info(f"Early stop. Best epoch: {best_epoch}/{t}")
        if train_timer.check_timeout():
            _log.info(f"Training time reached. Best epoch: {best_epoch}/{t}")
        if t == self.max_epochs:
            _log.info(f"Max epoch reached. Best epoch: {best_epoch}/{t}")
        _log.info(f"Mem usage END: {get_memory_usage_mb()} MB")

        if not self.save_best:
            self.best_epoch_model = self.model

        # Post-training scatter plot.
        if self.scatter_plot:
            self.save_post_scatter_plot(fold_idx)

        return best_loss

    def save_post_scatter_plot(self, fold_idx: int):
        """
        After all epochs are done, perform an extra evaluation to create the final scatter plot.
        """
        with torch.no_grad():
            if self.validation:
                for item in self.val_dataloader:
                    X, y = item[0].to(self.device), item[1].to(self.device)
                    self.val_y_pred_values = self.fill_y_pred(
                        X, y, self.model(X.float()), self.val_y_pred_values
                    )
            else:
                self.val_y_pred_values = None
            for item in self.train_dataloader:
                X, y = item[0].to(self.device), item[1].to(self.device)
                self.train_y_pred_values = self.fill_y_pred(
                    X, y, self.model(X.float()), self.train_y_pred_values
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
                #y_pred_values[x_str] = (y_h, pred_h)
                y_pred_values.append([x_str, y_h, pred_h])
            else:  # Regression
                y_pred_values.append([
                    x_str,
                    round(y[i][0].item()),
                    round(pred[i][0].item()),
                ])
        return y_pred_values
