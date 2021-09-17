import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.pytorch.model import HNN
from src.pytorch.utils.plot import save_y_pred_scatter

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
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.max_epochs = max_epochs
        self.plot_n_epochs = plot_n_epochs
        self.dirname = dirname
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.y_pred_values = {} #{state: (y, pred)} of the last epoch

    def train_loop(self):
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

    def save_traced_model(self, filename: str):
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

        example_input = self.train_dataloader.dataset[0][0]
        traced_model = torch.jit.trace(self.model, example_input)
        traced_model.save(filename)

    def save_scatter_plot(self, t: int):
        with torch.no_grad():
            for X, y in self.train_dataloader:
                pred = self.model(X)
                x_lst = X.tolist()

                for i, _ in enumerate(x_lst):
                    x_int = [int(x) for x in x_lst[i]]
                    x_str = ''.join(str(e) for e in x_int)
                    self.y_pred_values[x_str] = (int(y[i][0]), int(pred[i][0]))

            if t != -1:
                save_y_pred_scatter(self.y_pred_values, t, f"{self.dirname}")
                self.y_pred_values.clear()
            else:
                _log.info(f"Saving post-training scatter plot to {self.dirname}/plots.")
                save_y_pred_scatter(self.y_pred_values, t, f"{self.dirname}")


    def run(self, train_timer, validation=True):
        last_val_loss = 0
        max_epochs_without_improving = 100
        count = 0
        for t in range(self.max_epochs):
            cur_train_loss = self.train_loop()

            if t % self.plot_n_epochs == 0 and self.plot_n_epochs != -1:
                self.save_scatter_plot(t)

            if validation:
                cur_val_loss = self.val_loop()
                if (last_val_loss - cur_val_loss) > 0.01:
                    count = 0
                else:
                    count += 1
                    if count >= max_epochs_without_improving:
                        _log.info(
                            f"The loss on the validation data didn't improve "
                            f"in {max_epochs_without_improving} epochs."
                        )
                        break

                last_val_loss = cur_val_loss
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
        self.save_scatter_plot(-1)

        return cur_val_loss if validation else None
