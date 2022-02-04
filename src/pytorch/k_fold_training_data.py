import logging
from random import shuffle as randshuffle
import torch
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as skshuffle
from torch.utils.data import DataLoader
import src.pytorch.utils.default_args as default_args

from src.pytorch.training_data import (
    InstanceDataset,
    load_training_state_value_pairs,
)

_log = logging.getLogger(__name__)


class KFoldTrainingData:
    def __init__(
        self,
        samples_file: str,
        batch_size: int = default_args.BATCH_SIZE,
        num_folds: int = default_args.NUM_FOLDS,
        output_layer: str = default_args.OUTPUT_LAYER,
        shuffle: bool = default_args.SHUFFLE,
        seed: int = default_args.RANDOM_SEED,
        shuffle_seed: int = default_args.SHUFFLE_SEED,
        training_size: int = default_args.TRAINING_SIZE,
        data_num_workers: int = default_args.DATALOADER_NUM_WORKERS,
        normalize: bool = default_args.NORMALIZE_OUTPUT,
        clamping: int = default_args.CLAMPING,
        remove_goals: bool = default_args.REMOVE_GOALS,
        standard_first: bool = default_args.STANDARD_FIRST,
        contrast_first: bool = default_args.STANDARD_FIRST,
        intercalate_samples: int = default_args.INTERCALATE_SAMPLES,
        cut_non_intercalated_samples: bool = default_args.CUT_NON_INTERCALATED_SAMPLES,
        sample_percentage: float = default_args.SAMPLE_PERCENTAGE,
        model: str = default_args.MODEL,
    ):
        assert training_size > 0.0 and training_size <= 1.0
        assert sample_percentage > 0.0 and sample_percentage <= 1.0

        self.state_value_pairs, self.domain_max_value = load_training_state_value_pairs(
            samples_file,
            clamping,
            remove_goals,
        )

        self.normalize = normalize
        if self.normalize:
            for i in range(len(self.state_value_pairs)):
                self.state_value_pairs[i][1] /= self.domain_max_value
        self.batch_size = batch_size
        self.num_folds = num_folds
        self.output_layer = output_layer
        self.shuffle = shuffle
        self.seed = seed
        self.shuffle_seed = shuffle_seed
        self.training_size = training_size
        self.data_num_workers = data_num_workers
        self.standard_first = standard_first
        self.contrast_first = contrast_first
        self.intercalate_samples = intercalate_samples
        self.cut_non_intercalated_samples = cut_non_intercalated_samples
        self.sample_percentage = sample_percentage
        self.model = model
        self.kfolds = self.generate_kfold_training_data()

    def generate_kfold_training_data(self) -> list:
        """
        Generates the folds.
        Returns two list of tuples of size num_folds: dataloaders and problems.
        The first item corresponds to train set, the second to val set, and the
        third to test set.
        """
        _log.info(f"Generating {self.num_folds}-fold...")

        kfolds = []
        instances_per_fold = int(len(self.state_value_pairs) / self.num_folds)
        for i in range(self.num_folds):
            training_set, val_set, test_set = [], [], []
            if self.num_folds == 1:
                # Test set = complement of training+val set
                if self.sample_percentage < 1.0:
                    self.state_value_pairs, test_set = train_test_split(
                        self.state_value_pairs,
                        train_size=self.sample_percentage,
                        shuffle=self.shuffle,
                        random_state=self.shuffle_seed,
                    )

                if self.training_size == 1.0:
                    training_set = (
                        skshuffle(
                            self.state_value_pairs, random_state=self.shuffle_seed
                        )
                        if self.shuffle
                        else self.state_value_pairs
                    )
                else:
                    training_set, val_set = train_test_split(
                        self.state_value_pairs,
                        train_size=self.training_size,
                        shuffle=self.shuffle,
                        random_state=self.shuffle_seed,
                    )
            else:
                # TODO: `sample_percentage` and `test_set` not implemented here!
                for j in range(len(self.state_value_pairs)):
                    if int(j / instances_per_fold) == i:
                        test_set.append(self.state_value_pairs[j])
                    else:
                        training_set.append(self.state_value_pairs[j])

            # If necessary, change the ordering of the data.
            if (
                self.standard_first
                or self.contrast_first
                or self.intercalate_samples > 0
            ):
                self.shuffle = False
                training_set = self.change_sampling_order(training_set)
                test_set = self.change_sampling_order(test_set)

            worker_fn = (
                None
                if self.seed == -1
                else lambda id: np.random.seed(self.shuffle_seed % 2 ** 32)
            )

            g = None if self.seed == -1 else torch.Generator()
            if g != None:
                g.manual_seed(self.shuffle_seed)

            train_dataloader = DataLoader(
                dataset=InstanceDataset(
                    training_set, self.domain_max_value, self.output_layer
                ),
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.data_num_workers,
                worker_init_fn=worker_fn,
                generator=g,
            )

            val_dataloader = (
                DataLoader(
                    dataset=InstanceDataset(
                        val_set, self.domain_max_value, self.output_layer
                    ),
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    num_workers=self.data_num_workers,
                    worker_init_fn=worker_fn,
                    generator=g,
                )
                if len(val_set) != 0
                else None
            )

            test_dataloader = (
                DataLoader(
                    dataset=InstanceDataset(
                        test_set, self.domain_max_value, self.output_layer
                    ),
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    num_workers=self.data_num_workers,
                    worker_init_fn=worker_fn,
                    generator=g,
                )
                if len(test_set) != 0
                else None
            )

            kfolds.append((train_dataloader, val_dataloader, test_dataloader))

        return kfolds

    def get_fold(self, idx: int) -> tuple:
        """
        Returns a fold as tuple(train dataloader, test dataloader).
        Counting from 0.
        """
        return self.kfolds[idx]

    def change_sampling_order(self, samples: list) -> list:
        """
        Returns state-value pairs with a different order for samples:
        - `contrast_first`: contrasting samples appear first.
        - `standard_first`: non-contrasting samples appear first.
        - `intercalate_samples`: contrasting and non-contrasting samples appear intercalated.
        """
        if len(samples) == 0:
            return samples

        standard_samples = []
        contrast_samples = []
        interc_n = self.intercalate_samples

        for sv in samples:
            if sv[1] == self.domain_max_value:
                contrast_samples.append(sv)
            else:
                standard_samples.append(sv)

        if self.standard_first or self.contrast_first:
            return (
                standard_samples + contrast_samples
                if self.standard_first
                else contrast_samples + standard_samples
            )
        else:
            min_len = min(len(standard_samples), len(contrast_samples))
            new_state_value_pairs = []
            for i in range(0, min_len, interc_n):
                new_state_value_pairs += (
                    standard_samples[i : i + interc_n]
                    + contrast_samples[i : i + interc_n]
                )
            if not self.cut_non_intercalated_samples:
                if min_len == len(standard_samples):
                    new_state_value_pairs += contrast_samples[i + interc_n :]
                else:
                    new_state_value_pairs += standard_samples[i + interc_n :]

            return new_state_value_pairs


def seed_worker(worker_id: int):
    """
    Sets the seed of each worker.
    See: https://pytorch.org/docs/stable/notes/randomness.html
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
