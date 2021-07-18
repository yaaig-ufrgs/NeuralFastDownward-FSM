from random import shuffle as random_shuffle

from torch.utils.data import DataLoader

from training_data import InstanceDataset, generate_optimal_state_value_pairs


class KFoldTrainingData():
    def __init__(self, domain, problems, domain_max_value, batch_size, num_folds = 10, shuffle = False):
        self.state_value_pairs = generate_optimal_state_value_pairs(domain, problems)
        self.domain_max_value = domain_max_value
        self.batch_size = batch_size
        self.num_folds = num_folds
        if shuffle:
            random_shuffle(self.state_value_pairs)
        assert len(self.state_value_pairs) % self.num_folds == 0
        self.kfolds = self.generate_kfold_training_data()


    def generate_kfold_training_data(self):
        """
        Generates the folds.
        Return a tuple(training set, test set) of size num_folds
        """

        instances_per_fold = len(self.state_value_pairs) / self.num_folds
        kfolds = []
        for i in range(self.num_folds):
            training_set = []
            test_set = []
            for j, state_value_pair in enumerate(self.state_value_pairs):
                if int(j / instances_per_fold) == i:
                    test_set.append(state_value_pair)
                else:
                    training_set.append(state_value_pair)

            train_dataloader = DataLoader(dataset=InstanceDataset(training_set, self.domain_max_value),
                                          batch_size=self.batch_size, num_workers=1)
            test_dataloader = DataLoader(dataset=InstanceDataset(test_set, self.domain_max_value),
                                          batch_size=self.batch_size, num_workers=1)
            kfolds.append((train_dataloader, test_dataloader))
        
        return kfolds


    def get_fold(self, idx):
        """
        Returns a fold as tuple(train dataloader, test dataloader).
        Counting from 0.
        """
        return self.kfolds[idx]


    def get_num_fold(self):
        """
        Returns num_folds
        """
        return len(self.num_folds)
