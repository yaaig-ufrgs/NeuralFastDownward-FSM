from random import shuffle as random_shuffle

from torch.utils.data import DataLoader

from training_data import InstanceDataset, generate_optimal_state_value_pairs


class KFoldTrainingData():
    def __init__(self, domain, problems, domain_max_value, batch_size, num_folds = 10, shuffle = False):
        if shuffle:
            random_shuffle(problems)
        self.problems = problems
        self.state_value_pairs = generate_optimal_state_value_pairs(domain, problems)
        self.domain_max_value = domain_max_value
        self.batch_size = batch_size
        self.num_folds = num_folds
        assert len(self.state_value_pairs) % self.num_folds == 0
        self.kfolds, self.kfolds_problems = self.generate_kfold_training_data()


    def generate_kfold_training_data(self):
        """
        Generate the folds.
        Return two list of tuples of size num_folds: dataloaders and problems.
        The first item corresponds to train set, and the second to test set. 
        """

        instances_per_fold = len(self.state_value_pairs) / self.num_folds
        kfolds, kfolds_problems = [], []
        for i in range(self.num_folds):
            training_set, test_set = [], []
            training_problems, test_problems = [], []

            for j in range(len(self.state_value_pairs)):
                if int(j / instances_per_fold) == i:
                    test_set.append(self.state_value_pairs[j])
                    test_problems.append(self.problems[j])
                else:
                    training_set.append(self.state_value_pairs[j])
                    training_problems.append(self.problems[j])

            train_dataloader = DataLoader(dataset=InstanceDataset(training_set, self.domain_max_value),
                                          batch_size=self.batch_size, num_workers=1)
            test_dataloader = DataLoader(dataset=InstanceDataset(test_set, self.domain_max_value),
                                          batch_size=self.batch_size, num_workers=1)
            kfolds.append((train_dataloader, test_dataloader))
            kfolds_problems.append((training_problems, test_problems))
        
        return kfolds, kfolds_problems


    def get_fold(self, idx):
        """
        Returns a fold as tuple(train dataloader, test dataloader).
        Counting from 0.
        """
        return self.kfolds[idx]

    def get_test_problems_from_fold(self, idx):
        """
        Returns a list with the test problems from a fold.
        Counting from 0.
        """
        return self.kfolds_problems[idx][1]

    def get_num_fold(self):
        """
        Returns num_folds
        """
        return len(self.num_folds)
