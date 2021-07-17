from random import shuffle as random_shuffle

class KFoldTrainingData():
    def __init__(self, state_value_pairs, num_folds = 10, shuffle = False):
        self.state_value_pairs = state_value_pairs
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
            kfolds.append((training_set, test_set))
        
        return kfolds


    def get_fold(self, id):
        """
        Returns a fold as tuple(training set, test set).
        Counting from 0.
        """
        return self.kfolds[id]


    def get_num_fold(self):
        """
        Returns num_folds
        """
        return len(self.num_folds)
