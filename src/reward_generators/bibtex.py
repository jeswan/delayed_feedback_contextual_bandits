import numpy as np
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.datasets import load_svmlight_file

class BibtexRewards():
    """
    Transforms Bibtex multilabel classification dataset into Contextual Bandit
    problem.

    Attributes:
        path_to_bibtex (str): Path to the Bibtex data
        select_random (bool): Whether to select random data points on each step
    """
    def __init__(self, path_to_bibtex='data/Bibtex_data.txt', select_random=False):
        with open(path_to_bibtex, "rb") as f:
            infoline = f.readline()
            infoline = re.sub(r"^b'", "", str(infoline))
            n_features = int(re.sub(r"^\d+\s(\d+)\s\d+.*$", r"\1", infoline))
            features, labels = load_svmlight_file(f, n_features=n_features, multilabel=True)
        mlb = MultiLabelBinarizer()
        labels = mlb.fit_transform(labels)
        features = np.array(features.todense())
        features = np.ascontiguousarray(features)

        self.contexts = features
        self.rewards = labels
        self.n_contexts = self.contexts.shape[0]

        self.select_random = select_random
        self.cur_idx = -1

    def generate_context(self):
        """
        Selects a datapoint from the dataset and returns its feature vector
        """
        if self.select_random:
            self.cur_idx = np.random.choice(self.n_contexts)
        else:
            self.cur_idx = (self.cur_idx + 1) % self.n_contexts

        return self.contexts[self.cur_idx]

    def get_rewards(self, context):
        """
        Get rewards and reward means for a given context. Raises an Exception
        if a context is not found in the dataset

        Returns:
            rewards: Reward vector
            reward_means: Reward means
        """
        if (context == self.contexts[self.cur_idx]).all():
            return self.rewards[self.cur_idx], self.rewards[self.cur_idx]

        for i in range(self.n_contexts):
            if (context == self.contexts[i]).all():
                return self.rewards[i], self.rewards[i]

        raise Exception('Context not found in dataset')
