import numpy as np


class SimpleRewardsGenerator:

    """Generates rewards according to the procedure in Table 1 of "An Empirical Evaluation of
    Thompson Sampling" by Chapelle and Li.

    Attributes:
        change_prob (float): The probability one of the items is retired and replaced by a new one
        n_items (int): number of items
        reward_dist (np.random): The true reward probability of a given item
        reward_probs (ndarray or scalar): The true reward probability of each item
    """

    def __init__(
        self,
        n_items=10,
        change_prob=1e-3,
        reward_dist=lambda x: np.random.default_rng().beta(4.0,
                                                           4.0, size=x),
    ):
        self.n_items = n_items
        self.change_prob = change_prob
        self.reward_dist = reward_dist

        self.reward_probs = self.reward_dist(self.n_items)

    def get_rewards(self):
        """Get reward array for the simulator

        Returns:
            ndarray: Reward array with reward for each item
            item_changed: item that was replaced
        """
        item_changed = None
        if np.random.rand() < self.change_prob:
            item_to_change = np.random.choice(self.n_items)
            self.reward_probs = np.delete(self.reward_probs, item_to_change)
            self.reward_probs = np.append(self.reward_probs, [self.reward_dist
                                                              (1)[0]])
            item_changed = item_to_change

        rewards = (np.random.rand(self.n_items) < self.reward_probs) * 1

        return rewards, item_changed
