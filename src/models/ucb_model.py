import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.reward_generators.simple_rewards import SimpleRewardsGenerator


class UCBModel:
    """Model that uniformly chooses a random action
    """

    def __init__(self, init_rewards, n_items=10):
        self.n_items = n_items
        self.total_rewards = np.zeros(n_items)
        self.total_selections = np.zeros(n_items)
        self.is_item_changed = False
        self.item_changed = None

    def get_action(self, step_num, item_changed):
        """Returns item according to UCB algorithm

        Returns:
            int: Description
        """

        # k is total reward of arm
        # m is number of times arm has been selected

        if item_changed:
            self.total_rewards[item_changed] = 0.0
            self.total_selections[item_changed] = 0

        if np.all(self.total_selections):
            delta = np.sqrt(1.0/step_num)
            k = self.total_rewards
            m = self.total_selections
            ucbs = (k/m) + np.sqrt(2.0 * k/m * np.log(1/delta) / m) + (2.0 *
                                                                       np.log
                                                                       (1.0/delta) / m)
            action = np.argmax(ucbs)

        else:
            action = np.where(self.total_selections == 0)[0][0]
        # breakpoint()

        return action

    def update(self, actions, rewards, items_changed):
        # breakpoint()
        totals = np.bincount(actions, weights=rewards)
        unique, counts = np.unique(actions, return_counts=True)
        for i, count in zip(unique, counts):
            self.total_selections[i] += count
            self.total_rewards[i] += totals[i]


def evaluate(model, reward_gen, n_steps=1000000, delta=1000):
    """Evaulate the regrets and rewards of a given model based on a given reward
    generator

    Args:
        model (TYPE): Description
        n_steps (int, optional): Description
        delta (int, optional): Number of steps for feedback delay
        reward_gen (TYPE, optional): Description

    Returns:
        regrets (list): List of regrets for each round. Regret is the maximum
                        reward minus the selected action's reward for the round
        rewards (list): List of rewards for actions taken
    """
    regrets = []
    rewards = []
    last_rewards = []
    last_changes = []
    last_selected_actions = []
    for step in tqdm(range(1, n_steps + 1)):
        reward_vector, item_changed = reward_gen.get_rewards()
        selected_action = model.get_action(step, item_changed)

        regret = (
            np.max(reward_gen.reward_probs) - reward_gen.reward_probs[selected_action]
        )

        regrets.append(regret)
        rewards.append(reward_vector[selected_action])
        last_rewards.append(reward_vector[selected_action])
        last_changes.append(item_changed)
        last_selected_actions.append(selected_action)

        # Feedback if delta steps have passed
        if step % delta == 0:
            model.update(last_selected_actions, last_rewards, last_changes)
            breakpoint()
            last_rewards = []
            last_changes = []
            last_selected_actions = []
    return regrets, rewards


if __name__ == "__main__":
    gen = SimpleRewardsGenerator(change_prob=1e-3)
    print("Reward probabilities before: ", gen.reward_probs)
    rewards, change = gen.get_rewards()
    print("Rewards: ", rewards)
    print("Item changed: ", change)
    print("Reward probabilities after: ", gen.reward_probs, "\n")

    regrets, rewards = evaluate(UCBModel(init_rewards=rewards), reward_gen=gen)
    # plt.plot(range(len(regrets)), np.cumsum(regrets))
    # plt.title("Regret")
    # plt.show()

    # plt.plot(range(len(rewards)), np.cumsum(rewards))
    # plt.title("Reward")
    # plt.show()
    print(f"total regrets: {np.sum(regrets)}")
