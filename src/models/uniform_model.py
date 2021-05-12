import numpy as np
import matplotlib.pyplot as plt

from src.reward_generators.simple_rewards import SimpleRewardsGenerator


def evaluate(model, n_steps=1000, delta=10, reward_gen=SimpleRewardsGenerator
             ()):
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
    for step in range(1, n_steps + 1):
        reward_vector, item_changed = reward_gen.get_rewards()
        selected_action = model.get_action()

        regret = (
            np.max(reward_gen.reward_probs) - reward_gen.reward_probs[selected_action]
        )
        regrets.append(regret)

        rewards.append(reward_vector[selected_action])
        last_rewards.append(reward_vector[selected_action])
        last_changes.append(item_changed)

        # Feedback if delta steps have passed
        if step % delta == 0:
            model.update(last_rewards, last_changes)
            last_rewards = []
            last_changes = []
    return regrets, rewards


class UniformModel:
    """Model that uniformly chooses a random action
    """

    def __init__(self, n_items=10):
        self.n_items = n_items

    def get_action(self):
        """Returns a random item

        Returns:
            TYPE: Description
        """
        return np.random.choice(self.n_items)

    def update(self, x, y):
        """Summary

        Args:
            x (TYPE): Description
            y (TYPE): Description
        """
        pass


if __name__ == "__main__":
    for _ in range(10):
        gen = SimpleRewardsGenerator(change_prob=0.5)
        print("Reward probabilities before: ", gen.reward_probs)
        rewards, change = gen.get_rewards()
        print("Rewards: ", rewards)
        print("Item changed: ", change)
        print("Reward probabilities after: ", gen.reward_probs, "\n")

    regrets, rewards = evaluate(UniformModel())
    plt.plot(range(len(regrets)), np.cumsum(regrets))
    plt.title("Regret")
    plt.show()

    plt.plot(range(len(rewards)), np.cumsum(rewards))
    plt.title("Reward")
    plt.show()
