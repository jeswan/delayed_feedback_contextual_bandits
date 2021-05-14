import numpy as np

class WheelBandit():
    """
    Generates rewards according to the Wheel Bandit procedure in
    "Deep Bayesian Bandits Showdown" by Riquelme, Tucker and Snoek.

    Attributes:
        delta (float): Exploration parameter
        mu_1 (float): Mean reward for action 0
        mu_2 (float): Mean reward for non-optimal actions
        mu_3 (float): Mean reward for optimal actions
        sigma (float): Reward standard deviation
    """
    def __init__(self, delta, mu_1=1.2, mu_2=1., mu_3=50., sigma=0.01):
        self.delta = delta
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.mu_3 = mu_3
        self.sigma = sigma

    def generate_context(self):
        """
        Generate a sample from uniform distribution in the unit circle

        Returns:
            [x, y]: Coordinates of the sampled point
        """
        length = np.sqrt(np.random.uniform(0, 1))
        angle = np.pi * np.random.uniform(0, 2)

        x = length * np.cos(angle)
        y = length * np.sin(angle)

        return [x, y]

    def get_rewards(self, context):
        """
        Get rewards and reward means for a given context

        Returns:
            rewards: Reward vector
            reward_means: Reward means
        """
        rewards = np.zeros(5)
        reward_means = np.zeros(5)
        rewards[0] = np.random.normal(self.mu_1, self.sigma)
        reward_means[0] = self.mu_1

        for i in range(1, 5):
                rewards[i] = np.random.normal(self.mu_2, self.sigma)
                reward_means[i] = self.mu_2

        if np.linalg.norm(context) >= self.delta:
            if context[0] > 0 and context[1] > 0:
                rewards[1] = np.random.normal(self.mu_3, self.sigma)
                reward_means[1] = self.mu_3
            if context[0] > 0 and context[1] < 0:
                rewards[2] = np.random.normal(self.mu_3, self.sigma)
                reward_means[2] = self.mu_3
            if context[0] < 0 and context[1] > 0:
                rewards[3] = np.random.normal(self.mu_3, self.sigma)
                reward_means[3] = self.mu_3
            if context[0] < 0 and context[1] < 0:
                rewards[4] = np.random.normal(self.mu_3, self.sigma)
                reward_means[4] = self.mu_3
        return rewards, reward_means


def evaluate_with_context(model, n_steps=1000, delta=10, reward_gen=WheelBandit(0.5)):
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
    last_contexts = []
    for step in range(1, n_steps + 1):
        context = reward_gen.generate_context()
        reward_vector, reward_means = reward_gen.get_rewards(context)
        selected_action = model.get_action(context)

        regret = (
            np.max(reward_means) - reward_means[selected_action]
        )
        regrets.append(regret)

        rewards.append(reward_vector[selected_action])
        last_rewards.append(reward_vector[selected_action])
        last_contexts.append(context)

        # Feedback if delta steps have passed
        if step % delta == 0:
            model.update(last_contexts, last_rewards)
            last_rewards = []
            last_contexts = []
    return regrets, rewards
