import matplotlib.pyplot as plt
import multiprocessing
import numpy as np

from collections import Counter
from collections import defaultdict
from functools import partial
from tqdm import tqdm

from reward_generators.simple_rewards import SimpleRewardsGenerator

from numpy.random import beta


class Thompson_Model:
    """Model that uniformly chooses a random action
    """

    def __init__(self, init_rewards, n_items=10):
        self.n_items = n_items
        self.total_rewards = init_rewards
        self.total_selections = np.zeros(n_items)
        self.is_item_changed = False
        self.item_changed = None
        self.a_prior = np.ones(n_items)
        self.b_prior = np.ones(n_items)
        

    def get_action(self, step_num, item_changed, successes, failures):
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
            #draw thetat a vector of size n_items, according to beta distribution
            #using a_prior + successes and b_prior + failures as parameters.
            #every time this function is called, successes and failures will
            # have been udpdated from the previous round, so we do not need to change 
            #the priors themselves. 
            thetat = beta(self.a_prior + successes, self.b_prior+failures)
                
            action = np.argmax(thetat)
            

        else:
            action = np.where(self.total_selections == 0)[0][0]

        return action

    def update(self, actions, rewards, items_changed):
        totals = np.bincount(actions, weights=rewards)
        unique, counts = np.unique(actions, return_counts=True)
        for i, count in zip(unique, counts):
            self.total_selections[i] += count
            self.total_rewards[i] += totals[i]


def evaluate(model, reward_gen, n_steps=1000000, delta=1):
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
    #
    successes = np.zeros(model.n_items)
    failures = np.zeros(model.n_items)
    for step in range(1, n_steps + 1):
        reward_vector, item_changed = reward_gen.get_rewards()
        
        if item_changed:
            successes = np.zeros(model.n_items)
            failures = np.zeros(model.n_items)
        
        
        selected_action = model.get_action(step, item_changed, successes, failures)
        

        regret = (
            np.max(reward_gen.reward_probs) - reward_gen.reward_probs[selected_action]
        )
        
            
        

        regrets.append(regret)
        rewards.append(reward_vector[selected_action])
        last_rewards.append(reward_vector[selected_action])
        last_changes.append(item_changed)
        last_selected_actions.append(selected_action)
        
        
        
        if reward_vector[selected_action] == 1:
            successes[selected_action] += 1
        else:
            failures[selected_action] += 1
            
       

        # Feedback if delta steps have passed
        if step % delta == 0:
            model.update(last_selected_actions, last_rewards, last_changes)
            last_rewards = []
            last_changes = []
            last_selected_actions = []
    return regrets, rewards


def worker(num, delta):
    gen = SimpleRewardsGenerator(change_prob=1e-3)
    init_rewards, change = gen.get_rewards()
    model = Thompson_Model(init_rewards=init_rewards)

    regrets, rewards = evaluate(model, reward_gen=gen, delta=delta)
    return np.sum(regrets), np.sum(rewards)


if __name__ == "__main__":
    NUM_TRIALS = 100
    # DELTAS = [1, 3, 10, 32, 100, 316, 1000]
    DELTAS = [1000, 316, 100, 32, 10, 3,  1]

    # print("Reward probabilities before: ", gen.reward_probs)

    # print("Rewards: ", rewards)
    # print("Item changed: ", change)
    # print("Reward probabilities after: ", gen.reward_probs, "\n")

    for delta in tqdm(DELTAS):
        trial_sum_regrets = []
        trial_sum_rewards = []
        with multiprocessing.Pool() as pool:
            results = pool.map(partial(worker, delta=delta), range(NUM_TRIALS))

            for result in results:
                trial_sum_regrets = np.array([x[0] for x in results])
                trial_sum_rewards = np.array([x[1] for x in results])

            print(f"DELTA: {delta}")
            print(f"average total regrets: {trial_sum_regrets.mean()}")
            print(f"std total regrets: {trial_sum_regrets.std()}")

            print(f"average total rewards: {trial_sum_rewards.mean()}")
            print(f"std total rewards: {trial_sum_regrets.std()}")

    # plt.plot(range(len(regrets)), np.cumsum(regrets))
    # plt.title("Regret")
    # plt.show()

    # plt.plot(range(len(rewards)), np.cumsum(rewards))
    # plt.title("Reward")
    # plt.show()
