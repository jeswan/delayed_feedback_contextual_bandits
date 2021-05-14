from contextualbandits.online import ActiveExplorer
from contextualbandits.online import AdaptiveGreedy
from contextualbandits.online import BootstrappedTS
from contextualbandits.online import BootstrappedUCB
from contextualbandits.online import EpsilonGreedy
from contextualbandits.online import ExploreFirst
from contextualbandits.online import LogisticUCB
from contextualbandits.online import SeparateClassifiers
from contextualbandits.online import SoftmaxExplorer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import multiprocessing
from functools import partial

from contextualbandits.linreg import LinearRegression
from copy import deepcopy
from pylab import rcParams
from reward_generators.ext_simple_rewards import SimpleRewardsGenerator
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# rounds are simulated from the full dataset


def simulate_rounds_stoch(model, rewards, actions_hist, X_batch, y_batch,
                          items_changed, rnd_seed):
    np.random.seed(rnd_seed)

    for item_changed in items_changed:
        model.drop_arm(item_changed)
        model.add_arm(item_changed)
        model.choice_names = np.arange(10)

    # choosing actions for this batch
    actions_this_batch = model.predict(X_batch).astype('uint8')

    # keeping track of the sum of rewards received
    rewards.append(y_batch[np.arange(y_batch.shape[0]), actions_this_batch].sum())

    # adding this batch to the history of selected actions
    new_actions_hist = np.append(actions_hist, actions_this_batch)

    # rewards obtained now
    rewards_batch = y_batch[np.arange(y_batch.shape[0]), actions_this_batch]

    # now refitting the algorithms after observing these new rewards
    np.random.seed(rnd_seed)
    model.partial_fit(X_batch, actions_this_batch, rewards_batch)

    return actions_this_batch, new_actions_hist


def get_batch(batch_size, nchoices, reward_gen):
    rewards = np.zeros((batch_size, nchoices))
    reward_probs = np.zeros((batch_size, nchoices))
    items_changed = []
    for i in range(batch_size):
        reward_vector, item_changed = reward_gen.get_rewards()
        rewards[i] = reward_vector
        reward_probs[i] = reward_gen.reward_probs
        if item_changed:
            items_changed.append(item_changed)
    return rewards, reward_probs, items_changed


def get_mean_reward(reward_lst, batch_size):
    mean_rew = list()
    for r in range(len(reward_lst)):
        mean_rew.append(sum(reward_lst[:r+1]) * 1.0 / ((r+1)*batch_size))
    breakpoint()
    return mean_rew


def worker(worker_num, model, batch_size, nchoices, n_steps):
    # batch size - algorithms will be refit after N rounds

    # These lists will keep track of the rewards obtained by each policy
    # rewards_ucb, rewards_ts, rewards_ovr, rewards_egr, rewards_lucb, \
    #     rewards_agr, rewards_agr2, rewards_efr, rewards_ac, \
    #     rewards_aac, rewards_sft = [list() for i in range(len(models))]
    lst_rewards = []

    # initial seed - all policies start with the same small random selection of actions/rewards
    reward_gen = SimpleRewardsGenerator(change_prob=1e-3)

    init_reward_vector = np.zeros((batch_size, nchoices))
    for i in range(batch_size):
        reward_vector, _ = reward_gen.get_rewards()
        init_reward_vector[i] = reward_vector

    np.random.seed(1)
    first_batch = np.zeros((batch_size, 1))
    action_chosen = np.random.randint(nchoices, size=batch_size)
    rewards_received = init_reward_vector[np.arange(batch_size), action_chosen]

    # fitting models for the first time
    model.fit(X=first_batch, a=action_chosen, r=rewards_received)

    # these lists will keep track of which actions does each policy choose
    lst_actions = [action_chosen.copy()]

    lst_regrets = [0]

    # now running all the simulation
    for i in range(int(np.floor(n_steps / batch_size))):
        batch_st = (i + 1) * batch_size
        X_batch = np.zeros((batch_size, 1))
        rewards, reward_probs, items_changed = get_batch(batch_size, nchoices,
                                                         reward_gen)
        y_batch = rewards

        actions_this_batch, lst_actions = simulate_rounds_stoch(model,
                                                                lst_rewards,
                                                                lst_actions,
                                                                X_batch, y_batch,
                                                                items_changed,
                                                                rnd_seed=batch_st)

        lst_regrets += (
            np.max(reward_probs, axis=1) -
            reward_probs[np.arange(batch_size), actions_this_batch]
        ).sum()

    return lst_regrets


if __name__ == "__main__":
    NUM_TRIALS = 16
    # DELTAS = [1000, 316, 100, 32, 10, 3, 1]
    DELTAS = [1, 10, 100, 1000]
    NUM_ITEMS = 10
    n_steps = 10000

    nchoices = NUM_ITEMS

    base_sgd = SGDClassifier(random_state=123, loss='log', warm_start=False)
    base_ols = LinearRegression(lambda_=10., fit_intercept=True, method="sm")

    # The base algorithm is embedded in different metaheuristics
    adaptive_active_greedy = AdaptiveGreedy(deepcopy(base_ols), nchoices=nchoices,
                                            smoothing=None, beta_prior=((3./nchoices, 4.), 2),
                                            active_choice='weighted', decay_type='percentile',
                                            decay=0.9997, batch_train=True,
                                            random_state=2222)
    epsilon_greedy_nodecay = EpsilonGreedy(deepcopy(base_ols), nchoices=nchoices,
                                           smoothing=(1, 2), beta_prior=None,
                                           decay=None, batch_train=True,
                                           deep_copy_buffer=False, random_state=6666)

    models = [adaptive_active_greedy, epsilon_greedy_nodecay]

    results = np.zeros((len(models), len(DELTAS)))
    for i, model in tqdm(enumerate(models)):
        for j, batch_size in enumerate(DELTAS):
            with multiprocessing.Pool() as pool:
                results[i, j] = np.mean(pool.map(partial(worker,
                                                         batch_size=batch_size,
                                                         nchoices=nchoices, model=model,
                                                         n_steps=n_steps),
                                                 range
                                                 (NUM_TRIALS)))
    print(results)
