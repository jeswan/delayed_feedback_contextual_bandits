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

from contextualbandits.linreg import LinearRegression
from copy import deepcopy
from pylab import rcParams
from reward_generators.simple_rewards import SimpleRewardsGenerator
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

    return new_actions_hist


def get_batch(batch_size, nchoices, reward_gen):
    rewards = np.zeros((batch_size, nchoices))
    items_changed = []
    for i in range(batch_size):
        reward_vector, item_changed = reward_gen.get_rewards()
        rewards[i] = reward_vector
        if item_changed:
            items_changed.append(item_changed)
    return rewards, items_changed


if __name__ == "__main__":
    NUM_TRIALS = 100
    # DELTAS = [1000, 316, 100, 32, 10, 3, 1]
    DELTAS = [1000]
    NUM_ITEMS = 10
    n_steps = 1000000

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

    # models = [bootstrapped_ucb, bootstrapped_ts, one_vs_rest, epsilon_greedy, logistic_ucb,
    #           adaptive_greedy_thr, adaptive_greedy_perc, explore_first, active_explorer,
    #           adaptive_active_greedy, softmax_explorer]
    models = [adaptive_active_greedy, epsilon_greedy_nodecay]

    # These lists will keep track of the rewards obtained by each policy
    # rewards_ucb, rewards_ts, rewards_ovr, rewards_egr, rewards_lucb, \
    #     rewards_agr, rewards_agr2, rewards_efr, rewards_ac, \
    #     rewards_aac, rewards_sft = [list() for i in range(len(models))]
    rewards_ac, rewards_egr = [list() for i in range(len(models))]

    lst_rewards = [rewards_ac, rewards_egr]

    # batch size - algorithms will be refit after N rounds
    batch_size = 50

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
    for model in models:
        model.fit(X=first_batch, a=action_chosen, r=rewards_received)

    # these lists will keep track of which actions does each policy choose
    lst_a_ac, lst_a_egr = [action_chosen.copy() for i in range(len(models))]

    lst_actions = [lst_a_ac, lst_a_egr]

    # now running all the simulation
    for i in tqdm(range(int(np.floor(n_steps / batch_size)))):
        batch_st = (i + 1) * batch_size
        X_batch = np.zeros((batch_size, 1))
        rewards, items_changed = get_batch(batch_size, nchoices, reward_gen)
        y_batch = rewards

        for model in range(len(models)):
            lst_actions[model] = simulate_rounds_stoch(models[model],
                                                       lst_rewards[model],
                                                       lst_actions[model],
                                                       X_batch, y_batch,
                                                       items_changed,
                                                       rnd_seed=batch_st)

    def get_mean_reward(reward_lst, batch_size=batch_size):
        mean_rew = list()
        for r in range(len(reward_lst)):
            mean_rew.append(sum(reward_lst[:r+1]) * 1.0 / ((r+1)*batch_size))
        return mean_rew

    rcParams['figure.figsize'] = 25, 15
    lwd = 5
    cmap = plt.get_cmap('tab20')
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    rcParams['figure.figsize'] = 25, 15

    ax = plt.subplot(111)
    plt.plot(get_mean_reward(rewards_ac), label="Active Explorer (SGD)",
             linewidth=lwd, color=colors[15])
    plt.plot(get_mean_reward(rewards_egr),
             label="Epsilon-Greedy (p0=20%, decay=0.9999, OLS)", linewidth=lwd, color=colors[6])

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 1.25])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, ncol=3, prop={'size': 20})

    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.xticks([i*20 for i in range(8)], [i*1000 for i in range(8)])

    plt.xlabel('Rounds (models were updated every 50 rounds)', size=30)
    plt.ylabel('Cumulative Mean Reward', size=30)
    plt.title('Comparison of Online Contextual Bandit Policies\n(Streaming-data mode)\n\nBibtext Dataset\n(159 categories, 1836 attributes)', size=30)
    plt.grid()
    plt.show()

    # for delta in tqdm(DELTAS):
    #     trial_sum_regrets = []
    #     trial_sum_rewards = []
    #     with multiprocessing.Pool() as pool:
    #         results = pool.map(partial(worker, delta=delta), range
    #                            (NUM_TRIALS))

    #         for result in results:
    #             trial_sum_regrets = np.array([x[0] for x in results])
    #             trial_sum_rewards = np.array([x[1] for x in results])

    #         print(f"DELTA: {delta}")
    #         print(f"average total regrets: {trial_sum_regrets.mean()}")
    #         print(f"std total regrets: {trial_sum_regrets.std()}")

    #         print(f"average total rewards: {trial_sum_rewards.mean()}")
    #         print(f"std total rewards: {trial_sum_regrets.std()}")

    # plt.plot(range(len(regrets)), np.cumsum(regrets))
    # plt.title("Regret")
    # plt.show()
