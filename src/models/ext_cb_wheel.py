from contextualbandits.linreg import LinearRegression
from contextualbandits.online import ActiveExplorer
from contextualbandits.online import AdaptiveGreedy
from contextualbandits.online import EpsilonGreedy
from contextualbandits.online import LinUCB
from contextualbandits.online import LinTS
from sklearn.linear_model import SGDClassifier

import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pandas as pd
import re

from functools import partial
from tqdm import tqdm

from contextualbandits.linreg import LinearRegression
from copy import deepcopy
from pylab import rcParams
from reward_generators.wheel_bandit import WheelBandit
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# rounds are simulated from the full dataset


def simulate_rounds_stoch(model, rewards, actions_hist, X_batch, y_batch,
                          rnd_seed):
    np.random.seed(rnd_seed)

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

    return actions_this_batch, new_actions_hist, rewards_batch


def get_batch(batch_size, nchoices, reward_gen):
    rewards = np.zeros((batch_size, nchoices))
    reward_probs = np.zeros((batch_size, nchoices))
    contexts = []
    for i in range(batch_size):
        context = reward_gen.generate_context()
        reward_vector, reward_means = reward_gen.get_rewards(context)
        rewards[i] = reward_vector
        reward_probs[i] = reward_means
        contexts.append(context)
    return rewards, reward_probs, contexts


def get_mean_reward(reward_lst, batch_size):
    mean_rew = list()
    for r in range(len(reward_lst)):
        mean_rew.append(sum(reward_lst[:r+1]) * 1.0 / ((r+1)*batch_size))
    return mean_rew


def worker(worker_num, model, batch_size, nchoices, n_steps,
           exploration_param=0.5):
    # batch size - algorithms will be refit after N rounds

    # These lists will keep track of the rewards obtained by each policy
    # rewards_ucb, rewards_ts, rewards_ovr, rewards_egr, rewards_lucb, \
    #     rewards_agr, rewards_agr2, rewards_efr, rewards_ac, \
    #     rewards_aac, rewards_sft = [list() for i in range(len(models))]
    lst_rewards = []

    # initial seed - all policies start with the same small random selection of actions/rewards
    reward_gen = WheelBandit(delta=exploration_param)

    init_reward_vector = np.zeros((batch_size, nchoices))
    for i in range(batch_size):
        reward_vector, _ = reward_gen.get_rewards(reward_gen.generate_context())
        init_reward_vector[i] = reward_vector

    np.random.seed(1)
    first_batch = np.zeros((batch_size, 2))  # dim of contexts
    action_chosen = np.random.randint(nchoices, size=batch_size)
    rewards_received = init_reward_vector[np.arange(batch_size), action_chosen]

    # fitting models for the first time
    model.fit(X=first_batch, a=action_chosen, r=rewards_received)

    # these lists will keep track of which actions does each policy choose
    lst_actions = [action_chosen.copy()]

    ret_regrets = []
    ret_rewards = []

    # now running all the simulation
    for i in range(int(np.floor(n_steps / batch_size))):
        batch_st = (i + 1) * batch_size
        rewards, reward_probs, contexts = get_batch(batch_size,
                                                    nchoices,
                                                    reward_gen)
        X_batch = np.array(contexts)
        y_batch = rewards

        actions_this_batch, lst_actions, rewards_batch = simulate_rounds_stoch(model,
                                                                               lst_rewards,
                                                                               lst_actions,
                                                                               X_batch, y_batch,
                                                                               rnd_seed=batch_st)

        ret_regrets.extend(
            np.max(reward_probs, axis=1) -
            reward_probs[np.arange(batch_size), actions_this_batch]
        )
        ret_rewards.extend(rewards_batch)

    return ret_regrets, ret_rewards


def beta_prior_calc(beta_priors, nchoices=5, is_ts=False):
    '''
    nchoices is the number of arms
    returns list of tuples, each of which can be tried as hps
    '''
    if is_ts:
        nchoices = np.log2(nchoices)
    return [((p/nchoices, 4), 2) for p in beta_priors]


if __name__ == "__main__":
    NUM_TRIALS = 10
    DELTAS = [1, 10, 100, 1000]
    NUM_ITEMS = 5
    n_steps = 10000

    nchoices = NUM_ITEMS

    base_sgd = SGDClassifier(random_state=123, loss='log', warm_start=False)
    base_ols = LinearRegression(lambda_=10., fit_intercept=True, method="sm")

    beta_priors_ag = [7.0]
    beta_priors_eg = [3]
    beta_priors_ucb = [1]
    beta_priors_ts = [1]
    beta_priors_ae = [1]

    # print(f"beta_priors: {beta_prior_calc(beta_priors_ag)}")
    # print(‘UCB_priors: ’, beta_prior_calc(beta_prior_ucb))
    # print(‘TS_prios: ’, beta_prior_calc(beta_prior_ts, is_ts=True))

    # The base algorithm is embedded in different metaheuristics
    adaptive_active_greedy = AdaptiveGreedy(deepcopy(base_ols), nchoices=nchoices,
                                            smoothing=None,
                                            beta_prior=beta_prior_calc(beta_priors_ag)[0],
                                            active_choice='weighted', decay_type='percentile',
                                            decay=0.9997, batch_train=True,
                                            random_state=2222, njobs=4)

    epsilon_greedy_nodecay = EpsilonGreedy(deepcopy(base_ols), nchoices=nchoices,
                                           smoothing=None, beta_prior=beta_prior_calc(beta_priors_eg)[0],
                                           decay=None, batch_train=True,
                                           deep_copy_buffer=False, random_state=6666, njobs=4)

    linucb = LinUCB(nchoices=nchoices, beta_prior=beta_prior_calc(beta_priors_ucb)[0], alpha=0.1,
                    ucb_from_empty=False, random_state=1111, njobs=4)

    lints = LinTS(nchoices=nchoices, beta_prior=beta_prior_calc(beta_priors_ts)[0],
                  random_state=1111, njobs=4)

    ae = ActiveExplorer(deepcopy(base_ols), nchoices=nchoices,
                        beta_prior=beta_prior_calc(beta_priors_ae)[0],
                        random_state=1111, batch_train=True, njobs=4)

    models = [adaptive_active_greedy, epsilon_greedy_nodecay, linucb, lints, ae]

    mean_regret_results = np.zeros((len(models), len(DELTAS), n_steps))
    mean_reward_results = np.zeros((len(models), len(DELTAS), n_steps))
    for i, model in tqdm(enumerate(models)):
        for j, batch_size in enumerate(DELTAS):
            with multiprocessing.Pool() as pool:
                results = pool.map(partial(worker,
                                           batch_size=batch_size,
                                           nchoices=nchoices, model=model,
                                           n_steps=n_steps),
                                   range(NUM_TRIALS))
                mean_regret_results[i][j] = np.array([x[0] for x in
                                                      results]).sum(axis=0) / NUM_TRIALS

                mean_reward_results[i][j] = np.array([x[1] for x in
                                                      results]).sum(axis=0) / NUM_TRIALS

    fig, axs = plt.subplots(2, 2)

    for i, ax in enumerate(axs.flat):
        ax.plot(get_mean_reward(mean_reward_results[0][i],
                                batch_size=1),
                label="Adaptive Active Greedy")

        ax.plot(get_mean_reward(mean_reward_results[1][i], batch_size=1),
                label="Epsilon-Greedy No Decay")

        ax.plot(get_mean_reward(mean_reward_results[2][i], batch_size=1),
                label="Linear UCB")

        ax.plot(get_mean_reward(mean_reward_results[3][i], batch_size=1),
                label="Linear TS")

        ax.plot(get_mean_reward(mean_reward_results[4][i], batch_size=1),
                label="Active Explorer")

        ax.set(xlabel='Rounds', ylabel='Cumulative Mean Reward', title=f"$\delta$ = {DELTAS[i]}")

        ax.set_ylim([0, 20])

        if i == 0:
            ax.legend(loc="lower right", prop={'size': 7})

    fig.suptitle('Comparison of Online Contextual Bandit Policies with Delayed with Wheel Simulator')
    plt.show()
