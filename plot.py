"""Codes for plots """

# Author: Wenlu Xu wenluxu@ucla.edu

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def set_global(mvalue: int, kvalue: int, nvalue: int):
    """
    Initialize public parameters
    :param mvalue: int, 4, 8, 16; Number of arms
    :param kvalue: int, 5 low_D, 50 high_D ; Sampling size
    :param nvalue: int, 3; Number of groups

    :return: None

    Note
    ----
    path0, path1 can be modified dependent on outcome directory
    """
    global m, k, path0, path1, n
    n = nvalue
    m = mvalue
    k = kvalue

    # get file path
    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    path_str = 'm' + str(m) + 'k' + str(k) + '/'
    path0 = father_path + '/Outcomes/high_D/'
    path1 = father_path + '/Outcomes/high_D/' + path_str
    # use this in low_D
    '''
    path0 = father_path + '/Outcomes/low_D/'
    path1 = father_path + '/Outcomes/low_D/' + path_str
    '''

def renew_gp(gp: int):
    """
    Get the group label of gp-th cluster label
    :param gp: int; Cluster label
    :return: int; Group label
    """
    stat = np.array(pd.read_csv(path0 + 'cluster_stat.csv', index_col=0))
    return np.argsort(stat[gp])[-1]

def sub_plot_scatter():
    """
    Plot incentive indicator in the sampling stage
    :return: scatter plots; m: high_D, lm: low_D
    """
    plt.figure(figsize=(8, 8))

    # plot two subfigures (k=5/10; k=50/100) together
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    ax_list = [ax1, ax2]

    for sub in range(2):
        # sub = subfigure number
        set_global(mvalue=m, kvalue=(sub+1)*k, nvalue=3)
        if (sub == 1):
            ax2 = plt.subplot(212, sharex=ax1)
            ax_list = [ax1, ax2]
        ax = ax_list[sub]
        gp_plot = np.zeros([n, m])
        marker = ['o', '^', ',']
        color = ['r', 'g', 'b']
        # plot n groups using different colors
        for i in range(n):
            # get the incentive indicator
            gp_plot[i, :] = cal_incentive_arm(i)
            # 0: on line; 1: above line; -1: below line
            for j in range(m):
                gp_plot[i, j] += j
            # legend indicating group label
            lgd = 'Group ' + str(i + 1)
            ax.scatter(range(m), gp_plot[i, :], alpha=0.5, c=[],
                        edgecolors=color[i], marker=marker[i], label=lgd)

        ax.plot(range(m), range(m), label='D', linestyle=':',
                c='black', alpha=0.5)
        ax.legend()
        ax.grid()
        ax.set_title('m=' + str(m) + ', ' + 'k=' + str(k))
        ax.set_ylabel('Incentive Region')

    plt.xlabel('Sample Phase')
    plt.savefig('./Figures/m'+ str(m) +'.jpg', dpi=500)
    # use this in low_D
    '''
    plt.savefig('./Figures/lm'+ str(m) +'.jpg', dpi=500)
    '''
    plt.show()
    return

def cal_incentive_arm(gp: int):
    """
    Calculate incentive indicator
    :param gp: cluster label
    :return: size_m list; incentive indicator

    Notes
    -----
    incentive indicator in phase i:
    0 if D is BIC;
    -1 if i+1 is BIC;
    1 if l <= i, l ≠ D;
    """
    # sample_psi = historical data in the sampling stage
    sample_psi = np.array(pd.read_csv(path1 + str(gp) + 'cluster_sample_psi.csv', index_col=0))

    # sample_theta = theta
    sample_theta = np.array(pd.read_csv(path1 + str(gp) + 'cluster_sample_theta_store.csv', index_col=0))
    sample_theta = sample_theta[1:sample_theta.shape[0], ].astype(float)

    # p_i = prior / m
    p_i = np.array(pd.read_csv(path1 + 'cluster_p_i.csv', index_col=0))
    tru_gp = renew_gp(gp)
    gp_p_i = p_i[tru_gp]

    # arm_rank = prior rank in cluster gp
    arm_rank = list(np.argsort(-gp_p_i))

    # theta_0 = prior
    theta_0 = np.zeros([m])

    # q_t = q_t in the sampling stage
    q_t = np.zeros([m])

    # set theta_0, q_t
    for i in range(m):
        theta_0[i] = int(gp_p_i[i] * m)
        tmp_psi = sample_psi[np.where(sample_psi[:, 5] == i)]
        q_t[i] = tmp_psi[0, 4]

    # ans[i] = 0: incentive=D; 1: incentive=i+1; -1: incentive from [i]\D
    ans = np.zeros([m])

    # theta_hat = estimated theta, whose values are changed throughout sampling
    theta_hat = theta_0
    for i in range(1, m):
        arm = arm_rank[i]
        # up_value = expected reward of i+1
        if (i != m - 1):
            up_arm = arm_rank[i+1]
            up_value = theta_0[up_arm]
        else:
            up_value = 0

        # update theta_hat
        for j in range(m):
            ind = (i - 1) * m + j
            if (sample_theta[ind] != 0):
                theta_hat[j] = sample_theta[ind]
        # subtract baseline
        theta_hat -= (theta_hat[0] - theta_0[0])

        # D_value = expected reward of D
        D_value = q_t[i] * theta_0[arm] + (1-q_t[i]) * sum(theta_hat) / i

        # down_value = expected reward of [i]\D
        down_value = (1-q_t[i]) * theta_0[arm] + q_t[i] * sum(theta_hat) / i

        # compare three possibilities
        max_value = max(D_value, down_value, up_value)
        if (max_value == down_value): ans[i] = -1
        elif (max_value == up_value): ans[i] = 1
    return ans

def cal_regret(gp: int, file_name: str):
    """
    Calculate
    :param gp: int; cluster label
    :param file_name: str, 'cluster_race_psi.csv', 'RF_psi.csv', 'UCB_psi.csv';
    algorithm output
    :return: ndarray; historical data in racing stage
    """
    # race_psi = historical data in BIC racing stage
    race_psi = np.array(pd.read_csv(path1 + str(gp) + file_name, index_col=0))

    # theta = theta
    theta = np.array(pd.read_csv(path1 + 'cluster_theta.csv', index_col=0))

    # transfer cluster label to group label
    tru_gp = renew_gp(gp)
    gp_theta = theta[tru_gp]

    # get max possible reward
    max_reward = max(gp_theta)

    len_race = race_psi.shape[0]
    ans = np.zeros([len_race])

    # calculate regrets in each phase
    ans[0] = max_reward - gp_theta[int(race_psi[0, 1])]

    # accumulate regrets
    for i in range(1, len_race):
        change = max_reward - gp_theta[int(race_psi[i, 1])]
        ans[i] = ans[i-1] + change
    return ans

def int_regret(file_name: str):
    """
    Extend regrets to the same metric
    :param file_name: str, 'cluster_race_psi.csv', 'RF_psi.csv', 'UCB_psi.csv';
    algorithm output
    :return: ndarray; extended regrets for n groups

    Notes
    -----
    Due to different racing rounds of algorithms, the calculated regrets do not share the same X-axis.
    This function is to extend regrets to the same rounds
    and prolong to another 200 rounds to see the trend in racing exploition.
    """
    # leng = racing rounds of different groups
    leng = [None] * n
    for i in range(n):
        reg = cal_regret(i, file_name)
        leng[i] = reg.shape[0]

    # T = extended length
    T = max(leng) + 200

    # ans = extended regrets
    ans = np.zeros([n, T])
    for i in range(n):
        # get calculated regrets
        reg = cal_regret(i, file_name)

        # for the first leng[i] regrets, it is calculated
        for j in range(leng[i]):
            ans[i, j] = reg[j]

        # for the remaining regrets, keep the trend of the last racing choice
        for j in range(leng[i], T):
            ans[i, j] = ans[i, j - 1] + ans[i, leng[i] - 1] - ans[i, leng[i] - 2]
    return ans

def plot_regret():
    """
    Plot regret curve
    :return: line plots showing regret curve of different algorithms
    """
    plt.figure(figsize=(8, 8))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    ax_list = [ax1, ax2]

    # plot k=5/10 or 50/100 together
    for sub in range(2):
        set_global(mvalue=4, kvalue=(sub+1) * k, nvalue=3)
        ax = ax_list[sub]
        color = ['r', 'g', 'b']

        # get regret for different algorithms
        reg = int_regret('cluster_race_psi.csv')
        for i in range(n):
            gp_plot = reg[i, :]
            T = gp_plot.shape[0]
            lgd = 'Group ' + str(i + 1) + ' BIC'
            ax.plot(range(T), gp_plot, alpha=0.5, c=color[i], label=lgd, linestyle='-')

        # add this to plot RF regrets
        '''
        reg = int_regret('RF_psi.csv')
        for i in range(n):
            gp_plot = reg[i, :]
            T = gp_plot.shape[0]
            lgd = 'Group ' + str(i + 1) + ' RF'
            ax.plot(range(T), gp_plot, alpha=0.5, c=color[i], label=lgd, linestyle=':')
        '''

        # add this in low_D case to plot UCB regrets
        '''
        reg = int_regret('UCB_psi.csv')
        for i in range(n):
            gp_plot = reg[i, :]
            T = gp_plot.shape[0]
            lgd = 'Group ' + str(i + 1) + ' LinUCB'
            ax.plot(range(T), gp_plot, alpha=0.5, c=color[i], label=lgd, linestyle='--')
        '''

        ax.legend()
        ax.grid()
        ax.set_title('m=' + str(m) + ', ' + 'k=' + str(k))
        ax.set_ylabel('Regret')

    plt.xlabel('Race Round')
    plt.savefig('./Figures/m16c.jpg', dpi=500)
    plt.show()
    return

if __name__ == '__main__':
    # Initialize (high_D)
    set_global(mvalue=16, kvalue=50, nvalue=3)
    # Plot comparisons/ regret
    plot_regret()
    # Plot incentive indicator
    sub_plot_scatter()
    exit(0)
