"""Linear UCB in contextual bandit """

# Author: *

import math
import random
import pandas as pd
import numpy as np
import os

class User:
    def __init__(self, feature: np.array, tru_gp: int):
        """
        Initialize agent attributes
        :param feature: ndarray; Context
        :param tru_gp: int; Group label
        """
        self.choice = -1
        self.reward = -1
        self.feature = feature
        self.tru_gp = tru_gp
        self.cal_gp = -1
        self.id = -1

    def set_group(self, cal_gp: int):
        """
        Set cluster label
        :param cal_gp: int; Cluster label
        :return: None
        """
        self.cal_gp = cal_gp

    def set_id(self, id: int):
        """
        Set agent index
        :param id: int; Agent index
        :return: None
        """
        self.id = id

    def buy(self, D: int):
        """
        Set agent's chosen item
        :param D: int; Chosen arm
        :return: None
        """
        self.choice = D

    def set_reward(self, Y):
        """
        Set agent's reward
        :param Y: int; Reward
        :return: None
        """
        self.reward = Y

def set_global(pvalue: int, mvalue: int, Tvalue: int, nvalue: int, kvalue: int):
    """
    Initialize public parameters
    :param pvalue: int, 5, 20000; Data dimensions
    :param mvalue: int, 4, 8, 16; Number of arms
    :param Tvalue: int, 6000, 30000; Number of agents
    :param nvalue: int, 3; Number of groups
    :param kvalue: int, 5 low_D, 50 high_D ; Sampling size
    :return: None
    """
    global p, m, T, X, theta, n, path0, path1, k, user_list, cluster_list
    p = pvalue
    m = mvalue
    T = Tvalue
    n = nvalue
    k = kvalue

    # user_list = T agents
    user_list = [None] * T

    # cluster_list = agents with cluster label n
    cluster_list = [None] * n

    # get file path
    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    path_str = 'm' + str(m) + 'k' + str(k) + '/'

    # LinUCB unavailable in high_D
    path0 = father_path + '/Outcomes/low_D/'
    path1 = father_path + '/Outcomes/low_D/' + path_str

    # theta = theta
    theta = np.array(pd.read_csv(path1 + 'cluster_theta.csv', index_col=0))

    # X = contexts + group label
    X = np.array(pd.read_csv(path0 + 'cluster_X.csv', index_col=0))

    # initialize T agents
    for i in range(T):
        my_user = User(feature=X[i, 0:p], tru_gp=X[i, p])
        my_user.set_id(i)
        user_list[i] = my_user

    # update cluster label for agents
    cluster_pred = np.array(pd.read_csv(path0 + 'cluster_pred.csv', index_col=0))
    for i in range(T):
        user_list[i].set_group(cluster_pred[i])

    # record cluster label
    for i in range(n):
        cluster_list[i] = [j for j, x in enumerate(cluster_pred) if x == i]

def read_single_csv(input_path):
    """
    Read csv file, accelerate when the file size is huge
    :param input_path: str; File path
    :return: csv; Csv data
    """
    df_chunk = pd.read_csv(input_path, chunksize=1000, index_col=0)
    res_chunk = []
    for chunk in df_chunk:
        res_chunk.append(chunk)
    res_df = pd.concat(res_chunk)
    res_df = res_df.to_numpy()
    return res_df

def renew_gp(gp: int):
    """
    Get the group label of gp-th cluster label
    :param gp: int; Cluster label
    :return: int; Group label
    """
    stat = np.array(pd.read_csv(path0 + 'cluster_stat.csv', index_col=0))
    return np.argsort(stat[gp])[-1]

def g(x):
    """
    Individual preference function
    :param x: ndarray; Context
    :return: int; Individual preference
    """
    tmp = 0
    for i in range(p):
        if (abs(x[i]) >= 2 * i / p):
            tmp += 1
    tmp = int(m * tmp / p / 2)
    return tmp

def Y(my_user: User, D: int):
    """
    Reward function
    :param my_user: User; The agent who receives reward
    :param D: int; Agent choice
    :return: int; Reward
    """
    # U = random noise
    U = np.random.binomial(m, 1 / 2) - int(m / 2)
    U = int(U / 2)

    # gp = group label
    gp = int(my_user.tru_gp)

    # x = context
    x = my_user.feature

    ans = theta[gp, D] + g(x) + U
    return ans

def lin_UCB(gp: int, tau):
    """
    Simulate LinUCB racing for agents in cluster label gp
    :param gp: int; Cluster label
    :param tau: float; LinUCB threshold coefficient
    :return: None
    """
    # theta_store = predicted rewards
    theta_store = np.array(['theta_store'])

    # psi = recommended records
    psi = np.zeros([1, 3])
    gp_cluster_id = cluster_list[gp]

    # j = beginning position of racing stage
    j = np.array(pd.read_csv(path1 + str(gp) + 'cluster_sample_j.csv', index_col=0))
    j = j[0, 0]

    # rang = BIC racing rounds
    race_psi = np.array(pd.read_csv(path1 + str(gp) +'cluster_race_psi.csv', index_col=0))
    rang = race_psi.shape[0] - 1 + j

    # A, b are variables in LinUCB
    A = list()
    b = list()

    # initialize A, b
    for i in range(m):
        A.append(np.identity(p))
        b.append(np.zeros([p, 1]))

    while (j < rang):
        # tmp_psi = newly added data in each round
        tmp_psi = list()

        # compare = threshold
        compare = np.zeros([m])

        # get agent id
        id_j = gp_cluster_id[j]

        # tmp_user = the agent receiving recommendation at this round
        tmp_user = user_list[id_j]

        # Xj = context
        Xj = tmp_user.feature.reshape(p, 1)

        # cc = estimated rewards
        cc = np.zeros([m])

        # run LinUCB to make recommendation for the agent
        for i in range(m):
            A_inv = np.linalg.inv(A[i])
            hat_theta = np.dot(A_inv, b[i])

            # pp = thereshold
            pp = np.dot(hat_theta.T, Xj) + tau * math.sqrt(np.dot(np.dot(Xj.T, A_inv), Xj))

            compare[i] = pp
            cc[i] = np.dot(hat_theta.T, Xj)

        max_arm = int(np.argmax(compare))
        tmp_user.buy(max_arm)
        tmp_Y = Y(my_user=tmp_user, D=max_arm)
        tmp_user.set_reward(tmp_Y)
        tmp_psi.append([id_j, tmp_user.choice, tmp_user.reward])

        # update LinUCB parameters
        A[max_arm] = A[max_arm] + np.dot(Xj, Xj.T)
        b[max_arm] = b[max_arm] + tmp_Y * Xj

        # record racing data and estimated rewards
        theta_store = np.hstack((theta_store, cc))
        psi = np.vstack((psi, np.array(tmp_psi)))
        j += 1

    # sve results of recommended records and predicted rewards
    DF = pd.DataFrame(psi)
    DF.to_csv(str(gp) +'UCB_psi.csv')
    DF = pd.DataFrame(theta_store)
    DF.to_csv(str(gp) +'UCB_theta_store.csv')

if __name__ == '__main__':
    # fix random seed for reproducibility
    random.seed(15)
    np.random.seed(4)

    # initialize parameters (below is only for low dimension)
    set_global(pvalue=5, mvalue=16, Tvalue=6000, nvalue=3, kvalue=10)

    # tau = LinUCB threshold coefficient
    tau = 1/m

    # RF race for n groups
    for i in range(n):
        tru_gp = renew_gp(i)
        lin_UCB(gp=tru_gp, tau=tau)
