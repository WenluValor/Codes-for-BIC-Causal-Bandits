"""Random Forest in contextual bandit """

# Author: Wenlu Xu wenluxu@ucla.edu

import random
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
    path0 = father_path + '/Outcomes/high_D/'
    path1 = father_path + '/Outcomes/high_D/' + path_str
    # use this in low_D
    '''
    path0 = father_path + '/Outcomes/low_D/'
    path1 = father_path + '/Outcomes/low_D/' + path_str
    '''

    # theta = theta
    theta = np.array(pd.read_csv(path1 + 'cluster_theta.csv', index_col=0))

    # X = contexts + group label
    X = np.array(pd.read_csv(path0 + 'cluster_X.csv', index_col=0))
    for i in range(T):
        my_user = User(feature=X[i, 0:p], tru_gp=X[i, p])
        my_user.set_id(i)
        user_list[i] = my_user

    # cluster_pred = cluster label
    cluster_pred = np.array(pd.read_csv(path0 + 'cluster_pred.csv', index_col=0))

    # update cluster label for agents
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

def RF_reco(gp: int, sample_psi):
    """
    Simulate RF racing for agents in cluster label gp, initialize RF model using sampling historical data
    :param gp: int; Cluster label
    :param sample_psi: ndarray; Sampling historical data
    :return: None
    """
    # initialize RF model using sample_psi
    PSI0 = sample_psi.shape[0]
    x = np.zeros([PSI0, p + 1])
    y = np.zeros(PSI0)
    for j in range(PSI0):
        ind = int(sample_psi[j, 0])
        x[j, 0:p] = X[ind, 0:p]
        x[j, p] = sample_psi[j, 1]
        y[j] = sample_psi[j, 2]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / T, random_state=15)
    std = StandardScaler()
    x_train = std.fit_transform(x_train)

    forest = RandomForestRegressor(n_estimators=1,
                                   criterion='squared_error',
                                   random_state=1, warm_start=True)
    forest.fit(x_train, y_train)

    # j = beginning position of racing stage
    j = np.array(pd.read_csv(path1 + str(gp) + 'cluster_sample_j.csv', index_col=0))
    j = j[0, 0]

    # rang = BIC racing rounds
    race_psi = np.array(pd.read_csv(path1 + str(gp) + 'cluster_race_psi.csv', index_col=0))
    rang = race_psi.shape[0] - 1 + j
    gp_cluster_id = cluster_list[gp]

    # theta_store = predicted rewards
    theta_store = np.array(['theta_store'])

    # psi = recommended records
    psi = np.zeros([1, 3])
    while (j < rang - k):
        # tmp_psi = newly added data in each round
        tmp_psi = list()

        # recommend items to the next k agents
        for i in range(k):
            # get agent id
            id_j = gp_cluster_id[j]

            # tmp_user = the agent receiving recommendation at this round
            tmp_user = user_list[id_j]

            # run RF model to make recommendation for the agent
            ans = get_pred(tmp_user.feature, forest)
            max_arm = ans[1]
            est = ans[0]
            tmp_user.buy(max_arm)
            tmp_Y = Y(my_user=tmp_user, D=max_arm)
            tmp_user.set_reward(tmp_Y)

            # record data
            tmp_psi.append([id_j, tmp_user.choice, tmp_user.reward])
            theta_store = np.hstack((theta_store, est))
            j += 1

        # further train RF model on newly added data
        x = np.zeros([k, p + 1])
        y = np.zeros(k)
        for i in range(k):
            ind = int(tmp_psi[i][0])
            x[i, 0:p] = X[ind, 0:p]
            x[i, p] = tmp_psi[i][1]
            y[i] = tmp_psi[i][2]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / T, random_state=15)
        std = StandardScaler()
        x_train = std.fit_transform(x_train)
        forest.n_estimators += 1
        forest.fit(x_train, y_train)

        # record racing data
        psi = np.vstack((psi, np.array(tmp_psi)))

    # save results of recommended records and predicted rewards
    DF = pd.DataFrame(psi)
    DF.to_csv(str(gp) + 'RF_psi.csv')
    DF = pd.DataFrame(theta_store)
    DF.to_csv(str(gp) + 'RF_theta_store.csv')
    return

def get_pred(feature, forest):
    """
    Use RF model to predict rewards for certain context
    :param feature: ndarray; Context used to predict rewards
    :param forest: RF model; RF Model used to predict rewards
    :return: size 2 list; res[0] = max predicted reward, res[1] = best predicted arm
    """
    # ans_list = predicted rewards for all the arms
    ans_list = [None] * m

    # res = returned result
    res = [None] * 2

    # use model to predict rewards for all the arms
    for i in range(m):
        w1 = np.hstack((feature, i)).reshape(1, -1)
        std = StandardScaler()
        x_pred = std.fit_transform(w1)
        ans_list[i] = forest.predict(x_pred)

    # sort the max reward
    ans = max(ans_list)
    res[0] = ans

    # find the arm with the max reward
    for i in range(m):
        if (ans_list[i] == ans):
            res[1] = i
            return res
    return

if __name__ == '__main__':
    # fix random seed for reproducibility
    random.seed(15)
    np.random.seed(4)

    # initialize parameters (below is for high dimension)
    set_global(pvalue=20000, mvalue=16, Tvalue=30000, nvalue=3, kvalue=100)

    # RF race for n groups
    for i in range(n):
        tru_gp = renew_gp(i)
        sample_psi = np.array(pd.read_csv(path1 + str(tru_gp) + 'cluster_sample_psi.csv', index_col=0))
        RF_reco(gp=tru_gp, sample_psi=sample_psi)
