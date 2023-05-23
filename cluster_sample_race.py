"""BIC sampling and racing"""

# Author: Wenlu Xu wenluxu@ucla.edu

import math
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from scipy.stats import skellam

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

def set_global(pvalue: int, mvalue: int, Tvalue: int, Kvalue: int, nvalue: int, gprobvalue: np.array):
    """
    Initialize public parameters
    :param pvalue: int, 5, 20000; Data dimensions
    :param mvalue: int, 4, 8, 16; Number of arms
    :param Tvalue: int, 6000, 30000; Number of agents
    :param Kvalue: int; K-fold size in DML
    :param nvalue: int, 3; Number of groups
    :param gprobvalue: ndarray; Probability matrix
    :return: None
    """
    global p, m, T, p_i, K, X, theta, n, gprob, adj, user_list, cluster_list
    n = nvalue
    p = pvalue
    m = mvalue
    T = Tvalue
    K = Kvalue
    gprob = gprobvalue

    # adj = adjacency matrix
    adj = np.zeros([T, T])

    # p_i = theta_0 / m
    p_i = np.zeros([n, m])
    for i in range(n):
        p_i[i] = np.random.uniform(0, 1, size=m)

    # generate theta according to the Binomial disrtibution
    theta = create_theta()

    # save prior and theta
    DF = pd.DataFrame(p_i)
    DF.to_csv('cluster_p_i.csv')
    DF = pd.DataFrame(theta)
    DF.to_csv('cluster_theta.csv')

    # user_list = T agents
    user_list = [None] * T

    # cluster_list = agents with cluster label n
    cluster_list = [None] * n

    # run this if there is no cluster_X.csv file in the directory
    '''
    create_X()
    '''

    # this line takes 2-3 minutes in high_D
    X = read_single_csv('cluster_X.csv')

    # initialize T agents
    for i in range(T):
        my_user = User(feature=X[i, 0:p], tru_gp=X[i, p])
        my_user.set_id(i)
        user_list[i] = my_user

    # run this if there is no cluster_adj.csv file in the directory
    '''
    create_adj()
    '''

    # run this if there is no cluster_pred.csv file in the directory
    '''
    adj = read_single_csv('cluster_adj.csv')
    '''

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

def create_X():
    """
    Generate contexts in each group
    :return: None
    """
    # X = contexts + group label, X[:, 0:p] = context, X[:, p] = group label
    X = np.zeros([T, p+1])

    # Agents number in each group
    Tn = int(T / n)

    # generate contexts and group label
    for i in range(n):
        if i == n-1:
            next = T
        else: next = (i+1)*Tn
        size = next - i*Tn
        var = 2 + (-1)**i
        for j in range(i*Tn, next):
            # use this in low_D
            '''
            X[j, 0:p] = np.random.normal(loc=(-1)**i * i, scale=var, size=p)
            '''
            X[j, 0:p] = np.random.normal(loc= i, scale=var, size=p)
        for j in range(size):
            X[i*Tn + j, p] = i

    # shuffle agents
    np.random.shuffle(X)

    # save results
    DF = pd.DataFrame(X)
    DF.to_csv('cluster_X.csv')
    return

def create_theta():
    """
    Generate theta in each group by Binomial distribution
    :return: ndarray; Theta
    """
    theta = np.zeros([n, m])

    # generate theta
    for i in range(n):
        for j in range(m):
            theta[i, j] = np.random.binomial(m, p_i[i, j])
        max_arm = max(theta[i])

        # only keep one max arm for uniqueness of optimal arm
        ind = 0
        for j in range(m):
            if (theta[i, j] == max_arm):
                theta[i, j] -= 1
                ind = j
        theta[i, ind] += 1
    return theta

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

def create_adj():
    """
    Generate adjacency matrix according to probability matrix
    :return: None
    """
    for i in range(T):
        for j in range(i, T):
            # two nodes have probability gprob to link
            U = np.random.binomial(1, gprob[int(X[i, p]), int(X[j, p])])
            adj[i, j] = U
            adj[j, i] = U

        # diagonal elements = 1
        adj[i, i] = 1

    # save results
    DF = pd.DataFrame(adj)
    DF.to_csv('cluster_adj.csv')
    return

def get_affinity():
    """
    Calculate population version of adjacency matrix according to SBM clustering
    :return: ndarray; Population version of adjacency matrix
    """
    mat_D = np.zeros([T])
    mat_L = np.zeros([T, T])
    for i in range(T):
        mat_D[i] = 1 / math.sqrt(sum(adj[i]))

    for i in range(T):
        for j in range(i, T):
            mat_L[i, j] = mat_D[i] * mat_D[j] * adj[i, j]
            mat_L[j, i] = mat_L[i, j]

    return mat_L

def cluster():
    """
    Spectural clustering for SBM
    :return: None
    """

    # run this if there is no cluster_pred.csv file in the directory
    '''
    mat_L = get_affinity()
    cluster_pred = SpectralClustering(n_clusters=n, random_state=0).fit_predict(mat_L)
    DF = pd.DataFrame(cluster_pred)
    DF.to_csv('cluster_pred.csv')
    '''

    # cluster_pred = cluster label
    cluster_pred = np.array(pd.read_csv('cluster_pred.csv', index_col=0))

    # set cluster label for T agents
    for i in range(T):
        user_list[i].set_group(cluster_pred[i])

    # record cluster label
    for i in range(n):
        cluster_list[i] = [j for j, x in enumerate(cluster_pred) if x == i]

    # run this if there is no cluster_stat.csv file in the directory
    '''
    stat = np.zeros([n, n])
    for j in range(T):
        i = int(user_list[j].cal_gp)
        l = int(user_list[j].tru_gp)
        stat[i, l] += 1
    DF = pd.DataFrame(stat)
    DF.to_csv('cluster_stat.csv')
    '''
    return

def renew_gp(gp: int):
    """
    Get the group label of gp-th cluster label
    :param gp: int; Cluster label
    :return: int; Group label
    """
    stat = np.array(pd.read_csv('cluster_stat.csv', index_col=0))
    return np.argsort(stat[gp])[-1]

def sample(k: int, gp: int, c_t):
    """
    BIC sampling stage
    :param k: int, 5/10 low_D, 50/100 high_D ; Sampling size
    :param gp: int; Cluster label
    :param c_t: float; Parameter in BIC sampling
    :return: None
    """
    gp_cluster_id = cluster_list[gp]
    gp_p_i = p_i[gp]
    gp_theta = theta[gp]
    arm_rank = list(np.argsort(-gp_p_i))
    # arm_rank[i] = j means j-th arm has i-th rank reward (j=0 the biggest prior reward)
    # arm_rank.index(i) = j means i-th arm is j-th rank

    # psi = historical data
    psi = np.zeros([k, 6]) # psi = [id, D, Y, E_happen, q_t, arm_rank]
    j = 0 # position
    last_pi = np.zeros([m, 2]) # \pi calculated in the previous round
    last_pi[0, 0] = 1
    last_pi[0, 1] = 1

    # recommend rank-0 arm (with the biggest prior reward)
    for i in range(k):
        id_j = gp_cluster_id[j]
        tmp_user = user_list[id_j]
        tmp_arm = arm_rank[0]
        tmp_user.buy(tmp_arm)
        tmp_user.set_reward(Y(my_user=tmp_user, D=tmp_arm))
        psi[i] = [tmp_user.id, tmp_user.choice, tmp_user.reward, True, 1, 0]
        j += 1

    # sample in phase i, i = 1, 2, ..., m-1
    for i in range(1, m):
        count = 0
        tmp_psi = list()
        q_t = get_q_t(rank=i, gp_theta=gp_theta, gp_p_i=gp_p_i, c_t=c_t, arm_rank=arm_rank)
        q_t = max((2*m-i)*k/T*n, q_t) # to accelerate racing in case q_t is too small
        # use this in low_D
        '''
        q_t = max(5*(2*m-i)*k/T*n, q_t)
        '''

        # calculate \hat{E}_{i, t}, the event
        E_happen = get_E_happen(rank=i, arm_rank=arm_rank, psi=psi, last_pi=last_pi, gp_p_i=gp_p_i, c_t=c_t, k=k)

        # save the \pi in this round for future use
        last_pi = get_sample_pi(psi, last_pi, i)

        # sample k agents for each arm
        while (count < k):
            if (E_happen):
                i_star = arm_rank[i]
            else:
                i_star = arm_rank[0]
            U = np.random.uniform(0, 1, size=1)

            # flip the coin to determine exploration or exploition
            if (U <= q_t):
                id_j = gp_cluster_id[j]
                tmp_user = user_list[id_j]
                tmp_arm = arm_rank[i]
                tmp_user.buy(tmp_arm)
                tmp_user.set_reward(Y(my_user=tmp_user, D=tmp_arm))
                tmp_psi.append([tmp_user.id, tmp_user.choice, tmp_user.reward, E_happen, q_t, i])
                count += 1
            else:
                id_j = gp_cluster_id[j]
                tmp_user = user_list[id_j]
                tmp_arm = i_star
                tmp_user.buy(tmp_arm)
                tmp_user.set_reward(Y(my_user=tmp_user, D=tmp_arm))
                tmp_psi.append([tmp_user.id, tmp_user.choice, tmp_user.reward, E_happen, q_t, i])
                if (E_happen): count += 1
            j += 1

        # resample collected data and then record
        tmp_psi = resample_psi(tmp_psi=tmp_psi, k=k, special_arm=arm_rank[i])
        psi = np.vstack((psi, np.array(tmp_psi)))

    # save estimated rewards, recommended records, ending position
    theta_hat = est_sample_theta_hat(psi, arm_rank, m, last_pi, k)
    DF = pd.DataFrame(theta_hat)
    DF.to_csv(str(gp) + 'cluster_sample_theta_hat.csv')
    DF = pd.DataFrame(psi)
    DF.to_csv(str(gp) +'cluster_sample_psi.csv')
    DF = pd.DataFrame(np.array(list([j])))
    DF.to_csv(str(gp) +'cluster_sample_j.csv')
    return

def resample_psi(tmp_psi, k, special_arm):
    """
    Pick out all the explored data, and resample (k/m) exploited data
    :param tmp_psi: list of ndarray; sampling data at current round
    :param k: int; Sampling size
    :param special_arm: int; Explored arm
    :return: list of ndarray; Resampled data
    """
    other_num = int(k/m)
    ans = [None] * (k + other_num)
    j = 0

    # pick out all the k explored data
    for i in range(len(tmp_psi)):
        if (tmp_psi[i][1] == special_arm):
            ans[j] = tmp_psi[i]
            j += 1
            if (j == k): break

    # sample (k/m) exploited data
    last_pos = 0
    while (j < k + other_num):
        for i in range(last_pos, len(tmp_psi)):
            if (tmp_psi[i][1] != special_arm):
                ans[j] = tmp_psi[i]
                j += 1
                last_pos = i+1
                break

            # in case the exploited arm is the same as explored arm
            if (i == len(tmp_psi) - 1):
                ans = [ii for ii in ans if ii is not None]
                return ans
    return ans

def get_q_t(rank: int, gp_theta, gp_p_i, c_t, arm_rank):
    """
    Calculate q_t
    :param rank: int; Number of phase until now
    :param gp_theta: ndarray; True theta
    :param gp_p_i: ndarray; Prior
    :param c_t: float; Parameter in BIC sampling
    :param arm_rank: ndarray; Rank-arm matching
    :return: float; q_t
    """
    P = 1

    for j in range(rank):
        sum = 0
        ind_j = arm_rank[j]
        # here we use gp_theta just for a concise expression for P(C_2)
        # actually gp_theta is unknown to the principal, yet P(C_2) can be derived using Monte Carlo
        for k in range(int(gp_theta[ind_j] - 9 / 10 * c_t + 1)):
            # use Poisson distribution to approximate Binomial distribution
            sum += poisson(lam=m * gp_p_i[ind_j], k=k)
            P *= sum

    for j in range(1, rank):
        ind_j = arm_rank[j]
        ind_0 = arm_rank[0]
        # The subtraction between two Poisson variables follows Skellam distribution
        sum = 1 - skellam.cdf(c_t, m * gp_p_i[ind_j], m * gp_p_i[ind_0])
        P *= sum

    q_t = 1 - 1 / (1 + 1/10 * c_t * P)
    return q_t

def poisson(lam, k: int):
    """
    Get Poisson CDF
    :param lam: float; Parameters of Poisson distribution
    :param k: int; Parameters of Poisson distribution
    :return: float; Poisson CDF
    """
    ans = math.pow(lam, k) / math.factorial(k) * math.exp(-lam)
    return ans

def get_E_happen(rank: int, arm_rank, psi, last_pi, gp_p_i, c_t, k):
    """
    Caculate the event \hat{E}_{i, t} happens or not
    :param rank: int; Number of sampling phase
    :param arm_rank: ndarray; Rank-arm matching
    :param psi: ndarray; Historical data
    :param last_pi: ndarray; \pi calculated in the last round
    :param gp_p_i: ndarray; Prior
    :param c_t: float; Parameters in BIC sampling
    :param k: int; Sampling size
    :return: bool; \hat{E}_{i, t} happens or not
    """
    happen = False

    # get DML estimator of \hat{\theta} using historical data
    theta_hat = est_sample_theta_hat(psi, arm_rank, rank, last_pi, k)

    ind_arm = arm_rank[rank]
    for i in range(1, rank):
        ind_i = arm_rank[i]
        ind_0 = arm_rank[0]
        if (theta_hat[ind_i] - theta_hat[ind_0] >= 2/5 * c_t):
            happen = True
        else:
            happen = False
            break

    theta_hat -= theta_hat[0]
    theta_hat += m * gp_p_i[0]
    for i in range(rank):
        ind_i = arm_rank[i]
        if (gp_p_i[ind_arm] * m - theta_hat[ind_i] >= 2/5 * c_t):
            happen = True
        else:
            happen = False
            break

    return happen

def est_sample_theta_hat(psi, arm_rank, rank: int, last_pi, k):
    """
    DML estimator for \hat{\theta} in sampling stage
    :param psi: ndarray; Historical data
    :param arm_rank: ndarray; Rank-arm matching
    :param rank: int; Number of sampling phase
    :param last_pi: ndarray; \pi calculated in the last round
    :param k: int; Sampling size
    :return: size-m ndarray; DML estimator for \hat{\theta}
    """

    PSI0 = psi.shape[0] # number of data used
    N = int(PSI0 / K) # size of each fold

    # calculate complementary of K index set
    ind_set = np.zeros([K, N])
    for i in range(K):
        for j in range(N):
            ind_set[i, j] = j * K + i

    # ind_set_c = index set that removes additional index
    ind_set_c = np.zeros([K, PSI0 - N])
    for i in range(K):
        tmp_j = 0
        for j in range(N):
            a = list(range(K))
            a.remove(i)
            for l in a:
                ind_set_c[i, tmp_j] = j * K + l
                tmp_j += 1
        for j in range(PSI0 - N * K):
            ind_set_c[i, tmp_j] = N * K + j
            tmp_j += 1

    # K-fold training for h using random forest
    h_list = train_est_h(psi, ind_set_c, k)

    # get estimated \pi through the formula
    new_pi = get_sample_pi(psi, last_pi, rank)

    ans = np.zeros([m])
    for i in range(rank):
        arm_i = arm_rank[i]
        sum = 0
        # calculate DML estimators in each fold
        for kk in range(K):
            pipi = np.zeros([N])
            w1 = np.zeros([N, p + 1])
            for j in range(N):
                tmp_j = int(ind_set[kk][j])
                W0 = user_list[int(psi[tmp_j, 0])].feature
                w1[j] = np.hstack((W0, arm_i)).reshape(1, -1)
                tmp_rank = int(psi[tmp_j, 5]) # current phase in sampling stage
                rec_arm = int(psi[tmp_j, 1])  # recommended arm
                if (rec_arm == arm_rank[0]):
                    pipi[j] = new_pi[tmp_rank, 1]
                else:
                    pipi[j] = new_pi[tmp_rank, 0]

            std = StandardScaler()
            x_pred = std.fit_transform(w1)
            hh = h_list[kk].predict(x_pred)

            for j in range(N):
                tmp_j = int(ind_set[kk][j])
                W1 = psi[tmp_j, 1] # D
                W2 = psi[tmp_j, 2] # Y

                # Neyman score function
                if (W1 == arm_i):
                    coef = hh[j] + 1 / pipi[j] * (W2 - hh[j])
                else:
                    coef = hh[j]

                # rank == m is for the future use of racing stage
                if (rank == m): coef -= hh[j]

                sum += coef

        # since our Neyman score function is linear, the calculation is to take the mean
        sum /= PSI0
        ans[arm_i] = sum
    return ans

def get_sample_pi(psi, last_ans, rank_now: int):
    """
    Calculate \pi according to the formula
    :param psi: ndarray; Historical data
    :param last_ans: ndarray; \pi calculated in the last round
    :param rank_now: int; Number of phase in sampling stage
    :return: size-m*2 ndarray;
    ans[:, 0] = \pi for explored arm;
    arm[:, 1] =  \pi for exploited arm
    """
    ans = last_ans

    PSI0 = psi.shape[0]
    j = 0
    i = rank_now
    if (i == 1):
        return ans
    while (j < PSI0):
        if (psi[j, 5] == i - 1):
            # if E_happen, exploration arm = exploition arm
            if (psi[j, 3]):
                ans[i - 1, 0] = 1
                ans[i - 1, 1] = 0
            else:
                ans[i - 1, 0] = psi[j, 4]
                ans[i - 1, 1] = 1 - psi[j, 4]
            break
        j += 1
    return ans

def get_sample_theta_hat(gp: int, k):
    """
    Record DML estimators using sampling historical data
    :param gp: int; Cluster label
    :param k: int; Sampling size
    :return: None
    """
    # get historical data
    sample_psi = np.array(pd.read_csv(str(gp) + 'cluster_sample_psi.csv', index_col=0))

    # initialize
    gp_p_i = p_i[gp]
    arm_rank = list(np.argsort(-gp_p_i))
    last_pi = np.zeros([m, 2])
    last_pi[0, 0] = 1
    last_pi[0, 1] = 1
    theta_store = np.array(['sample_theta_store'])

    # get DML estimators
    for i in range(1, m):
        psi = sample_psi[np.where(sample_psi[:, 5] < i)]
        theta_hat = est_sample_theta_hat(psi, arm_rank, i, last_pi, k)
        last_pi = get_sample_pi(psi, last_pi, i)

        # save the result
        theta_store = np.hstack((theta_store, theta_hat))

    # save the result
    DF = pd.DataFrame(theta_store)
    DF.to_csv(str(gp) + 'cluster_sample_theta_store.csv')
    return


def race(k: int, gp: int, tau):
    """
    BIC racing stage
    :param k: int; Racing size (= sampling size)
    :param gp: int; Cluster label
    :param tau: float; Paramter in BIC racing
    :return: None
    """
    # initialize
    gp_cluster_id = cluster_list[gp]
    rang = len(gp_cluster_id)
    gp_p_i = p_i[gp]
    arm_rank = list(np.argsort(-gp_p_i))
    theta_store = np.array(['theta_store'])
    l = k
    count = 0
    tmpB = list() # tmpB = the racing pool at current round

    # sample_psi = sampling historical data
    sample_psi = np.array(pd.read_csv(str(gp) +'cluster_sample_psi.csv', index_col=0))

    # sample_theta = DML estimators in the sampling stage, for use in the racing stage
    sample_theta = np.array(pd.read_csv(str(gp) +'cluster_sample_theta_hat.csv', index_col=0))
    sample_theta = sample_theta[:, 0]

    # theta_hat = DML estimators in the sampling stage that subtract baseline, use it to initialize B
    theta_hat = np.array(pd.read_csv(str(gp) +'cluster_sample_theta_hat.csv', index_col=0))
    theta_hat = theta_hat[:, 0]
    theta_hat += (m * gp_p_i[0] - theta_hat[0])

    # j = sampling ending position = racing starting position
    j = np.array(pd.read_csv(str(gp) +'cluster_sample_j.csv', index_col=0))
    j = j[0, 0]

    # theta_store = DML estimators for \hat{\theta} in the racing stage
    theta_store = np.hstack((theta_store, theta_hat))

    # tmp_psi = records of recommended data in current round of the racing stage
    tmp_psi = list()

    # initialize B
    theta_hat_star = max(theta_hat)
    for i in range(m):
        ind_j = arm_rank[i]
        tmpB.append(ind_j)

    for i in range(len(tmpB)):
        for s in range(K):
            id_j = gp_cluster_id[j]
            tmp_user = user_list[id_j]
            tmp_arm = tmpB[i]
            tmp_user.buy(tmp_arm)
            tmp_user.set_reward(Y(my_user=tmp_user, D=tmp_arm))
            tmp_psi.append([id_j, tmp_user.choice, tmp_user.reward, len(tmpB)])
            j += 1
        count += 1
    B = tmpB
    psi = np.array(tmp_psi)
    l += k

    # check = check rounds, use it to accelerate so that we do not run DML at each round, but each 'check' rounds
    check = k/2

    # BIC racing
    while (len(B) > 1):
        tmp_psi = list()
        # do DML only when 'check' data are collected
        if (count > check):
            tmpB = list()
            theta_hat = est_race_theta_hat(sample_psi, sample_theta, psi, B, k)
            theta_store = np.hstack((theta_store, theta_hat))
            theta_hat_star = max(theta_hat)
            trim = trim_c(l, tau, gp)
            for i in B:
                if (theta_hat_star - theta_hat[i] <= trim):
                    tmpB.append(i)
            count = 0
        else:
            tmpB = B

        # collect data of receiving recommendation and rewards
        for i in range(len(tmpB)):
            id_j = gp_cluster_id[j]
            tmp_user = user_list[id_j]
            tmp_arm = tmpB[i]
            tmp_user.buy(tmp_arm)
            tmp_user.set_reward(Y(my_user=tmp_user, D=tmp_arm))
            tmp_psi.append([id_j, tmp_user.choice, tmp_user.reward, len(tmpB)])
            j += 1
            count += 1

        # update
        B = tmpB
        l += max(check, 1)
        psi = np.vstack((psi, np.array(tmp_psi)))

        # in case that T_n agents are used up, break the loop in advance and record the current optimal arm
        if (j >= rang - m):
            for i in B:
                if (theta_hat[i] == theta_hat_star):
                    B = [i]
                    tmpB = [i]
                    break

    # add last data
    tmp_psi = list()
    id_j = gp_cluster_id[j]
    tmp_user = user_list[id_j]
    tmp_arm = tmpB[0]
    tmp_psi.append([id_j, tmp_arm, Y(my_user=tmp_user, D=tmp_arm), len(tmpB)])
    psi = np.vstack((psi, np.array(tmp_psi)))

    # save recommended records and estimated rewards
    DF = pd.DataFrame(psi)
    DF.to_csv(str(gp) +'cluster_race_psi.csv')
    DF = pd.DataFrame(theta_store)
    DF.to_csv(str(gp) +'cluster_race_theta_store.csv')

def est_race_theta_hat(sample_psi, sample_theta, race_psi, B: list, k):
    """
    DML estimator for \hat{\theta} in racing stage. Very similar to the sampling stage.
    The difference is that RF for \hat{h} in racing stage uses combination of sampling data and racing data.
    :param sample_psi: ndarray; Historical data in sampling stage
    :param sample_theta: ndarray; DML estimators for \hat{\theta} using sampling data.
    :param race_psi: ndarray; Historical data in racing stage
    :param B: list; Arms left in the racing pool
    :param k: int; Sampling/ Racing size
    :return: size-m ndarray; \hat{\theta}
    """
    # calculate sampling index and racing index separately
    sample_PSI0 = sample_psi.shape[0]
    race_PSI0 = race_psi.shape[0]
    sample_N = int(sample_PSI0 / K)
    race_N = int(race_PSI0 / K)
    sample_ind_set = np.zeros([K, sample_N])
    race_ind_set = np.zeros([K, race_N])

    for i in range(K):
        for j in range(sample_N):
            sample_ind_set[i, j] = j*K + i
        for j in range(race_N):
            race_ind_set[i, j]= j*K + i

    sample_ind_set_c = np.zeros([K, sample_PSI0 - sample_N])
    race_ind_set_c = np.zeros([K, race_PSI0 - race_N])

    for i in range(K):
        tmp_j = 0
        for j in range(sample_N):
            a = list(range(K))
            a.remove(i)
            for l in a:
                sample_ind_set_c[i, tmp_j] = j * K + l
                tmp_j += 1
        for j in range(sample_PSI0 - sample_N*K):
            sample_ind_set_c[i, tmp_j] = sample_N*K + j
            tmp_j += 1

    for i in range(K):
        tmp_j = 0
        for j in range(race_N):
            a = list(range(K))
            a.remove(i)
            for l in a:
                race_ind_set_c[i, tmp_j] = j * K + l
                tmp_j += 1
        for j in range(race_PSI0 - race_N*K):
            race_ind_set_c[i, tmp_j] = race_N*K + j
            tmp_j += 1

    # get RF models trained on the combination of sampling and racing data
    h_list = train_est_race_h(race_psi, race_ind_set_c, sample_psi, sample_ind_set_c, k)

    ans = np.zeros([m])
    for i in B:
        sum = 0
        # calculate DML estimators in each fold
        for kk in range(K):
            pipi = np.zeros([race_N])
            w1 = np.zeros([sample_N + race_N, p+1])
            for j in range(sample_N):
                tmp_j = int(sample_ind_set[kk][j])
                W0 = user_list[int(sample_psi[tmp_j, 0])].feature
                w1[j] = np.hstack((W0, i)).reshape(1, -1)
            for j in range(sample_N, sample_N + race_N):
                cc = j - sample_N
                tmp_j = int(race_ind_set[kk][cc])
                W0 = user_list[int(race_psi[tmp_j, 0])].feature
                w1[j] = np.hstack((W0, i)).reshape(1, -1)

            std = StandardScaler()
            x_pred = std.fit_transform(w1)
            hh = h_list[kk].predict(x_pred)

            for j in range(sample_N):
                coef = hh[j]
                sum += coef

            for j in range(sample_N, sample_N + race_N):
                cc = j - sample_N
                tmp_j = int(race_ind_set[kk][cc])
                W1 = race_psi[tmp_j, 1]  # D
                W2 = race_psi[tmp_j, 2]  # Y
                pipi[cc] = 1 / int(race_psi[tmp_j, 3]) # \pi is implied by |B| at each round

                # Neyman score function
                if (W1 == i):
                    coef = hh[j] + 1 / pipi[cc] * (W2 - hh[j])
                else:
                    coef = hh[j]
                sum += coef

        ans[i] = (sample_theta[i] * sample_PSI0 + sum) / (sample_PSI0 + race_PSI0)
    return ans

def train_est_h(psi, indset_c, k):
    """
    Train K Random Forest models using part of the data in the sampling stage
    :param psi: ndarray; Complete sampling data being splitted to train RF models
    :param indset_c: ndarray; Index set of sampling dataset that removes K index
    :param k: int; Sampling size, to determine the depth of decision trees
    :return: size-K list of RF models; RF models trained in each fold
    """
    ans = list()

    # train RF model in each fold
    for i in range(K):
        # transform trained data
        x = np.zeros([indset_c[i].shape[0], p+1])
        y = np.zeros(indset_c[i].shape[0])
        for j in range(indset_c[i].shape[0]):
            tmp_user = user_list[int(psi[int(indset_c[i, j]), 0])]
            x[j, 0:p] = tmp_user.feature
            x[j, p] = tmp_user.choice
            y[j] = tmp_user.reward

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/T, random_state=15)
        std = StandardScaler()
        x_train = std.fit_transform(x_train)
        clf = RandomForestClassifier(max_depth=int(math.log(k)), random_state=1, n_jobs=-1)

        # train the model
        clf.fit(x_train, y_train)

        # save the model
        ans.append(clf)
    return ans

def train_est_race_h(race_psi, race_indset_c, sample_psi, sample_indset_c, k):
    """
    Train K Random Forest models using part of the data in the sampling and racing stage
    :param race_psi: ndarray; Complete racing data being splitted to train RF models
    :param race_indset_c: ndarray; Index set of racing dataset that removes K index
    :param sample_psi: ndarray; Complete sampling data being splitted to train RF models
    :param sample_indset_c: ndarray; Index set of sampling dataset that removes K index
    :param k: int; Sampling/ Racing size, to determine the depth of decision trees
    :return: size-K list of RF models; RF models trained in each fold
    """
    ans = list()

    # train RF model in each fold
    for i in range(K):
        x = np.zeros([sample_indset_c[i].shape[0] + race_indset_c[i].shape[0], p+1])
        y = np.zeros(sample_indset_c[i].shape[0] + race_indset_c[i].shape[0])

        # transform trained data in sampling stage
        for j in range(sample_indset_c[i].shape[0]):
            tmp_user = user_list[int(sample_psi[int(sample_indset_c[i, j]), 0])]
            x[j, 0:p] = tmp_user.feature
            x[j, p] = tmp_user.choice
            y[j] = tmp_user.reward

        # transform trained data in racing stage
        for j in range(sample_indset_c[i].shape[0], sample_indset_c[i].shape[0]+race_indset_c.shape[0]):
            cc = j - sample_indset_c[i].shape[0]
            tmp_user = user_list[int(race_psi[int(race_indset_c[i, cc]), 0])]
            x[j, 0:p] = tmp_user.feature
            x[j, p] = tmp_user.choice
            y[j] = tmp_user.reward

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/T, random_state=15)
        std = StandardScaler()
        x_train = std.fit_transform(x_train)
        clf = RandomForestClassifier(max_depth=int(math.log(k)), random_state=1, n_jobs=-1)

        # train the model
        clf.fit(x_train, y_train)

        # save the model
        ans.append(clf)
    return ans

def trim_c(n: int, tau, gp: int):
    """
    Calculate BIC racing threshold C_l
    :param n: int; Number of rounds
    :param tau: float; Parameter in BIC racing
    :param gp: int; Cluster label
    :return: float; BIC racing threshold C_l
    """
    zeta = zeta_tau(tau, gp)
    ans = math.sqrt((8 * math.log(4 * zeta * T)) / n)

    # use this in low_D
    '''
    return ans
    '''
    return ans * 2 * m


def zeta_tau(tau, gp):
    """
    Calculate \zeta_\tau
    :param tau: float; Parameter in BIC racing
    :param gp: int; Cluster label
    :return: float; \zeta_\tau
    """
    ans = 4 * (m**2) / (tau * get_monte_P(gp, tau))
    return ans

def get_monte_P(gp: int, tau, time=int(1e4)):
    """
    Use Monte Carlo to estimate \min_{i}P(\theta_i - \max_{j \neq i}\theta_j > \tau)
    :param gp: int; Cluster label
    :param tau: float; Parameter in BIC racing
    :param time: int; Monte Carlo size
    :return: float; \min_{i}P(\theta_i - \max_{j \neq i}\theta_j > \tau)
    """
    P = np.zeros([m])
    for i in range(m):
        count = 0
        for j in range(time):
            gp_theta = create_gp_theta(gp)
            max_i = 0
            for l in range(m):
                if (l != i):
                    max_i = max(gp_theta[l], max_i)
            if (gp_theta[i] - max_i >= m * tau):
                count += 1

        # at least 10 counts to accelerate racing
        count = max(10, count)
        P[i] = count / time
    return min(P)

def create_gp_theta(gp: int):
    """
    Derive theta^0 for cluster gp following the same creation process
    :param gp: int; Cluster label
    :return: size-m ndarray; theta^0 for cluster gp
    """
    gp_theta = np.zeros([m])
    gp_p_i = p_i[gp]
    for j in range(m):
        gp_theta[j] = np.random.binomial(m, gp_p_i[j])
    max_arm = max(gp_theta)

    ind = 0
    for j in range(m):
        if (gp_theta[j] == max_arm):
            gp_theta[j] -= 1
            ind = j
    gp_theta[ind] += 1
    return gp_theta

if __name__ == '__main__':
    # fix random seed for reproducibility
    random.seed(15)
    np.random.seed(4)

    # initialize parameters (below is for high_D)
    prob = np.array([[1, 0.2, 0.4], [0.2, 0.8, 0.6], [0.4, 0.6, 0.9]])
    set_global(pvalue=2 * 10000, mvalue=4, Tvalue=3 * 10000, Kvalue=5, nvalue=3, gprobvalue=prob)
    # For low_D, use this
    '''
    low_D: set_global(pvalue=5, mvalue=4, Tvalue=6 * 1000, Kvalue=5, nvalue=3, gprobvalue=prob)
    '''

    # do the clustering
    cluster()

    # set sampling size
    # low_D: k_num = 5/10
    # high_D: k_num = 50/100
    k_num = 100

    # BIC sampling and racing for n groups
    for i in range(n):
        tru_gp = renew_gp(i)
        # BIC sampling
        sample(k=k_num, gp=tru_gp, c_t=2/5)

        # record estimated rewards in sampling stage
        get_sample_theta_hat(tru_gp, k_num)

        # BIC racing
        race(k=k_num, gp=tru_gp, tau=1/T)
    exit(0)
