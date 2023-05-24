# Codes-for-BIC-Causal-Bandits

This project contains codes and outcomes of the simulation in A Contextual Bayesian-Incentive-Compatible Algorithm for Causal Bandit in Recommender Systems.

We simulate on two datasets (high_D, low_D) using different $m, k$. 'low_D' stands for low dimensional; 'high_D' stands for high dimensional.

The algorithms studied are

-   BIC sampling and racing (both high_D and low_D);

-   Lin UCB (only for low_D);

-   Random Forest (both high_D and low_D).

All of the three algorithms are conducted based on SBM clustering.

## Data

The training dataset 'cluster_X.csv' can be created through 'create_X()' function in 'cluster_sample_race.py'. The created file is saved in the root directory.

For low_D, 'cluster_X.csv' is uploaded to the directory './Outcomes/low_D'.

For high_D, 'cluster_X.csv' is 11.43 GB thus not uploaded to the repository. The user can generate 'cluster_X.csv' using function 'create_X()' in 'cluster_sample_race.py', which takes 10-20 minutes in the Mac Book Pro with M1 chip.

## Setup

We recommend to run in the following order:

1.  'cluster_sample_race.py'

2.  'cluster_linUCB.py' or 'random_forest.py'

3.  'plot.py'.

### Before running 'cluster_sample_race.py'

**First,** move 'cluster_X.csv' to the root directory (the same directory as 'cluster_sample_race.py').

For low_D, 'cluster_X.csv' is located under the directory './Outcomes/low_D'; for high_D, generate 'cluster_X.csv' using function 'create_X()' in 'cluster_sample_race.py'.

**Second,** move 'cluster_pred.csv' and 'cluster_stat.csv' to the root directory.

Interested user can generate through commented out codes in 'cluster()'. Both files have been uploaded to the directory './Outcomes/low_D' and './Outcomes/high_D'.

There is no need for 'cluster_adj.csv' once we have 'cluster_pred.csv'. Plus, due to size limit, we do not upload them to the repository (low_D is 144 MB; high_D is 3.6 GB). Still, interested user can generate the adjacency matrix 'cluster_adj.csv' through 'create_adj()' in 'cluster_sample_race.py'.

**Third,** set global parameters properly, see in detail in the comments of 'set_global()' in 'cluster_sample_race.py'.

**Fourth,** run 'cluster_sample_race.py'.

Running 'cluster_sample_race.py' will produce several files in the root directory. Except for the files we move before running, move all of the output files into the corresponding directory e.g., './Outcomes/low_D/m4k5' before another new round, otherwise the files will be overwritten. The user can modify the save path to avoid moving files frequently.

### Before running 'cluster_linUCB.py' or 'random_forest.py'

**First,** move 'cluster_X.csv', 'cluster_pred.csv' and 'cluster_stat.csv' to the directory './Outcomes/low_D' or './Outcomes/high_D'.

**Second,** make sure the corresponding directories contain all of the output files from 'cluster_sample_race.py', e.g., './Outcomes/low_D/m4k5' should contain all the output files.

**Third,** set global parameters properly, see in detail in the comments of 'set_global()' in 'cluster_linUCB.py' or 'random_forest.py'.

**Fourth,** run 'cluster_linUCB.py' or 'random_forest.py'.

Again, running these two py files can produce several output files in the root directory. The user should move all th output files into the corresponding directory e.g., './Outcomes/low_D/m4k5' before another new round, otherwise the files will be overwritten. The user can modify the save path to avoid moving files frequently.

### Before running 'plot.py'

**First,** make sure the corresponding directories contain all of the output files from 'cluster_sample_race.py', and 'cluster_linUCB.py' e.g., './Outcomes/low_D/m4k5' should contain all the output files from these two py files.

**Second,** set global parameters properly, see in detail in the comments of 'set_global()' in 'plot.py'.

**Third,** run 'plot.py'.

The output figures will be saved in the directory './Figures'.

## Running time

The tests have been carried out on a Mac Book Pro with M1 chip (8 cores, 16GB RAM), using CPU only.

The clustering in 'cluster_sample_race.py' in high_D takes 2-3 hours. The BIC racing in 'cluster_sample_race.py' in high_D with $m=16, k=100$ takes 12 hours, $m=16, k=50$ takes 6 hours. During this period, three cores are mostly used.

## Results

The incentive indicator of BIC can be viewed in the directory './Figures' with 'lm\*.jpg' and ''m\*.jpg', where 'l' in prefix stands for low_D.

The regret of BIC can be viewed in the directory './Figures' with '\*\*\*r.jpg', where 'r' in sufffix stands for regret.

The regret of BIC, RF, LinUCB can be viewed in the directory './Figures' with '\*\*\*c.jpg', where 'c' in suffix stands for comparisons.
