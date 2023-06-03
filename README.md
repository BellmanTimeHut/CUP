# CUP: Constrained Update Projection Algorithms for Safety Robot Control
Code to reproduce the experiments in CUP: Constrained Update Projection Algorithms for Safety Robot Control



## Installation
1. Clone `CUP`
```
git clone https://github.com/BellmanTimeHut/CUP.git
```
2. Installations of [MuJoCo](https://github.com/deepmind/mujoco) and [SafetyGym](https://github.com/openai/safety-gym) are needed. 
3. Create a conda environment
```
cd CUP
conda env create -f environment.yml
conda activate CUP
```

## Usage
To get start with CUP, run:
```
python main.py --file-prefix 'Test' --env-id Ant-v3 --algo CUP --constraint velocity --cost-lim 103.12 --seed 1024
```


## Hyperparameters

To obtain the results in our experiments, the following hyperparameters are the required.


| Hyperparameter                   | CUP  | PPO-L | TRPO-L | CPO   | FOCOPS |
|---------------------------------|------|-------|--------|-------|--------|
| No. of hidden layers            | 2    | 2     | 2      | 2     | 2      |
| No. of hidden nodes             | 64   | 64    | 64     | 64    | 64     |
| Activation                      | tanh | tanh  | tanh   | tanh  | tanh   |
| Initial log std                 | -0.5 | -0.5  | -1     | -0.5  | -0.5   |
| Discount for reward $\gamma$           | 0.99 | 0.99  | 0.99   | 0.99  | 0.99   |
| Discount for cost $\gamma_C$           | 0.99 | 0.99  | 0.99   | 0.99  | 0.99   |
| Batch size                      | 5000 | 5000  | 5000   | 5000  | 5000   |
| Minibatch size                  | 64   | 64    | N/A    | N/A   | 64     |
| No. of optimization epochs      | 10   | 10    | N/A    | N/A   | 10     |
| Maximum episode length          | 1000 | 1000  | 1000   | 1000  | 1000   |
| GAE parameter (reward)          | 0.95 | 0.95  | 0.95   | 0.95  | 0.95   |
| GAE parameter (cost)            | 0.95 | 0.95  | 0.95   | 0.95  | 0.95   |
| Learning rate schedule for Actor-Critic | True | False | False | False | True |
| Learning rate for policy | $3\times10^{-4}$ | $3\times10^{-4}$  | N/A  | N/A | $3\times10^{-4}$ |
| Learning rate for reward value net | $3\times10^{-4}$ | $3\times10^{-4}$  | $3\times10^{-4}$   | $3\times10^{-4}$  | $3\times10^{-4}$ |
| Learning rate for cost value net | $3\times10^{-4}$ | $3\times10^{-4}$ | $3\times10^{-4}$ | $3\times10^{-4}$ | $3\times10^{-4}$ |
| Learning rate for $\nu$            | 0.01 | 0.01  | 0.01   | N/A   | 0.01   |
| Clip gradient norm | True | False | False | False | True |
| L2-regularization coeff. for value net | $10^{-3}$ | $3\times10^{-3}$  | $3\times10^{-3}$  | $3\times10^{-3}$  | $10^{-3}$   |
| Clipping coefficient            | N/A  | 0.2   | N/A    | N/A   | N/A    |
| Damping coeff.                  | N/A  | N/A   | 0.01   | 0.01  | N/A    |
| Backtracking coeff.             | N/A  | N/A   | 0.8    | 0.8   | N/A    |
| Max backtracking iterations     | N/A  | N/A   | 10     | 10    | N/A    |
| Max conjugate gradient iterations | N/A | N/A   | 10     | 10    | N/A    |
| Iterations for training value net | 1    | 1     | 1      | 80    | 1      |
| Temperature $\lambda$                  | 1.5  | N/A   | N/A    | N/A   | 1.5    |
| Trust region bound $\delta$            | 0.02 | N/A   | 0.01   | 0.01  | 0.02   |
| Initial $\nu , \nu_{max}$                  | 0, 2 | 0, 1  | 0, 2   | N/A   | 0, 2   |

#### Additional max gradient norm clip for our experiments

| Hyperparameter | Default for MuJoCo and SafetyGym | (Swimmer-v3) |  (HumanoidCircle-v0)|
| -------------- | ------- | ------------ |  ------------------ | 
| Critic grad norm (Reward) | 0.5 | 0.5 | 0.5 |
| Critic grad norm (Cost) | 0.5 | 0.5 | 0.5 |
| Actor grad norm in policy improvement | 0.5 | 0.5 |  0.5 |
| Actor grad norm in projection | 0.5 | 10 |  10 |
