import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from vmdpy import VMD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from antropy import sample_entropy, spectral_entropy, app_entropy # 使用antropy库计算样本熵
from scipy.linalg import svd
from scipy.signal import hilbert

file = pd.read_excel("../2010-2019-15-140.xlsx")
data = file['shuju'].values


def envelope_entropy(imfs):
    entropy_values = []
    for imf in imfs:
        analytic_signal = hilbert(imf)
        envelope = np.abs(analytic_signal)
        envelope = envelope / np.sum(envelope)
        entropy = -np.sum(envelope * np.log(envelope + 1e-10))
        entropy_values.append(entropy)

    return np.mean(entropy_values)




# 组合适应度函数
def combined_fitness(alpha, k, weights=[1]):
    alpha = int(alpha)
    k = int(k)
    imfs, _, _ = VMD(data, alpha=alpha, tau=0, K=k, DC=0, init=1, tol=1e-7)

    env_entropy = envelope_entropy(imfs)

    scaler = MinMaxScaler()

    normalized_metrics = scaler.fit_transform(
        np.array([env_entropy]).reshape(-1, 1)
    ).flatten()

    fitness_value = np.dot(weights, normalized_metrics)
    return fitness_value


def vmd_target_function(params):
    alpha, k = params
    return combined_fitness(alpha, k)


bounds = [(200, 3000), (3, 15)]

result = differential_evolution(vmd_target_function, bounds, strategy='best1bin', maxiter=100, popsize=15, tol=0.01)

best_alpha, best_k = result.x
print("best alpha and k :", int(best_alpha), int(best_k))



