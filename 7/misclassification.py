from scipy.special import comb
import math

def ensemble_error(n_classifier, error):
    k_start = int(math.ceil(n_classifier / 2))
    prob = 0
    for k in range(k_start, n_classifier):
        prob += comb(n_classifier, k) * (error**k) * (1-error)**(n_classifier-k)

    return prob

"""
n_classifier = 11
error = 0.25
print(ensemble_error(n_classifier,error))
"""
