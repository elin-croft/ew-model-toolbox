import math

def hot_suppress_sample(coef, cnt, total):
    """
    Parameters:
    coef: coefficient for hot suppress
    cnt: current item count in dataset
    total: total item count in dataset
    """
    freq = cnt / total
    # rate might bigger than 1
    rate = (math.sqrt(freq/coef) + 1)*(coef/freq)
    return rate