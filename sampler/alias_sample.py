import numpy as np
class AliasSample:
    def __init__(self):
        pass
    
    def set_alias(self, prob):
        small, large = [], []
        if not isinstance(prob, np.ndarray):
            prob = np.array(prob)
        k = len(prob)
        prob = k * prob / np.sum(prob)
        index = np.zeros(k, dtype=np.int64)
        for i, p in enumerate(prob):
            if p < 1:
                small.append(i)
            else:
                large.append(i)
        
        while small and large:
            small_idx, large_idx = small.pop(), large.pop()

            # fill the prob of samll index with part of the prob of large index
            index[small_idx] = large_idx
            prob[large_idx] = prob[large_idx] - (1 - prob[small_idx])

            if prob[large_idx] < 1:
                small.append(large_idx)
            else:
                large.append(large_idx)
        return index.tolist(), prob.tolist()
    
    def draw_alias(self, index, prob):
        k = len(index)
        idx = np.random.randint(0, k)
        if np.random.rand() < prob[idx]:
            return idx
        else:
            return index[idx]


if __name__ == '__main__':
    prob = [0.1, 0.2, 0.3, 0.4]
    alias = AliasSample()
    index, prob = alias.set_alias(prob)
    print(index, prob)
    from collections import Counter
    cnt = Counter()
    try_times = 100000
    for i in range(try_times):
        idx = alias.draw_alias(index, prob)
        cnt[idx] += 1
    for k, v in cnt.items():
        print(f"prob of index {k}: {v / try_times}")
    