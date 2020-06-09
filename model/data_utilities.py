import numpy as np
import random

def split_negative_test(test_data, data, neg_pro):
    u_ids = data[:, 0]
    spl_ids = set(data[:, 1])
    test_ids = test_data[:, 0]
    idset = set(test_ids)
    neg_list = []
    for u in idset:
        nb_test = np.count_nonzero(test_ids == u)
        ids = data[u_ids == u, 1]
        if len(ids) > 0:
            samples = random.sample(spl_ids, min(len(spl_ids), nb_test * (neg_pro + 1)))
            samples = [s for s in samples if s not in ids]
            datalen = int(min(len(spl_ids), nb_test * neg_pro))
            neg_list.append([u] + samples[:datalen])
    return neg_list