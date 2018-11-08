import numpy as np



def mrrAtK( pScore, nScore, atK=None ):
    T = len(pScore)

    if atK is None: atK = T

    nScore = np.asarray(nScore)
    mrr = np.zeros_like(atK, dtype=float)
    for p in pScore:
        rank = np.sum(nScore > p) + 1
        mrr += (rank <= atK) * (1 / rank)

    return mrr / T



def precisionAtK( pScore, nScore, atK ):
    """
    pScore: scores for positive instances
    nScore: scores for negative instances
    atK:    top K
    """

    T = len(pScore)
    N = len(nScore)
    R = T + N

    if atK is None: atK = R

    IX = (-np.concatenate([pScore, nScore]).flatten()).argsort()
    rankvec = np.concatenate([np.ones(T), np.zeros(N)])
    rankvec = np.cumsum(rankvec[IX])

    topK = np.minimum(atK, R)
    hits = rankvec[topK - 1]
    return hits / topK



def recallAtK( pScore, nScore, atK ):
    """
    pScore: scores for positive instances
    nScore: scores for negative instances
    atK:    top K
    """

    T = len(pScore)
    N = len(nScore)
    R = T + N

    if atK is None: atK = R

    IX = (-np.concatenate([pScore, nScore]).flatten()).argsort()
    rankvec = np.concatenate([np.ones(T), np.zeros(N)])
    rankvec = np.cumsum(rankvec[IX])

    topK = np.minimum(atK, R)
    hits = rankvec[topK  - 1]
    return hits / T



def nDCG( pScore, nScore, atK ):
    """
    pScore: scores for positive instances
    nScore: scores for negative instances
    atK:    top K
    """
    T = len(pScore)
    N = len(nScore)
    R = T + N

    if atK is None: atK = R

    den = np.log2( np.arange(2, R+2) )

    rankvec = np.concatenate([np.ones(T), np.zeros(N)])
    IDCG = np.cumsum( (np.power(2, rankvec) - 1) / den)

    IX = (-np.concatenate([pScore, nScore]).flatten()).argsort()
    rankvec = rankvec[IX]
    DCG = np.cumsum( (np.power(2, rankvec) - 1) / den )

    topK = np.minimum(atK, R) - 1
    return DCG[topK] / IDCG[topK]



def avgPrecisionAtK( pScore, nScore, atK ):
    """
    pScore: scores for positive instances
    nScore: scores for negative instances
    atK:    top K
    """

    T = len(pScore)
    N = len(nScore)
    R = T + N

    if atK is None: atK = R

    IX = (-np.concatenate([pScore, nScore]).flatten()).argsort()
    rankvec = np.concatenate([np.ones(T), np.zeros(N)])
    rankvec = rankvec[IX]

    topK = np.minimum(atK, R)
    avp = (np.cumsum(rankvec) * rankvec) / np.arange(1, R+1)
    avp = np.cumsum(avp)
    return avp[topK - 1] / np.minimum(topK, T)

# Unit Testing
pScore = [1,2,8,9,10]
nScore = [6,7,3,4,5]
atK = [2, 8]

# print(mrrAtK(pScore, nScore, atK))
# print(precisionAtK(pScore, nScore, atK))
# print(avgPrecisionAtK(pScore, nScore, atK))
# print(recallAtK(pScore, nScore, atK))
# print(nDCG(pScore, nScore, atK))