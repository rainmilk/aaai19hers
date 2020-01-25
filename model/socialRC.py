import numpy as np
from model.ranking import precisionAtK,recallAtK,nDCG,mrrAtK,avgPrecisionAtK
import csv

def cn_scores(G,from_node,to_node):
    common_neighbor = set(G.neighbors(from_node)).intersection(set(G.neighbors(to_node)))
    score = len(common_neighbor)
    return score

def score_connection(from_rep, to_rep, score_model):
    return score_model.predict([from_rep, to_rep])


def test_recommendation(user_rep, item_rep, score_model, test_path, neg_test_path):
    # read train_embedding and test_node


    test_data=np.loadtxt(test_path,dtype=np.int32)


    test_item_list=list(test_data[:,1])
    test_user_list=list(set(test_data[:,0]))


    prek=np.zeros(len(test_user_list))
    prek_inner=np.zeros(len(test_user_list))
    k=[5, 10, 15, 20,25,30,35,40,45, 50]

    ap_vector = np.zeros([len(test_user_list), len(k)])
    recall_vector = np.zeros([len(test_user_list), len(k)])
    ndcg_vector = np.zeros([len(test_user_list), len(k)])
    with open(neg_test_path, "r") as f:
        negative_nodes = list(csv.reader(f, quoting=csv.QUOTE_NONNUMERIC))

    for j, user in enumerate(test_user_list):
        neighbors_list = test_data[test_data[:, 0] == user, 1]
        non_neighbors_list = negative_nodes[j][1:]
        non_neighbors_list = np.asarray(non_neighbors_list)
        non_neighbors_list = non_neighbors_list.astype(np.int32)

        from_rep = user_rep[None, user - 1]
        to_rep = item_rep[neighbors_list - 1]
        pscore = score_connection(np.repeat(from_rep, to_rep.shape[0], axis=0), to_rep, score_model)

        to_rep = item_rep[non_neighbors_list - 1]
        nscore = score_connection(np.repeat(from_rep, to_rep.shape[0], axis=0), to_rep, score_model)

        ndcg_vector[j]=nDCG(pscore,nscore,k)
        recall_vector[j]= recallAtK(pscore,nscore,k)
        ap_vector[j]=avgPrecisionAtK(pscore,nscore,k)


    mean_ap= np.mean(ap_vector,axis=0)
    mean_recall = np.mean(recall_vector,axis=0)
    mean_ndcg=np.mean(ndcg_vector,axis=0)

    print("the MAP at")
    print(mean_ap, sep='\n')
    print("the mean recall at ")
    print(mean_recall, sep='\n')
    print("the mean ndcg at")
    print(mean_ndcg, sep='\n')


# data_name="book"
#
# train_path = "networkRS/%s_rating_train.txt"%data_name
# test_path = "networkRS/%s_rating_test.txt"%data_name
# neg_test_path = "networkRS/%s_rating_test_neg.csv"%data_name
# item_rep_path = "networkRS/%s_item_rep.txt"%data_name
# user_rep_path = "networkRS/%s_user_rep.txt"%data_name
# user_rep=np.loadtxt(user_rep_path)
# item_rep=np.loadtxt(item_rep_path)
# test_recommendation(user_rep, item_rep, train_path, test_path, neg_test_path)