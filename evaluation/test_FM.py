from scipy.sparse import coo_matrix,csr_matrix,lil_matrix,csc_matrix
from sklearn.preprocessing import normalize
import numpy as np
import csv
import random
from fastFM import bpr
from fastFM import utils
from model.ranking import precisionAtK,recallAtK,nDCG,mrrAtK,avgPrecisionAtK
from scipy.sparse import lil_matrix, hstack, spdiags

data_name='book'
user_net_path='../datasets/%s/%s_userNet.txt'%(data_name,data_name)
ui_net_path ='../datasets/%s/%s_rating.txt'%(data_name,data_name)
item_path = '../datasets/%s/%s_itemNet.txt'%(data_name,data_name)


train_path = "../datasets/%s/%s_rating_train.txt"%(data_name,data_name)
test_path = "../datasets/%s/%s_rating_test.txt"%(data_name,data_name)
neg_test_path = "../datasets/%s/%s_rating_test_neg.csv"%(data_name,data_name)

rating_train= np.loadtxt(train_path,dtype=np.int32) - 1
rating_test= np.loadtxt(test_path,dtype=np.int32) - 1

user_edge= np.loadtxt(user_net_path,dtype=np.int32) - 1
item_edge=np.loadtxt(item_path,dtype=np.int32) - 1

user_num = np.max(user_edge) + 1
item_num = np.max(item_edge) + 1

with open(neg_test_path, "r") as f:
    test_neg = list(csv.reader(f, quoting=csv.QUOTE_NONNUMERIC))

data=np.ones(len(user_edge))
user_data= coo_matrix((data, (user_edge[:,0], user_edge[:,1])), shape=(user_num, user_num))
user_data = hstack([spdiags([1]*user_num, 0, user_num, user_num), normalize(user_data, norm='l1', axis=1)]).tolil()

data= np.ones(len(item_edge))
item_data= coo_matrix((data, (item_edge[:,0], item_edge[:,1])), shape=(item_num, item_num))
item_data = hstack([spdiags([1]*item_num, 0, item_num, item_num), normalize(item_data, norm='l1', axis=1)]).tolil()

train_rating_num=len(rating_train)
train_item_list=np.unique(rating_train[:,1])

print('Start constructing training set')
sap = 10
X_User = lil_matrix(((sap+1)*train_rating_num, 2*user_num), dtype=np.float32)
X_Item = lil_matrix(((sap+1)*train_rating_num, 2*item_num), dtype=np.float32)
Y_train = np.zeros([sap*train_rating_num,2], dtype=np.int32)
for i,rating in enumerate(rating_train):
    begin = (sap+1)*i
    user_idx = rating[0]
    X_User[begin:begin+1+sap] = user_data[user_idx]

    item_idx = rating[1]
    X_Item[begin] = item_data[item_idx]
    neg_itemId = random.choices(train_item_list, k=sap)
    X_Item[begin+1:begin+1+sap] = item_data[neg_itemId]

    begin = sap*i
    Y_train[begin:begin + sap, 0] = item_idx
    Y_train[begin:begin + sap, 1] = neg_itemId

    if i%100 == 0:
        print('Constructing: %d/%d'%(i+1, train_rating_num))

X_train = hstack([X_User, X_Item]).tocsc()


# Build Model
print('Start training')
fm = bpr.FMRecommender(n_iter=5000000, init_stdev=0.1, rank=20,
                       l2_reg_w=0.0, l2_reg_V=0.0, l2_reg=0, step_size=0.1)
fm.fit(X_train, Y_train)

print("Fitting is done")
test_user_list=list(set(rating_test[:,0]))

k=[5, 10, 15, 20,25,30,35,40,45, 50]

ap_vector = np.zeros([len(test_user_list), len(k)])
recall_vector = np.zeros([len(test_user_list), len(k)])
ndcg_vector = np.zeros([len(test_user_list), len(k)])
for i, user in enumerate(test_user_list):

    pos_list = rating_test[rating_test[:, 0] == user, 1]
    neg_list = np.asarray(test_neg[i][1:]) - 1
    nb_pos = len(pos_list)
    nb_neg = len(neg_list)

    X_TS = lil_matrix((nb_pos + nb_neg, 2*(user_num + item_num)), dtype=np.float32)
    X_TS[:, :2*user_num] = user_data[user_idx]

    item_idx = pos_list
    X_TS[:nb_pos, 2*user_num:] = item_data[item_idx]

    item_idx = neg_list
    X_TS[nb_pos:, 2*user_num:] = item_data[item_idx]

    score = fm.predict_aspect_scores(X_TS.tocsc())
    pscore = score[:nb_pos]
    nscore = score[nb_pos:]

    ndcg_vector[i] = nDCG(pscore, nscore, k)
    recall_vector[i] = recallAtK(pscore, nscore, k)
    ap_vector[i] = avgPrecisionAtK(pscore, nscore, k)

mean_ap= np.mean(ap_vector,axis=0)
mean_recall = np.mean(recall_vector,axis=0)
mean_ndcg=np.mean(ndcg_vector,axis=0)

print("the MAP at")
print(mean_ap, sep='\n')
print("the mean recall at ")
print(mean_recall, sep='\n')
print("the mean ndcg at")
print(mean_ndcg, sep='\n')
