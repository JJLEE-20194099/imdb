from os import sep
import pandas as pd;
import numpy as np;
from scipy import sparse;
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class collaborative_filering:
    def __init__(self, Y, k_neighbors, distance_func = cosine_similarity, mode = 1):
        self.mode = mode;
        if self.mode == 1:
            self.Y = Y
        else: 
            self.Y = Y[:, [1, 0, 2]]
        
        self.k_neighbors = k_neighbors
        self.distance_func = distance_func
        self.Y_utility = None
        # self.no_users = int(np.max(self.Y[:, 0])) + 1
        # self.no_movies = int(np.max(self.Y[:, 1])) + 1
        self.no_users = 9385
        self.no_movies = 1415

    def insert(self, data):
        self.Y = np.concatenate((self.Y, data), axis = 0)

    def normalize_data(self):
        users = self.Y[:, 0]
        self.Y_utility = self.Y.copy()
        self.mean_user = np.zeros((self.no_users, ))
        for i in range(self.no_users):
            user_id_list = np.where(users == i)[0].astype(np.int32)
            item_id_list = self.Y[user_id_list, 1]
            rating_list = self.Y[user_id_list, 2]
            if (len(rating_list) == 0):
                m = 0
            else :
                m = np.mean(rating_list)
                if np.isnan(m):
                    m = 0
            self.mean_user[i] = m
            self.Y_utility[user_id_list, 2] = rating_list - self.mean_user[i]
            
        self.Y_utility_sparse = sparse.coo_matrix((self.Y_utility[:, 2], (self.Y_utility[:, 1], self.Y_utility[:, 0])),  (self.no_movies, self.no_users))
        self.Y_utility_sparse = self.Y_utility_sparse.tocsr()
    

    def cal_similarity(self):
        self.similarity_matrix = self.distance_func(self.Y_utility_sparse.T, self.Y_utility_sparse.T)
       
    
    def refresh(self):
        self.normalize_data()
        self.cal_similarity()
    
    def fit(self):
        self.refresh()
    

    def predict_utils(self, user, i):
        user = int(user)
        i = int(i)
        user_id_list = np.where(self.Y[:, 1] == i)[0].astype(np.int32)
        user_id_list = (self.Y[user_id_list, 0]).astype(np.int32)
        similar = self.similarity_matrix[user, user_id_list]

        k_user_id_nearest = np.argsort(similar)[-self.k_neighbors: ]

        k_simlilar_nearest = similar[k_user_id_nearest]
        
        r = self.Y_utility_sparse[i, user_id_list[k_user_id_nearest]]
        
        
        return (r * k_simlilar_nearest)[0] / (np.abs(k_simlilar_nearest).sum() + 1e-8) + self.mean_user[user]

 

    def predict(self, user, i):
        
        if self.mode:
            return self.predict_utils(user, i)
        return self.predict_utils(i, user)
    

    def suggest(self, user):

        row_id_list = np.where(self.Y[:, 0] == user)[0]
        movie_list = self.Y[row_id_list, 1].tolist()

        res = []

        for i in range(self.no_movies):
            if i not in movie_list:
                rating = self.predict_utils(user, i)
                
                if rating > self.mean_user[user]:
                    res.append(i)
        
        return res
    
    def print(self):
        for i in range(self.no_users):
            res = self.suggest(i)
            if self.mode:
                print ('Recommend item(s):', res, 'to user', i)
            else: 
                print ('Recommend item', i, 'to user(s) : ', res)

# develop test

# cols = ['user_id', 'item_id', 'rating']
# ratings = pd.read_csv('./data/ml-100k/me_test.dat', sep=' ', names=cols, encoding='latin-1')
# Y = ratings.values

# CF_model = collaborative_filering(Y, k_neighbors = 2, mode= 1)
# CF_model.fit()

# CF_model.print()

# real test

neighbors = [1, 10, 20, 30, 50, 100]
for neighbor in neighbors:
    res = 0
    for i in range(1, 2):
        train_path = f'./data/datasets/rating/kfold/u{i}.base.csv'
        test_path = f'./data/datasets/rating/kfold/u{i}.test.csv'
        rating_train = pd.read_csv(train_path, sep=',', encoding='latin-1')[['user index', 'movie index', 'rating']].values
        rating_test = pd.read_csv(test_path, sep=',', encoding='latin-1')[['user index', 'movie index', 'rating']].values


        CF_model = collaborative_filering(rating_train, k_neighbors=neighbor, mode=0)
        CF_model.fit()
        print(CF_model.similarity_matrix.shape)
        print(CF_model.similarity_matrix)

        no_tests = rating_test.shape[0]
        square_error = 0

        Y_predict = []
        for i in range(no_tests):
            predict = CF_model.predict(rating_test[i, 0], rating_test[i, 1])
            Y_predict.append(predict)
            square_error += (predict - rating_test[i, 2]) ** 2

        RMSE = np.sqrt(square_error/(no_tests))

        print(RMSE)
        res = res + RMSE
        # print(CF_model.suggest(1368)[:10])

        # X = [i for i in range(no_tests)][:100]

        # Y_true = rating_test[:, 2][:100]

        # print(Y_predict[:100])
        # print(Y_true)
        # plt.plot(X, Y_predict[:100], color='green')
        # plt.scatter(X, Y_true)
        # plt.show()
    print('neighbor: ', neighbor, '-', 'RMSE: ', res / 5)





