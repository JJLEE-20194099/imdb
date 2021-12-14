from re import A
import pandas as pd;
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np;

n_users = 1415

print ('Number of users: ', n_users)

ratings_base = pd.read_csv('./data/datasets/rating/kfold/u1.base.csv', sep=',', encoding='latin-1')
ratings_test = pd.read_csv('./data/datasets/rating/kfold/u1.test.csv', sep=',', encoding='latin-1')

# print(type(ratings_base))
ratings_train_arr = ratings_base.values[1:, :]
# print(type(ratings_train_arr))
ratings_test_arr = ratings_test.values[1:, :]


print('ratings_train_shape: ', ratings_train_arr.shape)
print('ratings_test_shape: ', ratings_test_arr.shape)

movies = pd.read_csv('./data/datasets/movie/ml_details.csv', sep=',', encoding='latin-1')

no_movies = movies.shape[0]
print('No movie themes: ', no_movies)

X_train = movies[["Biography","Music","History","Thriller","Fantasy","Sport","Animation","Game-Show","Horror","Musical","Family","Mystery","Talk-Show","Documentary","Sci-Fi","Film-Noir","Short","Western","Romance","Drama","Reality-TV","Crime","Comedy","Adventure","News","War","Action",]].values[1:, :]
print(X_train)

transformer = TfidfTransformer(smooth_idf=True, norm='l2')
tfidf = transformer.fit_transform(X_train).toarray()

print("tfidf: ", tfidf.shape)

def get_movies_rated_by_user(utility_matrix, user_id):
    user_id_list = utility_matrix[:, -1]
    row_ids = np.where(user_id_list == user_id)[0]
    movie_id_list = utility_matrix[row_ids, 0]
    rating_list = utility_matrix[row_ids, 3]
    return (movie_id_list, rating_list)

from sklearn.linear_model import Ridge
from sklearn import linear_model

no_movie_theme = tfidf.shape[1]
w = np.zeros((no_movie_theme, n_users))
b = np.zeros((1, n_users))

for i in range(n_users):
    movie_id_list, rating_list = get_movies_rated_by_user(ratings_train_arr, i)
    ridge = Ridge(alpha=0.01, fit_intercept=True)
    print(movie_id_list)
    print(rating_list)
    print(tfdif_by_user)
    tfdif_by_user = tfidf[movie_id_list, :]
    ridge.fit(tfdif_by_user, rating_list)

    w[:, i] = ridge.coef_
    b[0, i] = ridge.intercept_

Y = tfidf.dot(w)  + b;
n = 4
np.set_printoptions(precision=2)
movie_id_list, rating_list = get_movies_rated_by_user(ratings_test_arr, n)

print('True Rating List: ', rating_list)
print('Predict Rating List: ', Y[movie_id_list, n])





