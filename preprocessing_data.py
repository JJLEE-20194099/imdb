import pandas as pd
import os
from src.text_csv_utils import write_csv_file

rating_data = pd.read_csv('./data/datasets/rating/data.csv', sep=',', encoding='utf-8')
rating_data.drop_duplicates(keep='first')
rating_data.to_csv('./data/datasets/rating/data.csv', sep=',', encoding='utf-8', index=False)


movie_ids = open('./data/datasets/movie/ids.txt', 'r')
movie_ids = movie_ids.readlines()
movie_ids = set(movie_ids)
movie_ids = [[movie_id.strip()] for movie_id in movie_ids]

movie_id_path = os.path.join('data/datasets/movie/', 'ids.txt')
write_csv_file(movie_ids, movie_id_path, 'w')

fake_user_ids = open('./data/datasets/user/fake_ids.txt', 'r')
user_ids = fake_user_ids.readlines()
user_ids = set(user_ids)
user_ids = [[user_id.strip()] for user_id in user_ids]

user_id_path = os.path.join('data/datasets/user/', 'ids.txt')
write_csv_file(user_ids, user_id_path, 'w')