import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
import scipy.sparse as sp
from scipy.sparse.linalg import svds



def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))






#Reading users file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
 encoding='latin-1')
users_data=pd.DataFrame(users)



#Reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,
 encoding='latin-1')
items_data=pd.DataFrame(items)
items_data['movie id'] = items_data['movie id'].apply(pd.to_numeric)

#Reading ratings file:
r_cols = ['user_id', 'item_id', 'rating', 'timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,
 encoding='latin-1')
n_users = ratings.user_id.unique().shape[0]
n_items = ratings.item_id.unique().shape[0]
print ('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

'''
print(users.head())
print(ratings.head())
print(items.head())
'''

#make test and train datasets
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')
#print(ratings_base.shape)
#print(ratings_test.shape)


train_data=pd.DataFrame(ratings_base)
test_data=pd.DataFrame(ratings_test)




'''Memory-Based Collaborative Filtering'''
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


#Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')
item_pred_df=pd.DataFrame(item_prediction)
user_pred_df=pd.DataFrame(user_prediction)



print ('Memory-User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print ('Memory-Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))

'''
user_pred_df = user_pred_df.apply(lambda row: user_pred_df.columns[np.argsort(row)], axis=1)
print(user_pred_df.head())

item_pred_df = item_pred_df.apply(lambda row: item_pred_df.columns[np.argsort(row)], axis=1)
print(item_pred_df.head())
'''



'''Model-based Collaborative Filtering'''

sparsity=round(1.0-len(train_data)/float(n_users*n_items),3)
#print ('The sparsity level of MovieLens100K is ' +  str(sparsity*100) + '%')

#get SVD components from train matrix. Choose k
u, s, vt = svds(train_data_matrix, k = 20)
s_diag_matrix=np.diag(s)
pred = np.dot(np.dot(u, s_diag_matrix), vt)
pred_df=pd.DataFrame(pred)

print ('Model-User-based(SVD) CF RMSE: ' + str(rmse(pred, test_data_matrix)))
pred_df = pred_df.apply(lambda row: pred_df.columns[np.argsort(row)], axis=1)


