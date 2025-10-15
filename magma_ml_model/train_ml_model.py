import sqlite3
import numpy as np
import pandas as pd
import implicit
from implicit.datasets.lastfm import get_lastfm
from implicit.nearest_neighbours import bm25_weight
from implicit.evaluation import precision_at_k,train_test_split,mean_average_precision_at_k,ndcg_at_k
from implicit.als import AlternatingLeastSquares
from implicit.cpu.lmf import LogisticMatrixFactorization
from sklearn.utils import resample
from numpy.random import default_rng
from scipy.sparse import csr_matrix
from tqdm import tqdm

# TODO:
## TRY TRAIN_TEST_SPLIT, check docs for COO vs CSR? 

# table names:
# lastfm_similars.db : similars_src similars_dest_tmp similars_dest
# lastfm_tags.db : tags tids tid_tag


if __name__ == "__main__":
    artists, users, artist_user_plays = get_lastfm()
    artist_user_plays = bm25_weight(artist_user_plays, K1=100, B=0.8)
    user_plays = artist_user_plays.T.tocsr()
    # I want to mask some percent of real plays to 0, then test to see if those are recommended. 

    rng = default_rng(1066)
    mask_fraction = 0.2

    for i in tqdm(range(100)):
        #TODO
        sample_rows = rng.choice(user_plays.shape[0], size=1000, replace=False)
        user_subset = user_plays[sample_rows, :]

        row_idx, col_idx = user_subset.nonzero()
        n_nonzero = len(row_idx)


        # Randomly choose which ones to mask
        n_mask = int(np.floor(mask_fraction * n_nonzero))
        mask_indices = rng.choice(n_nonzero, size=n_mask, replace=False)

        masked_rows = row_idx[mask_indices]
        masked_cols = col_idx[mask_indices] # These are all we actually care about (I think?) Maybe some metrics care about # of plays vs played at all
        masked_vals = user_subset[masked_rows, masked_cols].A1  # extract as 1D array

        # Construct test set sparse matrix
        test_matrix = csr_matrix((masked_vals, (masked_rows, masked_cols)), shape=user_subset.shape)


        # Copy original and zero out masked entries for train
        train_matrix = user_subset.copy()
        train_matrix[masked_rows, masked_cols] = 0
        train_matrix.eliminate_zeros()  # clean up

        #print(train_matrix.shape, test_matrix.shape)
        print(type(train_matrix),type(test_matrix))
        ## Modeling ##
        model = AlternatingLeastSquares(factors=64, regularization=0.05, alpha=2.0)
        model.fit(train_matrix)

        # How well did the model do on the train set?
        # TODO Fix
        #print('Train p@10: ',precision_at_k(model,train_matrix,train_matrix,K=10))
        #break
        # How well did the model do on the test set?

        print('Test p@10: ',precision_at_k(model,train_matrix,test_matrix,K=10))
        break


        #ids, scores = model.recommend(userid, user_plays[userid], N=10, filter_already_liked_items=False)
        #score_res = pd.DataFrame({"artist": artists[ids], "score": scores, "already_liked": np.in1d(ids, user_plays[userid].indices)})
        #print(score_res)