import pandas as pd
import numpy as np
from hpfrec import HPF

import math
from sklearn.model_selection import train_test_split

"""

df_a = pd.read_csv("data/preprocess_data/preproc_100k_exposure_data.csv", header=0)
df_a = df_a.sort_values("user_id").reset_index(drop=True)
df_a = df_a.drop(["user_id"], axis=1)

df_y = pd.read_csv("data/preprocess_data/preproc_100k_rating_data.csv", header=0)
df_y = df_y.sort_values("user_id").reset_index(drop=True)
df_y = df_y.drop(["user_id"], axis=1)
"""
# Reading exposure and rating data from csv
a_train = pd.read_csv("data/new_pp/exposure_100k_train.csv", header=0)
a_train = a_train.sort_values("user_id").reset_index(drop=True)
a_train = a_train.drop(["user_id"], axis=1)

a_test = pd.read_csv("data/new_pp/exposure_100k_test.csv", header=0)
a_test = a_test.sort_values("user_id").reset_index(drop=True)
a_test = a_test.drop(["user_id"], axis=1)

y_train = pd.read_csv("data/new_pp/rating_100k_train.csv", header=0)
y_train = y_train.sort_values("user_id").reset_index(drop=True)
y_train = y_train.drop(["user_id"], axis=1)

y_test = pd.read_csv("data/new_pp/rating_100k_test.csv", header=0)
y_test = y_test.sort_values("user_id").reset_index(drop=True)
y_test = y_test.drop(["user_id"], axis=1)

test = pd.read_csv("data/new_pp/udata_train_100k.csv")
test["rating"] = np.ones(test.shape[0], dtype=int)
test.columns = ["UserId", "ItemId", "Count"]

np.random.seed(1)
counts_df = test
counts_df = counts_df.loc[~counts_df[['UserId', 'ItemId']].duplicated()].reset_index(drop=True)

# Full function call
recommender = HPF(
    k=30, a=0.3, a_prime=0.3, b_prime=1.0,
    c=0.3, c_prime=0.3, d_prime=1.0, ncores=-1,
    stop_crit='train-llk', check_every=10, stop_thr=1e-3,
    users_per_batch=None, items_per_batch=None, step_size=lambda x: 1 / np.sqrt(x + 2),
    maxiter=100, use_float=True, reindex=True, verbose=True,
    random_seed=None, allow_inconsistent_math=False, full_llk=False,
    alloc_full_phi=False, keep_data=True, save_folder=None,
    produce_dicts=True, keep_all_objs=True, sum_exp_trick=False
)

# Fitting the exposure model to the data
model = recommender.fit(counts_df)

a_hat_train = np.array([model.predict(user=[i for j in range(1, len(counts_df.ItemId.unique()) + 1)],
                                      item=counts_df["ItemId"].unique()) for i in range(1, counts_df.UserId.max() + 1)])
upper_bound = np.ones(a_hat_train.shape)

a_hat_train = np.minimum(a_hat_train, upper_bound)

# a_hat_train = np.zeros((943, 1658))

# train test split

# y_train, y_test, a_hat_train, a_hat_test, a_train, a_test = train_test_split(df_y, a_hat, df_a,
#                                                                             test_size=0.2, random_state=42)


def compute_loss_gradient(df_y, df_a, theta, beta, gamma, a_hat, lambda_t, lambda_b, lambda_g):
    """
# computes Gradient loss
    :param df_y:
    :param df_a:
    :param theta:
    :param beta:
    :param gamma:
    :param a_hat:
    :param lambda_t:
    :param lambda_b:
    :param lambda_g:
    """

    loss = np.linalg.norm(df_y.to_numpy() - np.multiply(np.dot(theta.T, beta), df_a.to_numpy()) -
                          np.multiply(gamma, a_hat), 2) + lambda_t * np.linalg.norm(theta, 2) + \
           lambda_b * np.linalg.norm(beta, 2) + lambda_g * np.linalg.norm(gamma, 2)

    grad_theta = (- 2 * np.dot(np.multiply(df_y.to_numpy() - np.multiply(np.dot(theta.T, beta), df_a.to_numpy()) -
                                           np.multiply(gamma, a_hat), df_a.to_numpy()),
                               beta.T)).T + 2 * lambda_t * theta

    grad_beta = (- 2 * np.dot((np.multiply(df_y.to_numpy() - np.multiply(np.dot(theta.T, beta), df_a.to_numpy()) -
                                           np.multiply(gamma, a_hat), df_a.to_numpy())).T,
                              theta.T)).T + 2 * lambda_b * beta

    grad_gamma = (- 2 * np.multiply(df_y.to_numpy() - np.multiply(np.dot(theta.T, beta), df_a.to_numpy()) -
                                    np.multiply(gamma, a_hat), a_hat)) + 2 * lambda_g * gamma

    return loss, grad_theta, grad_beta, grad_gamma


def train(y_train, y_test, a_hat_train, a_train, a_test, lr_t=1e-3, lr_b=1e-3, lr_g=1e-3,
          lambda_t=1e-5, lambda_b=1e-5, lambda_g=1e-5, max_epochs=1000):
    """
    Fit Theta (users attributes), Beta (items attributes), Gamma (importance of substitute deconfounder by user) by
    plain gradient descent

    :param a_test:
    :param a_train:
    :param y_train:
    :param y_test:
    :param a_hat_train:
    :param lr_t:
    :param lr_b:
    :param lr_g:
    :param lambda_t:
    :param lambda_b:
    :param lambda_g:
    :param max_epochs:
    :return:
    """
    # num_train = y_train.size
    num_users, num_movies = y_train.shape
    k = 30

    # num_iters_per_epoch = int(math.floor(1.0 * num_train / batch_size))

    # randomly initialize theta, beta and gamma

    theta = np.random.randn(k, num_users)
    beta = np.random.randn(k, num_movies)
    gamma = np.random.randn(num_users, 1)

    """ mini batch SGD uncomplete (problem with num iters per epoch and the learning in general)
    for epoch in range(max_epochs):
        perm_idx_row = np.random.permutation(num_row_train)
        perm_idx_col = np.random.permutation(num_col_train)
        # perform mini-batch SGD update
        for it in range(num_iters_per_epoch):
            idx_row = perm_idx_row[it * batch_size:(it + 1) * batch_size]
            idx_col = perm_idx_col[it * batch_size:(it + 1) * batch_size]
            batch_a = pd.DataFrame(a_train.to_numpy()[idx_row, idx_col])
            batch_y = pd.DataFrame(y_train.to_numpy()[idx_row, idx_col])
            batch_a_hat = a_hat_train[idx_row, idx_col]
            batch_theta = theta[:, idx_row]
            batch_beta = beta[:, idx_col]
            batch_gamma = gamma[idx_row, :]


            # evaluate loss and gradient
            loss, grad_theta, grad_beta, grad_gamma = compute_loss_gradient(
                batch_y, batch_a, batch_theta, batch_beta, batch_gamma, batch_a_hat, lambda_t, lambda_b, lambda_g)

            # update parameters
            theta[:, idx_row] += -lr_t * grad_theta
            beta[:, idx_col] += -lr_b * grad_beta
            for i in range(grad_gamma.shape[1]):
                gamma[idx_row, :] += (-lr_g * grad_gamma[:, i]).reshape((batch_size, 1))
            
            """
    loss_vector = [np.inf, np.inf]
    theta_vector = []
    beta_vector = []
    gamma_vector = []
    for epoch in range(max_epochs):
        loss, grad_theta, grad_beta, grad_gamma = compute_loss_gradient(
            y_train, a_train, theta, beta, gamma, a_hat_train, lambda_t, lambda_b, lambda_g)

        theta += -lr_t * grad_theta
        beta += -lr_b * grad_beta
        for i in range(grad_gamma.shape[1]):
            gamma += -lr_g * grad_gamma[:, i].reshape(gamma.shape)

        train_err_mse = acc(y_train, predict(theta, beta, gamma, a_hat_train), a_train)
        test_err_mse = acc(y_test, predict(theta, beta, gamma, a_hat_train), a_test)
        print('Epoch %4d: loss = %.2f, train_acc = %.4f, test_acc = %.4f' % (epoch, loss, train_err_mse, test_err_mse))

        loss_vector.append(loss)
        theta_vector.append(theta)
        beta_vector.append(beta)
        gamma_vector.append(gamma)
        if loss_vector[-1] > loss_vector[-3]:
            return theta_vector, beta_vector, gamma_vector

        theta_vector.clear()
        beta_vector.clear()
        gamma_vector.clear()
    return


def acc(y, y_hat, a):
    """
    Computes the average MSE per item (movie)

    :param a:
    :param y:
    :param y_hat:
    :return:
    """
    num_user = y.shape[0]
    return 1 / num_user * np.sum(1 / np.sum(a, axis=0) * np.sum(np.square(y.to_numpy() - np.multiply(y_hat, a)), axis=0),
                                 axis=0)


def predict(theta, beta, gamma, a_hat):
    """
    Computes the predicted rating

    :param theta:
    :param beta:
    :param gamma:
    :param a_hat:
    :return:
    """

    rating_array = np.dot(theta.T, beta) + np.multiply(gamma, a_hat)

    lower_bound = np.ones(rating_array.shape)

    upper_bound = 5 * np.ones(rating_array.shape)

    return np.minimum(np.maximum(rating_array, lower_bound), upper_bound)


train(y_train, y_test, a_hat_train, a_train, a_test, lr_t=1e-4, lr_b=1e-4, lr_g=1e-4,
      lambda_t=1e-2, lambda_b=1e-2, lambda_g=1e-2, max_epochs=1000)
