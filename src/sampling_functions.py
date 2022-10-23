import numpy as np


def get_data_statistics(X):
    mu = np.mean(X, axis=0)
    sigma = np.cov(X.T)
    params_dict = {"X_mu": mu,
                   "X_sigma": sigma}
    return params_dict


def create_conditional_gauss(X, j, mu, sigma):
    """"
    This function learns the conditional distribution of X_j|X_-j

    :param X: A batch of b samples with d features.
    :param j: The index of the feature under test.
    :param mu, sigma: The mean and covariance of X.
    :return: The mean and covariance of the conditional distribution.
    To learn more about the implementation see: https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
    """
    a = np.delete(X, j, 1)
    mu_1 = np.array([mu[j]])
    mu_2 = np.delete(mu, j, 0)
    sigma_11 = sigma[j, j]
    sigma_12 = np.delete(sigma, j, 1)[j, :]
    sigma_21 = np.delete(sigma, j, 0)[:, j]
    sigma_22 = np.delete(np.delete(sigma, j, 0), j, 1)
    mu_bar_vec = []
    sigma12_22 = sigma_12 @ np.linalg.inv(sigma_22)
    sigma_bar = sigma_11 - sigma12_22 @ sigma_21
    for a_i in a:
        mu_bar = mu_1 + sigma12_22 @ (a_i - mu_2)
        mu_bar_vec.append(mu_bar)

    return mu_bar_vec, np.sqrt(sigma_bar)


def sample_from_gaussian(X, j, X_mu, X_sigma):
    """
    This function samples the dummy features for gaussian distribution.
    :param X: A batch of b samples with d features.
    :param j: The index of the feature under test.
    :param X_mu, X_sigma: The mean and covariance of X.
    :return: A copy of the batch X, with the dummy features in the j-th column.
    """
    mu_tilde, sigma_tilde = create_conditional_gauss(X, j, X_mu, X_sigma)
    n = X.shape[0]
    X_tilde = X.copy()
    Xj_tilde = np.random.normal(mu_tilde, sigma_tilde, (n, 1))
    X_tilde[:, j] = Xj_tilde.ravel()
    return X_tilde


def get_hiv_prob(X, clf):
    return clf.predict_proba(X)[:, 1]


def sample_hiv_data(X, j, clf):
    """
    This function sample the dummy features for binary original features.
    :param X: A batch of b samples with d features.
    :param j: The index of the feature under test.
    :param clf: A trained classifier, trained to predict the j-th feature given all other features.
    :return: A copy of the batch X, with the dummy features in the j-th column.
    """
    features_idx = np.arange(X.shape[1])
    train_features = np.setdiff1d(features_idx, j)
    prob = get_hiv_prob(X[:, train_features], clf)
    n = X.shape[0]
    X_tilde = X.copy()
    u_sample = np.random.uniform(0, 1, n)
    Xj_tilde = np.zeros((n,))
    Xj_tilde[u_sample < prob] = 1
    X_tilde[:, j] = Xj_tilde.ravel()
    return X_tilde
