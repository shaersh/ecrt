import numpy as np
import json
import os
from sklearn.linear_model import LassoCV, Lasso
from src.sampling_functions import get_data_statistics, sample_from_gaussian
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from src.utils import BettingFunction, TestStatistic, default, lasso_cv_online_learning
simplefilter("ignore", category=ConvergenceWarning)


class EcrtTester:
    """
    Conditional Testing with e-CRT
    """

    def __init__(self, batch_list=[2, 5, 10], n_init=50, K=20, j=0,
                 g_func=BettingFunction.sign, test_statistic=TestStatistic.mse, offline=False,
                 path="../results", load_name="", save_name="martingale_dict",
                 learn_conditional_distribution=get_data_statistics,
                 sampling_func=sample_from_gaussian, sampling_args={}
                 ):
        """
        :param batch_list: A list of batch sizes for the batch-ensemble.
        All batches must be divisors of the maximal one.
        :param n_init: The number of samples for the initial training.
        :param K: The de-randomization parameter. Number of dummy copies to be used for the wealth computation.
        :param j: The index of the tested feature. If you wish to test a different feature,
        you should create a new instance.
        :param g_func: The betting score function. Must be antisymmetric.
        :param test_statistic: The test statistic function, used to compare between the original and the dummy features.
        :param offline: Train offline LassoCV instead of online Lasso.
        :param path: Folder path to save and load martingales data.
        :param load_name: File name to load old martingales data.
        If given, the online updates start from the last saved wealth, and the last used point.
        If not given, the test starts from initial wealth 1.
        If you choose to load previous data, make sure to run with the same batch list, and on the same feature j.
        :param save_name: File name to save martingales data.
        :param learn_conditional_distribution: This function get X, the dataset,
        and returns learned arguments that are needed for the sampling of X_tilde.
        The returned arguments are saved to the dictionary sampling_args, and passed to the sampling function.
        :param sampling_func: This function gets X, j, and the additional arguments in sampling_args,
        and returns the dummy features X_tilde.
        :param sampling_args: A dictionary with all the non-learned arguments to pass to the sampling functions.
        """
        max_b = np.max(batch_list)
        for b in batch_list:
            assert max_b % b == 0
        self.batch_list = batch_list
        self.n_init = n_init
        self.K = K
        self.j = j
        self.g_func = g_func
        self.test_statistic = test_statistic
        self.offline = offline
        self.path = path
        self.load_name = load_name
        self.save_name = save_name
        self.integral_vector = np.linspace(0, 1, 1001, endpoint=False)[1:]
        self._initialize_martingales()
        self.model = None
        self.models_dict = {}
        self.sampling_args = sampling_args
        self.sampling_func = sampling_func
        self.learn_conditional_distribution = learn_conditional_distribution

    def _initialize_martingales(self):
        if self.load_name:
            with open(f"{self.path}/{self.load_name}.json") as json_file:
                self.martingale_dict = json.load(json_file)
            for b in self.batch_list:
                self.martingale_dict[b] = self.martingale_dict.pop(str(b))
        else:
            self.martingale_dict = {}
            for b in self.batch_list:
                self.martingale_dict[b] = {"St": 1,
                                           "St_v": np.ones((1000,)),
                                           "last_used_idx": self.n_init}

    def save_martingales(self):
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        with open(f"{self.path}/{self.save_name}.json", 'w') as json_file:
            json.dump(self.martingale_dict, json_file, default=default, indent=4)

    def _sample_dummy(self, X):
        X_tilde = self.sampling_func(X, self.j, **self.sampling_args)
        return X_tilde

    def _initialize_online_lasso(self, X, y):
        # In the online learning we use 20 Lasso models to choose the best eta hyper-parameter.
        # The grid of eta is set as follows:
        eps = 5e-3
        train_size = X.shape[0]
        Xy = np.dot(X.T, y)
        eta_max = np.sqrt(np.sum(Xy ** 2, axis=1)).max() / (train_size * 1)
        eta_vec = np.logspace(np.log10(eta_max * eps), np.log10(eta_max), num=20)
        # The models are trained using 80% of the train set. The hold out points are used to choose the best eta.
        models_dict = {}
        for eta in eta_vec:
            models_dict[eta] = Lasso(alpha=eta, warm_start=True)
        best_eta = lasso_cv_online_learning(X, y, models_dict, val_prcg=0.2)
        # We set the best eta to the main model, and fit the main model on the train set.
        model = Lasso(alpha=best_eta, warm_start=True)
        model.fit(X, y.ravel())
        model.max_iter = 50
        self.model = model
        self.models_dict = models_dict

    def _update_martingale(self, X, y, batch, test_idx, St_v):
        """
        The online update of one batch of samples (a single update).
        :param X: The full data set with d features.
        :param y: The full labels set.
        :param batch: The batch size b
        :param test_idx: The index of the first new point to be used in the update.
        The evaluation will be applied on the batch [test_idx : test_idx + batch].
        :param St_v: A vector with 1000 values, hold the martingales from the previous update.
        :return: A scalar St, the result of the integral over the 1000 martingales, with uniform density.
        :return A vector St_v, with the updated martingales.
        """
        wealth = 0
        # Compute the MSE (or any other test statistic) of the original features once.
        y_predict = self.model.predict(X[test_idx:test_idx + batch, :])
        q = self.test_statistic(y_predict.ravel(), y[test_idx:test_idx + batch].ravel())
        # For K iterations, sample the dummy features, compute the dummy MSE
        # and update the wealth using the betting score function, g_func.
        for k in range(self.K):
            X_tilde = self._sample_dummy(X[test_idx:test_idx + batch, :])
            y_tilde = self.model.predict(X_tilde)
            q_tilde = self.test_statistic(y_tilde.ravel(), y[test_idx:test_idx + batch].ravel())
            wealth += self.g_func(q, q_tilde)
        # Update the martingales using the average betting score.
        St_v = St_v * (1 + self.integral_vector * wealth/self.K)
        St = np.mean(St_v)  # integral with uniform density.
        return St, St_v

    def run(self, X, y, start_idx=None, alpha=0.05):
        """
        :param X: The data matrix with size (n, d).
        Note that even if you run using old martingales data, and start_idx is not None,
        you should provide the old data. The data will not be used to update the martingales,
        but will be used to train the learning model.
        :param y: A vector of labels with size (n, 1)
        :param start_idx: The first sample that will be used to update the martingales.
        All points before it will be used for training only. If None, the first sample will be n_init.
        :param alpha: The target level. The null will be rejected when the martingale will reach 1/alpha.
        :return: Whether the null is rejected or not, i.e., whether the tested feature is important or not.
        """
        rejected = False
        if start_idx is None:
            start_idx = self.n_init
        # Train the model on the available data points, that are not used for the martingales update.
        # If you wish to use a different predictive model, please replace the Lasso model here.
        if self.offline:
            self.model = LassoCV(max_iter=10000, eps=5e-3).fit(X[:start_idx, :], y[:start_idx].ravel())
        else:
            self._initialize_online_lasso(X[:start_idx, :], y[:start_idx])
        n = X.shape[0]
        # Run the sequential updates
        for new_points in np.arange(start_idx, n, 1):
            b_last_used_list = []
            st_list = []
            update = False
            # Ensemble over batches
            for b in self.batch_list:
                # Once we use a point to update the martingale, we can not use it again for updates, only for training.
                # In this condition we validate that we train the model using old samples, and that we update the
                # martingale using new unseen points.
                if new_points == self.martingale_dict[b]["last_used_idx"]:
                    # This step is skipped if we already updated the model in a previous iteration,
                    # for a smaller batch size.
                    if not update:
                        # Learn the variables that are needed for the dummies generation, if there are any.
                        new_sampling_args = self.learn_conditional_distribution(X[:new_points, :])
                        self.sampling_args = {**self.sampling_args, **new_sampling_args}
                        # Train the model on the valid training data.
                        if not self.offline:
                            best_eta = lasso_cv_online_learning(X[:new_points, :], y[:new_points], self.models_dict,
                                                                  val_prcg=0.2)
                            self.model.alpha = best_eta
                        self.model.fit(X[:new_points, :], y[:new_points].ravel())
                    update = True
                    # Update the martingale using the new batch of samples.
                    self.martingale_dict[b]["St"], self.martingale_dict[b]["St_v"] = self._update_martingale(
                        X, y, b, new_points, self.martingale_dict[b]["St_v"])
                    self.martingale_dict[b]["last_used_idx"] = min(new_points + b, n)
                b_last_used_list.append(self.martingale_dict[b]["last_used_idx"])
                st_list.append(self.martingale_dict[b]["St"])
            # If all the martingales used the same number of points, update the ensemble.
            if len(set(b_last_used_list)) == 1 and update:
                st_vec = np.array(st_list)
                St = st_vec.mean()
                # If the ensemble martingale passes the test level, we can safely reject the null.
                if St > 1/alpha:
                    rejected = True
                    break
        return rejected

