from typing import NoReturn

import IMLearn.metrics.loss_functions
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
import pandas as pd


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        x_df = pd.DataFrame(X)
        y_df = pd.DataFrame(y)
        concetenated_df = pd.concat([y_df, x_df], ignore_index=True, axis=1)
        self.mu_ = concetenated_df.groupby(by = 0).mean().to_numpy()
        self.classes_ = np.unique(y)
        cov = np.zeros(shape=(X.shape[1], X.shape[1]))
        for i, row in enumerate(X):
            mu = self.mu_[self.classes_ == y[i]]
            mat = row - mu
            res = np.matmul(np.transpose(mat), mat)
            cov += res
        self.cov_ = cov / (len(y) - len(self.classes_))
        self._cov_inv = np.linalg.inv(self.cov_)
        self.pi_ = concetenated_df.groupby(by = 0).size(). to_numpy() /concetenated_df.shape[0]

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        likelihood = self.likelihood(X)
        prediction = np.array([])
        for ll in likelihood:
            predicted_sample = np.argmax(ll)
            prediction = np.hstack((prediction,predicted_sample))
        return prediction
    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        det_cov = np.linalg.det(self.cov_)
        d = X.shape[1]
        const_term = 1/np.sqrt(((np.pi * 2) ** d) * det_cov)
        likelihood = None
        for sample in X:
            sample_lilkelihood = np.array([])
            for k in self.classes_:
                exp_term = (np.transpose(sample - self.mu_[int(k)])@self._cov_inv @
                            (sample - self.mu_[int(k)]))
                sample_lilkelihood = np.hstack((sample_lilkelihood, const_term *
                                                np.exp(-0.5 * exp_term) * self.pi_[int(k)]))
            if likelihood is not None:
                likelihood = np.vstack((likelihood, sample_lilkelihood))
            else:
                likelihood = sample_lilkelihood
        return likelihood

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return (IMLearn.metrics.loss_functions.misclassification_error(y, self.predict(X)))

