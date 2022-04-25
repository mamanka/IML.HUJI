from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
import pandas as pd


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

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
        self.classes_ = np.unique(y)
        self.mu_ = concetenated_df.groupby(by=0).mean().to_numpy()
        self.vars_ = concetenated_df.groupby(by=0).var(ddof = 1).to_numpy()
        self.pi_ = concetenated_df.groupby(by=0).size().to_numpy() / concetenated_df.shape[0]
        a=8
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
            prediction = np.hstack((prediction, predicted_sample))
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
        likelihood = None
        d = X.shape[1]
        for sample in X:
            sample_likelihood = np.array([])
            for k in self.pi_:
                k = int(k)
                cov_k = self.vars_[k]
                m = self.vars_[k]
                det_cov_k = np.linalg.det(cov_k)
                inverse_det_cov_k = np.linalg.inv(cov_k)
                const_term = 1 / np.sqrt(((np.pi * 2) ** d) * det_cov_k)
                exp_term = (-0.5 * np.transpose(sample - self.mu_[k]) @ inverse_det_cov_k
                            @ (sample - self.mu_[k]))
                addition = const_term * np.exp(-0.5 * exp_term) * self.pi_[k]
                sample_likelihood = np.hstack((sample_likelihood, addition))
            if likelihood is not None:
                likelihood = np.vstack((likelihood, sample_likelihood))
            else:
                likelihood = sample_likelihood
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
        from ...metrics import misclassification_error

        return misclassification_error(y, self.predict(X))
