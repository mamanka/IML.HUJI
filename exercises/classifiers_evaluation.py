import IMLearn.metrics
from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from utils import custom
from math import atan2, pi
from utils import *

pio.templates.default = "simple_white"
import plotly.express as px

FILE_STATIC = "../datasets/{f}"
model_names = ['LDA', 'Bayes Naive Gaussian']
symbols = ['circle', 'x', 'diamond']


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    dataset = np.load(filename)
    x_array = dataset[:, 0:2]
    y = dataset[:, 2]
    return x_array, y


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, response = load_dataset(FILE_STATIC.format(f=f))
        # Fit Perceptron and record loss in each fit iteration
        losses = []
        perceptron = Perceptron()
        perceptron.fit(X, response)
        scatter_plot = px.line(x=range(len(perceptron.loss_evaluation)), y=perceptron.loss_evaluation,
                               title="Loss as function of Perceptron iterations for {seperability}".format(
                                   seperability=n))
        scatter_plot.update_layout(xaxis_title="Iterations", yaxis_title="Loss")
        # scatter_plot.show()


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, response = load_dataset(FILE_STATIC.format(f=f))
        lda = LDA()
        lda.fit(X, response)

        # Fit models and predict over training set
        prediction_array = np.array([])
        prediction_array = lda.predict(X)
        subplots = make_subplots(rows=1, cols=2, subplot_titles=[rf"$\textbf{{{m}}}$" for m in model_names],
                                 horizontal_spacing=0.01, vertical_spacing=.03)
        scatter = go.Figure(data=[go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                             marker=dict(color=response, symbol=symbols, size=20,
                                                         colorscale=[custom[0], custom[-1], custom[1]],
                                                         line=dict(color='black', width=1)))])
        scatter_x0_lim = scatter.get_xlim
        scatter.show()
        # for i,m in enumerate ()
        # sample_scatter_plot = px.scatter(x=X ,y = response)
        # sample_scatter_plot.update_traces (mode = 'markers')
        # sample_scatter_plot.show()

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy
        raise NotImplementedError()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
