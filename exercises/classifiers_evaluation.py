import numpy as np

import IMLearn.metrics
from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import custom
from math import atan2, pi
from utils import *

pio.templates.default = "simple_white"
import plotly.express as px

FILE_STATIC = "../datasets/{f}"
model_names = ['LDA', 'Bayes Naive Gaussian']
from math import atan2, pi
from IMLearn.metrics import accuracy


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
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


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

        # Plot figure of loss as function of fitting iteration


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, response = load_dataset(FILE_STATIC.format(f=f))
        lims = np.array([X.min(axis=0), X.max(axis=0)]).T + np.array([-.4, .4])
        symbols = np.array(['square', 'circle', 'triangle-up'])
        models = [LDA(),GaussianNaiveBayes()]
        accuracies = []
        from IMLearn.metrics import accuracy

        for model in models:
            model.fit(X,response)
            accuracies.append(accuracy(response, model.predict(X)))

        subplots = make_subplots(rows=1, cols=2, subplot_titles=[f"LDA- Accuracy: {accuracies[0]}",
                                                                 f"Gaussian Naive Bayes - Accuracy: {accuracies[1]}"],
                                 horizontal_spacing=0.01, vertical_spacing=.03)
        for i, m in enumerate(models):
            subplots.add_traces([decision_surface(m.fit(X, response).predict, lims[0], lims[1], showscale=False),
                                 go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                            marker=dict(color=m.predict(X), symbol= symbols[response], size=10,
                                                        colorscale=[custom[0], custom[-1], custom[1]],
                                                        line=dict(color='black', width=1)))],
                                rows =1 , cols = i+1)
            points = []
            ellipsis = []
            traces = []
            for i, mu in enumerate(m.mu_):
                points.append (mu)
                if isinstance(m, LDA):
                    cov = m.cov_
                ellipsis = get_ellipse(mu, cov)
                traces.append(ellipsis)

            centers = np.array(points)
            s = go.Scatter(x= centers[:,0], y= centers[:,1], mode = 'markers',
                                marker = dict (color = 'black', symbol = 'x', size = 15), showlegend = False)
            traces.append(s)
            subplots.add_traces(traces)

        subplots.update_layout(title_text = "Decision boundry of Models {f} Dataset".format(f=f), title_font_size = 20)
        subplots.show()

        # Fit models and predict over training set
        # prediction_array = np.array([])
        # prediction_array = lda.predict(X)

        # for i,m in enumerate ()
        # sample_scatter_plot = px.scatter(x=X ,y = response)
        # sample_scatter_plot.update_traces (mode = 'markers')
        # sample_scatter_plot.show()

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy

        # Add traces for data-points setting symbols and colors
    raise NotImplementedError()

        # Add `X` dots specifying fitted Gaussians' means
    raise NotImplementedError()

        # Add ellipses depicting the covariances of the fitted Gaussians
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

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black", showlegend= False)


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
