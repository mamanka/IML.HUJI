from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import os

pio.templates.default = "simple_white"


# TODO:1. questions 6,7,8 2. write to file, and pick 2 different correlations. 3. adjust data 4. polynomial part.

def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    house_prices_data = pd.read_csv(filename)
    X, y = preprocess_data(house_prices_data)
    return X, y


def preprocess_data(df: pd.DataFrame):
    # TODO : 1. abs value between renovation year and house built year 2. group by decade of year.
    # clean dirty info about prices and bedrooms

    # drop all rows with price < 0
    df = df.loc[(df['price'] > 0) & (df['price'] != np.nan)]
    # drop all rows with n.bedrooms >= 25.
    df = df.loc[df['bedrooms'] < 20]
    # drop samples  with no bedroom no bedrooms and no bathrooms
    df = df.loc[(df['bedrooms'] > 0) & (df['bathrooms'] > 0)]
    # drop samples with subzero sqft types
    df = df.loc[(df['sqft_living'] > 0)]
    df = df.loc[(df['sqft_lot'] > 0)]
    df = df.loc[(df['sqft_above'] > 0)]
    # drop NA rows, due to small numbers
    df = df.dropna()
    # drop id column
    ###################################################3
    # create new column, if renovated or built after 1990
    df['is_newly_renovated_or_built'] = 0
    df.loc[(df['yr_renovated'] >= 1990), 'is_newly_renovated_or_built'] = 1
    # dummies for years built  by decade
    df = df.loc[(df['yr_built'] > 0)]  # remove year built - zero

    df['yr_built'] = df['yr_built'] // 10 * 10
    df = pd.get_dummies(df, prefix='yr_built_', columns=['yr_built'])
    # create dummies for zipcodes
    df = pd.get_dummies(df, prefix='zipcode_', columns=['zipcode'])
    # check if number of bathrooms between 0.5 and 1, if True, create for them new
    df['no_bathroom'] = 0
    df.loc[(df['bathrooms'] < 1), 'no_bathroom'] = -1
    # ratio between house living space and neighbors. as ratio is higher, house price rises.

    df['house_vs_neighbors'] = (df['sqft_living'] / df['sqft_living15']) ** 0.5

    df['area_value'] = 0
    df.loc[((df['lat'] >= 47.6) & (df['lat'] <= 47.71) & (df['long'] >= -122.45) & (
            df['long'] <= -121.92)), 'area_value'] = 1
    response = df['price']
    df.drop(['price'], axis=1, inplace=True)
    df.drop(['date'], axis=1, inplace=True)
    df.drop(['lat', 'long'], axis=1, inplace=True)
    df.drop(['id'], axis=1, inplace=True)
    # df.drop(['zipcode'], axis = 1, inplace = True)
    return df, response


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    if not os.path.exists("images"):
        os.mkdir("images")
    std_y = np.std(y)
    for feature in X:
        cov_feature = np.cov(X[feature], y)[0][1]
        std_feature = np.std(X[feature])
        p_corr = cov_feature / (std_feature * std_y)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=X[feature],
            y=y,
            name="response as a function of {feature}.\n Corr = {correlation}".format(feature=feature,
                                                                                      correlation=p_corr),
            mode="markers",
            marker=go.scatter.Marker(
                opacity=0.6,
                colorscale="Viridis"
            )
        ))
        fig.update_layout(
            title_text="response as a function of {feature} \n Corr = {correlation}".format(feature=feature,
                                                                                            correlation=p_corr),
            xaxis_title="{feature}".format(feature=feature), yaxis_title="Response")
        # fig.write_html("..\submissions\{feature}corr.html".format(feature = feature))


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, response = load_data("..\datasets\house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, response, "")

    # Question 3 - Split samples into training- and testing sets.
    x_train, y_train, x_test, y_test = split_train_test(X, response, 0.75)
    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    # Then plot average loss as function of training size with error ribbon of size (
    #   4) Store average and variance of loss over test setmean-2*std, mean+2*std)
    regression = LinearRegression()
    mean_list = []
    plus_std = []
    minus_std = []
    x_axis = list(range(10, 101))
    for p in range(10, 101):
        inner_loss = []
        for j in range(10):
            x_sample = x_train.sample(frac=p / 100)
            y_hat = y_train.loc[x_sample.index]
            regression.fit(x_sample.to_numpy(), y_hat.to_numpy())
            inner_loss.append(regression.loss(x_test.to_numpy(), y_test.to_numpy()))
        loss = np.array(inner_loss)
        mean_loss = np.mean(loss)
        mean_list.append(mean_loss)
        var_loss = np.std(loss)
        plus_std.append(mean_loss + 2 * var_loss)
        minus_std.append(mean_loss - 2 * var_loss)

    data = [
        go.Scatter(x=x_axis, y=mean_list, name="Loss", mode='markers+lines', marker=dict(color='blue', opacity=0.7)),
        go.Scatter(x=x_axis, y=plus_std, name="post_std", fill=None, mode="lines", line=dict(color="indigo")),
        go.Scatter(x=x_axis, y=minus_std, name="neg_std", fill='tonexty', mode="lines", line=dict(color="indigo"))]
    fig = go.Figure(data=data, layout=go.Layout(title="MSE as a Function of Sample Percentage",
                                                xaxis={"title": "Sample Percentage"},
                                                yaxis={"title": "Mean Square Loss"}))
    fig.show()
