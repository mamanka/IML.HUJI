import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    city_tempratures_data = pd.read_csv(filename, parse_dates=['Date']).dropna()
    X = preprocess_data(city_tempratures_data)
    return X


def preprocess_data(df: pd.DataFrame):
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Year'] = df['Year'].astype(str)
    df = df.loc[(df['Temp'] > -70)]
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X = load_data("..\datasets\City_Temperature.csv")
    # Question 2 - Exploring data for specific country
    israel_df = X[X['Country'] == 'Israel']
    # Question 3 - Exploring differences between countries
    israel_temp_to_day = px.scatter(israel_df, x='DayOfYear', y='Temp',
                                    color='Year', title=
                                    "Temperature as a function of DOTY , color coded by Years")
    israel_temp_to_day.show()
    by_monthes_std = israel_df.groupby('Month').agg('std')
    israel_monthes_std = px.bar(by_monthes_std, y='Temp', title=
    'Standard deviation of temprature, grouped by month')
    israel_monthes_std.update_layout(xaxis_title='Months')
    israel_monthes_std.show()

    # Question 4 - Fitting model for different values of `k`
    group_by_countries_months = X.groupby(['Country', 'Month']).agg(std=("Temp", "std"),
                                                                    mean=("Temp", 'mean')).reset_index()
    line_fig = px.line(group_by_countries_months, x='Month', y='mean', color='Country', error_y='std')
    line_fig.update_layout(
        xaxis_title='Months')
    line_fig
    # Question 5 - Evaluating fitted model on different countries
    israel_x = israel_df['DayOfYear']
    israel_response = israel_df['Temp']
    x_train, y_train, x_test, y_test = split_train_test(israel_x, israel_response, 0.75)
    loss_list = []
    for k in range(1, 11):
        polyfit = PolynomialFitting(k)
        polyfit.fit(x_train.to_numpy(), y_train.to_numpy())
        loss_value = round(polyfit.loss(x_test.to_numpy(), y_test.to_numpy()), 2)
        print("loss for {k} degree polynomial is: {loss} \n".format(k=k, loss=loss_value))
        loss_list.append(loss_value)
    bar_plot = px.bar(x=range(1, 11), y=loss_list, color=loss_list
                      , color_continuous_scale=px.colors.sequential.Rainbow,
                      title="Loss as a function of polynomial degree")
    bar_plot.update_layout(xaxis_title="Polynomial Degree", yaxis_title = "Loss")
    bar_plot.show()

    # Question 5
    polyfit_k = PolynomialFitting(5)
    polyfit_k.fit(israel_x.to_numpy(), israel_response.to_numpy())
    no_israel_df = X.loc[X['Country'] != 'Israel']
    country_values = []
    for country in no_israel_df['Country'].unique():
        country_df = no_israel_df.loc[X['Country'] == country]
        cur_loss = polyfit_k.loss(country_df['DayOfYear'], country_df['Temp'])
        country_values.append(cur_loss)
    countries_bar = px.bar(x=no_israel_df['Country'].unique(), y=country_values, title="Mean Squared Loss "
                                                                                       "for Countries fitted on Israel K=4 Model",
                           labels={'country_value : MSE'}, pattern_shape=["*", "+", "*"])
    countries_bar.update_layout(xaxis_title="Country", yaxis_title = "MSE")
    countries_bar.show()
