from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express
import pandas as pd
pio.templates.default = "simple_white"

def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    X = np.random.normal(10,1, size= 1000)
    uni = UnivariateGaussian()
    uni.fit(X)
    tup = (uni.mu_, uni.var_)
    print (tup)

    # Question 2 - Empirically showing sample mean is consistent
    absolute_distance = []
    ms = np.linspace(10, 1000, 100).astype(int)
    mu = 10
    for m in ms:
       uni.fit(X[:m])
       mean = uni.mu_
       absolute_distance.append(np.abs(mean - mu))
    df_difference_for_sample_size = pd.DataFrame({"number of samples" : ms, "r$\hat\mu - \mu$": absolute_distance})
    plotly.express.bar(df_difference_for_sample_size, x ="number of samples", y= "r$\hat\mu - \mu$",
                       title = "(Question 2) Absolute Value Between Estimated Mean and True value As Function Of Number Of Samples").show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = uni.pdf(X)
    df_pdfs_for_samples = pd.DataFrame({"Samples in increasing order" : X, "PDF'S": pdfs})
    plotly.express.scatter(df_pdfs_for_samples, x ="Samples in increasing order", y= "PDF'S",
                       title = r"$\text{(Question 3) PDF'S As Function Of Samples in increasing order}$").show()



def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0,0,4,0])
    sigma = np.array (([1,0.2,0,0.5],[0.2,2,0,0],[0,0,1,0],[0.5,0,0,1]))
    X= np.random.multivariate_normal(mu, sigma, size = 1000)
    multigaussi = MultivariateGaussian()
    multigaussi.fit(X)
    print(multigaussi.mu_, "\n")
    print(multigaussi.cov_)

    #Question 5 - Likelihood evaluation
    max_value = np.NINF
    max_f1 = 0
    max_f3 = 0
    expectations = np.linspace(-10,10,200)
    list_of_likelihoods = []
    for f1 in expectations:
        for f3 in expectations:
            cur_mu = np.array([f1, 0,f3,0])
            cur_likelihood = MultivariateGaussian.log_likelihood(cur_mu, sigma, X)
            if cur_likelihood > max_value:
                max_f1, max_f3 = f1, f3
                max_value = cur_likelihood
            list_of_likelihoods.append((f1,f3 ,cur_likelihood))

    df = pd.DataFrame(list_of_likelihoods, columns=['f1', 'f3', 'log-likelihood'])
    plotly.express.density_heatmap(df, x='f3', y='f1', z='log-likelihood',
                                   title='Heatmap of Loglikelihoods ranging from -10 to 10, with fixed Co-variance matrix', histfunc='avg').show()


    # Question 6 - Maximum likelihood
    print("maximizing f1 and f3 respectively: {f1}, {f3}".format(f1 = max_f1, f3 = max_f3))
    print("the maximum likelihood is: {max_likelihood}".format(max_likelihood = max_value))

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
