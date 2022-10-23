# e-CRT

This repository provides a python code for e-CRT; a model-free sequential test for conditional independence.
![martingales.png](martingales.png)
##
We present a novel sequential test for conditional independence, 
inspired by the model-X conditional randomization test and the approach of testing by betting.
Our online framework allows researches to analyze the data more efficiently,
without affecting the validity of the test.

To learn more about our testing framework, see our [paper](https://arxiv.org/abs/2210.00354).

## Dependencies
- numpy
- pandas
- sklearn
- matplotlib

## Usage 

Our test is available in [src/e_crt.py](src/e_crt.py).
Please note that this code supports continuous label data, 
but can be easily adapted to classification tasks. To do so, replace the default Lasso regression to 
another predictive model, and the default MSE function to another test statistic.

In conditional independence tests, given data ![eq](https://latex.codecogs.com/svg.image?X) 
and a response ![eq](https://latex.codecogs.com/svg.image?Y), 
the null hypothesis is true if ![eq](https://latex.codecogs.com/svg.image?X_j) is 
independent of the response ![eq](https://latex.codecogs.com/svg.image?Y) given 
![eq](https://latex.codecogs.com/svg.image?X_%7B-j%7D),
where ![eq](https://latex.codecogs.com/svg.image?X_j) is the j-th feature, 
and ![eq](https://latex.codecogs.com/svg.image?X_%7B-j%7D) are all the features except the j-th one.

A crucial step in our test is to sample the dummy features from the
conditional distribution of ![eq](https://latex.codecogs.com/svg.image?X_{j}&space;|&space;X_{-j}). 
Yet, in many real-world applications, this distribution may not be known.

In our code we provide an estimation for the conditional distribution by fitting a multivariate Gaussian,
and demonstrate the performance using this approximation both with synthetic data, in 
[examples/synthetic_experiment.ipynb](examples/synthetic_experiment.ipynb), 
and with real data, in [examples/fund_experiment.ipynb](examples/fund_experiment.ipynb).

We demonstrate how the user can provide a function of its own to sample the dummies 
in [examples/hiv_experiment.ipynb](examples/hiv_experiment.ipynb)

## License

This project is licensed under the MIT License - see the [LICENSE](License.txt) file for details.

