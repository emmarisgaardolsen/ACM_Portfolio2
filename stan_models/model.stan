// data 
data {
    int<lower=0> N; // number of data points
    int<lower=0> K; // number of predictors
    matrix[N, K] x; // predictor matrix
    vector[N] y; // outcome vector
}