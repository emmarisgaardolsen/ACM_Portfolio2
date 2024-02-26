data {
  int<lower=0> N; // number of trials
  int<lower=0,upper=1> choice[N]; // choice outcome for each trial
  real value_left[N]; // value for left for each trial
  real value_right[N]; // value for right for each trial
}

parameters {
  real alpha; // intercept
  real beta; // slope
}

model {
  alpha ~ normal(0, 1); // weakly informative prior for alpha
  beta ~ normal(0, 1); // weakly informative prior for beta

  for (n in 1:N) {
    // binomial likelihood with logit link function
    choice[n] ~ bernoulli_logit(alpha + beta * (value_right[n] - value_left[n]));
  }
}