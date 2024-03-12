data {
    int<lower=1> trials; // number of trials
    array[trials] int<lower=1,upper=2> choice; // 0 or 1
    array[trials] int<lower=0,upper=1> feedback; // feedback 
} 

transformed data {
  vector[2] initValue;  // initial values for V
  initValue = rep_vector(0.5, 2); // 0.5 for both V (should it be 0??)
}

parameters {
    real<lower=0, upper=1> alpha; // learning rate
    real<lower=0, upper=20> logTau; // logtau, logarithm of the inverse temperature 
}

transformed parameters {
  real<lower=0> tau; 
  tau = exp(logTau); // tau is the exponential of logTau
}


model {
    real pe; // prediction error
    vector[2] value; // value of each action
    vector[2] prob; // action probabilities
    
    target += beta_lpdf(alpha | 1, 1); // prior for alpha
    target += normal_lpdf(logTau | 0, 1);
    
    value = initValue;
    
    for (t in 1:trials) {
        prob = 1 / (1 + exp(-tau * (value[1] - value[2]))); // action prob. computed via softmax
        target += categorical_lpmf(choice[t] | prob);
        
        pe = feedback[t] - value[choice[t]]; // compute pe for chosen value only
        value[choice[t]] = value[choice[t]] + alpha * pe; // update chosen V
    }
}

generated quantities{
  real<lower=0, upper=1> alpha_prior;
  real<lower=0> tau_prior; // prior for tau
  
  real pe;
  vector[2] value;
  vector[2] theta;
  
  real log_lik;
  
  alpha_prior = beta_rng(2, 2);
  tau_prior = exp(normal_rng(0, 1));
  
  value = initValue;
  log_lik = 0;
  
  for (t in 1:trials) {
        prob = 1 / (1 + exp(-tau * (value[1] - value[2]))); // action prob. computed via softmax
        log_lik = log_lik + categorical_lpmf(choice[t] | prob);
        
        pe = feedback[t] - value[choice[t]]; // compute pe for chosen value only
        value[choice[t]] = value[choice[t]] + alpha * pe; // update chosen V
    }
  
}
