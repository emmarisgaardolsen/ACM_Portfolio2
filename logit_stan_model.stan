data {
    int<lower=1> trials; // number of trials
    array[trials] int<lower=1,upper=2> choice; // 0 or 1
    array[trials] int<lower=0,upper=1> feedback; // feedback i
} 

transformed data {
  vector[2] initValue;  // initial values for V
  initValue = rep_vector(0.5, 2); // 0.5 for both V (should it be 0??)
}

parameters {
    real<lower=0, upper=1> alpha; // learning rate
    real<lower=0, upper=20> temperature; // softmax inv.temp. OBS is this a real number?
}

model {
    real pe; // prediction error
    vector[2] value; // value of each action
    vector[2] theta; // action probabilities
    
    target += uniform_lpdf(alpha | 0, 1); // prior for alpha
    target += uniform_lpdf(temperature | 0, 20); // prior for temperature
    
    value = initValue;
    
    for (t in 1:trials) {
        theta = softmax( temperature * value); // action prob. computed via softmax
        target += categorical_lpmf(choice[t] | theta);
        
        pe = feedback[t] - value[choice[t]]; // compute pe for chosen value only
        value[choice[t]] = value[choice[t]] + alpha * pe; // update chosen V
    }
    
}

generated quantities{
  real<lower=0, upper=1> alpha_prior;
  real<lower=0, upper=1> alpha_posterior;
  real<lower=0, upper=20> temperature_prior;
  real<lower=0, upper=20> temperature_posterior;
  
  int<lower=0, upper=t> prior_preds_alpha;
  int<lower=0, upper=t> posterior_preds_alpha;
  int<lower=0, upper=t> prior_preds_temp;
  int<lower=0, upper=t> posterior_preds_temp;
  
  real pe;
  vector[2] value;
  vector[2] theta;
  
  real log_lik;
  
  alpha_prior = inv_logit(uniform_rng(0,1));
  alpha_posterior = inv_logit(alpha);
  temperature_prior = inv_logit(uniform_rng(0,20));
  temperature_prior = inv_logit(temperature);
  
  prior_preds_alpha=binomial_rng(t, alpha_prior);
  posterior_preds_alpha=binomial_rng(t, inv_logit(alpha));
  prior_preds_temp=binomial_rng(t, temperature_prior);
  posterior_preds_temp=binomial_rng(t, inv_logit(temperature));
  
  value = initValue;
  log_lik = 0;
  
  for (t in 1:trials) {
        theta = softmax( temperature * value); // action prob. computed via softmax
        log_lik = log_lik + bernoulli_lpdf(choice[t] | theta);
        
        pe = feedback[t] - value[choice[t]]; // compute pe for chosen value only
        value[choice[t]] = value[choice[t]] + alpha * pe; // update chosen V
    }
  
}
