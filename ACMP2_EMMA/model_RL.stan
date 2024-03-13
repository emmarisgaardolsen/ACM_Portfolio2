data {
    int<lower=1> trials; // number of trials
    array[trials] int<lower=0,upper=1> choice; // 1 or 2
    array[trials] int<lower=0,upper=1> feedback; // feedback 
} 

transformed data {
  real<lower=0, upper=1> initialValue;  // initial values for V
  initialValue = 0.5; // 0.5 for both V (should it be 0??)
}

parameters {
    real<lower=0, upper=1> alpha; // learning rate
    real logTau; // logtau, logarithm of the inverse temperature. will be transformed to tau in the transformed parameter block
}

transformed parameters {
  real<lower=0> tau; // tau, inverse temperature 
  tau = exp(logTau); // tau is the exponential of logTau
  
}


model {
    real pe; // prediction error
    array[trials] real v1; // value of left choice, 1d array
    array[trials] real v2; // value of right choice, 1d array
    real prob; // action probabilities
    
    
    target += beta_lpdf(alpha | 1, 1); // prior for alpha
    target += normal_lpdf(logTau | 0, 1);
    
    // generate log likelihood of the choice on first trial
    v1[1] = initialValue;
    v2[1] = initialValue;
    
    prob = 1 / (1 + exp(-tau * (v1[1] - v2[1])));
    target += bernoulli_lpmf(choice[1] | prob);
    
    // log likelihood for trials 2:n
    for (t in 2:trials) {
      
      v1[t] = v1[t-1] + alpha * choice[t-1] * (feedback[t-1] - v1[t-1]); // update value of left choice
      v2[t] = v2[t-1] + alpha * (1 - choice[t-1]) * (feedback[t-1] - v2[t-1]); // update value of right choice
      
      prob = 1 / (1 + exp(-tau * (v1[t] - v2[t]))); // action prob. computed via softmax

      target += bernoulli_lpmf(choice[t] | prob); // log likelihood of the choice
      }
}


/// OBS BELOW I AM IN DOUBT, RICCARDO (how to posteriors, and prior preds???)
generated quantities{
  real<lower=0, upper=1> alpha_prior;
  real<lower=0> tau_prior; // prior for tau

  array[trials] int<lower=0, upper=1> choice_prediction;
  array[trials] real<lower=0, upper=1> v1; 
  array[trials] real<lower=0, upper=1> v2;
  real prob;

  // define prior
  alpha_prior = beta_rng(2, 2); // prior for alpha, could try also uniform_rng(0,1)
  tau_prior = exp(normal_rng(0, 1)); // prior for tau
  
  v1[1] = initialValue;
  v2[1] = initialValue;
  
  // choice on trial 1 
  prob = 1 / (1 + exp(-tau * (v1[1] - v2[1])));
  choice_prediction[1] = bernoulli_rng(prob); // choice on trial 1 
  
  // predicting choices on trials 2:n
  for (t in 2:trials) {
    
        v1[t] = v1[t-1] + alpha * choice[t-1] * (feedback[t-1] - v1[t-1]);
        v2[t] = v2[t-1] + alpha * (1 - choice[t-1]) * (feedback[t-1] - v2[t-1]);
        
        prob = 1 / (1 + exp(-tau * (v1[t] - v2[t]))); // action prob. computed via softmax
        choice_prediction[t] = bernoulli_rng(prob); // choice on trial t
    }
}
