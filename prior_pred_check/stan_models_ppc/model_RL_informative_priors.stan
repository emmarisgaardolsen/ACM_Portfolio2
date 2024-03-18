
// Input data provided to the stan model
data {
    int<lower=1> trials;                            // Number of trials
    array[trials] int<lower=0,upper=1> choice;      // Array of integers between 0 and 1, containing agent choice 
    array[trials] int<lower=0,upper=1> feedback;    // Feedback is an array of integers between 0 and 1. Prediction error is calculated based on this. 
} 

// Transformed data 
transformed data {
  real<lower=0, upper=1> initialValue;              // The initial expected value is a real number between 0 and 1
  initialValue = 0.5;                               // The initial value is 0.5, will be used for both v1 and v2. So equal initial values for both choices
}

// Parameters to be sampled by HMC (alpha and logTau)
parameters {
    real<lower=0, upper=1> alpha;                   // alpha is the learning rate, real number between 0 and 1
    real logTau;                                    // logtau, logarithm of the inverse temperature. used to sample in unconstrained space. will be transformed to tau in the transformed parameter block
}


// transformed data block declaring and defining variables 
// that do not need to be changed when running the program.
transformed parameters {
  real<lower=0> tau;                                // tau, inverse temperature. can't be negative 
  tau = exp(logTau);                                // tau is specified as the exponential of logTau to make sure lower boundary is 0
}


// the cognitive model (includes likelihood function and priors)
model {
    real pe; // prediction error is an unbounded real number
    array[trials] real v1; // value of left choice, 1d array
    array[trials] real v2; // value of right choice, 1d array
    real prob; // choice probabilities
    
    
    target += beta_lpdf(alpha | 4, 4); // Strong belief in a central tendency.
    target += normal_lpdf(logTau | 0, 0.5); // Narrower, expressing more certainty.
    
    // generate log likelihood of the choice on first trial
    v1[1] = initialValue; // initial value of left choice
    v2[1] = initialValue; // initial value of right choice
    
    prob = 1 / (1 + exp(-tau * (v1[1] - v2[1]))); // choice prob. computed via softmax
    target += bernoulli_lpmf(choice[1] | prob); // log likelihood of the choice
    
    // log likelihood for trials 2:n
    for (t in 2:trials) {  // loop over trials
      
      v1[t] = v1[t-1] + alpha * choice[t-1] * (feedback[t-1] - v1[t-1]); // update value of left choice
      v2[t] = v2[t-1] + alpha * (1 - choice[t-1]) * (feedback[t-1] - v2[t-1]); // update value of right choice
      
      prob = 1 / (1 + exp(-tau * (v1[t] - v2[t]))); // action prob. computed via softmax

      target += bernoulli_lpmf(choice[t] | prob); // log likelihood of the choice
      }
}


/// post-processing of the model
generated quantities{
  real<lower=0, upper=1> alpha_prior; // prior for alpha, bounded between 0 and 1
  real<lower=0> tau_prior; // prior for tau

  array[trials] int<lower=0, upper=1> choice_prediction; // predicted choices
  array[trials] real<lower=0, upper=1> v1; // value of left choice, to be stored in 1d array
  array[trials] real<lower=0, upper=1> v2; // value of right choice, to be stored in 1d array
  real prob; // choice probability

  // define prior
  alpha_prior = beta_rng(2, 2); // beta prior for alpha (could try also uniform_rng(0,1) or uniform_lp)
  tau_prior = exp(normal_rng(0, 1)); // log tau is sampled from gaussian prior and transformed to tau by exponentiation

  v1[1] = initialValue; // value of left choice in the first trial
  v2[1] = initialValue; // value of right choice in the first trial
  
  // choice on trial 1 
  prob = 1 / (1 + exp(-tau * (v1[1] - v2[1]))); // choice prob. computed via softmax, using the initial values on first trial
  choice_prediction[1] = bernoulli_rng(prob); // choice on trial 1 is sampled 
  
  // predicting choices on trials 2:n
  for (t in 2:trials) {  // loop over remaining trials
    
        v1[t] = v1[t-1] + alpha * choice[t-1] * (feedback[t-1] - v1[t-1]); // update value of left choice
        v2[t] = v2[t-1] + alpha * (1 - choice[t-1]) * (feedback[t-1] - v2[t-1]); // update value of right choice
        
        prob = 1 / (1 + exp(-tau * (v1[t] - v2[t]))); // choice prob. computed via softmax
        choice_prediction[t] = bernoulli_rng(prob); // choice on trial t sampled from likelihood function
    }
}
