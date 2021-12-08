# EchoStateNetwork
A python framework for Echo State Network applications.

## Current capabilities
ESN.py allows:

- Supervised Learning
  * 'excite', 'reg_fit' and 'validate' can be used in combination to complete supervised learning tasks.
  * Another easier way to execute a supervised learning job is to call 'session' method, where one passes X_train, y_train, X_val, y_val as arguments.
  It is not necessary to pass all of these and in that case ESN automatically recognizes the type of training/validation.
  
- Reinforcement Learning

- Batch Learning

  Use ESNX to enable minibatching.

- Parallel Learning

  Use ESNS to learn with an ensemble of ESNs (and minibatching). This is useful when using vectorized environments in reinforcement learning applications.

- Gradient Based Learning
  * Use ESNN to make the reservoir computer compatible with PyTorch. 
  * It is possible to use PyTorch with the rest of the reservoir types (ESN, ESNX, ESNS, ESNN) as well without the usage of gradients but regression. 
  * ESNN and ESNS only works with PyTorch whereas the other reservoir types also work with Numpy. 
    
