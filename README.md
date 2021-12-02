# EchoStateNetwork
A python framework for Echo State Network applications.

## Current capabilities

- [Supervised Learning](https://quantumyilmaz.github.io/MTFS21/Examples/RC_SP/LOB/RC_SP_LOB.html)
  * 'excite', 'reg_fit' and 'validate' can be used in combination as shown in the link.
  * Another easier way to execute a supervised learning job is to call 'session' method, where one passes X_train, y_train, X_val, y_val as arguments.
  It is not necessary to pass all of these and in that case ESN automatically recognizes the [type of training/validation](https://quantumyilmaz.github.io/MTFS21/Examples/RC_SP/summary/RC_SP_summary.html).
  
- Reinforcement Learning

- Batch Learning
  Use ESNX to enable minibatching.

- Parallel Learning
Use ESNS to learn with an ensemble of ESNs (and minibatching). This is useful when using vectorized environments in reinforcement learning applications.
