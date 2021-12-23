# [EchoStateNetwork](https://echostatenetwork.readthedocs.io/)
A python framework for Echo State Network applications.

## Documentation

The documentation can be found [here](https://echostatenetwork.readthedocs.io/).

## Installation
[Download EchoStateNetwork.py](https://github.com/Quantumyilmaz/EchoStateNetwork) and place it, where you want to import it from.

## Current capabilities
- Supervised Learning
  * [``excite``](https://echostatenetwork.readthedocs.io/en/latest/ESN.html#esn-excite), [``reg_fit``](https://echostatenetwork.readthedocs.io/en/latest/ESN.html#esn-reg-fit) and [``validate``](https://echostatenetwork.readthedocs.io/en/latest/ESN.html#esn-validate) can be used in combination to complete supervised learning tasks.
  * Another easier way to execute a supervised learning job is to call [``session``](https://echostatenetwork.readthedocs.io/en/latest/ESN.html#esn-session) method, where one passes all training and validation data at once as arguments.
  
- Reinforcement Learning
  * Examples of PPO and DQN algorithms will be provided soon.

- Learning with Batches

  Use [ESNX](https://echostatenetwork.readthedocs.io/en/latest/ESNX.html), [ESNS](https://echostatenetwork.readthedocs.io/en/latest/ESNS.html) or [ESNN](https://echostatenetwork.readthedocs.io/en/latest/ESNN.html) to enable minibatching.

- Learning in Parallel

  Use [ESNS](https://echostatenetwork.readthedocs.io/en/latest/ESNS.html) or [ESNN](https://echostatenetwork.readthedocs.io/en/latest/ESNN.html) to learn with an ensemble of Echo State Networks and if desired minibatching. This is useful when using vectorized environments in reinforcement learning applications.

- Gradient Based Optimization
  * Use [ESNN](https://echostatenetwork.readthedocs.io/en/latest/ESNN.html) to make the reservoir computer compatible with PyTorch. 
  * It is possible to use PyTorch with the rest of the reservoir objects ([ESN](https://echostatenetwork.readthedocs.io/en/latest/ESN.html), [ESNX](https://echostatenetwork.readthedocs.io/en/latest/ESNX.html), [ESNS](https://echostatenetwork.readthedocs.io/en/latest/ESNS.html), [ESNN](https://echostatenetwork.readthedocs.io/en/latest/ESNN.html)) as well without the usage of gradients but regression. 
  * [ESNN](https://echostatenetwork.readthedocs.io/en/latest/ESNN.html) and [ESNS](https://echostatenetwork.readthedocs.io/en/latest/ESNS.html) work with PyTorch only whereas the other reservoir objects also work with Numpy. 
    
