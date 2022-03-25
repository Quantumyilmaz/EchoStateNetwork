.. default-domain::py
.. default-role:: math

.. _Jaeger's recursion formula: https://www.researchgate.net/publication/215385037_The_echo_state_approach_to_analysing_and_training_recurrent_neural_networks-with_an_erratum_note'
.. _ESNRLS paper: https://ieeexplore.ieee.org/document/9458984
.. _set the reservoir layer mode: https://echostatenetwork.readthedocs.io/en/latest/ESN.html#esn-set-reservoir-layer-mode
.. _Module class: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
.. _Linear layer: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
.. _the expanded reservoir layer: https://echostatenetwork.readthedocs.io/en/latest/ESN.html#esn-set-reservoir-layer-mode
.. _Updates reservoir layer.: https://echostatenetwork.readthedocs.io/en/latest/ESN.html#esn-update-reservoir-layer
.. _ESN: https://echostatenetwork.readthedocs.io/en/latest/ESN.html

====
ESNN
====

Echo State Network N

Echo State Network using Pytorch's `Module class`_. `\textbf W_{out}` is a `Linear layer`_ and can be trained via gradients.

It will `set the reservoir layer mode`_ to 

    - ``batch`` if ``batch_size`` is specified
    - ``ensemble`` if ``no_of_reservoirs`` is specified

\ \

.. Note:: ESNN_ inherits all methods from `ESN`_.

\ \

    .. class:: ESNN(  \
                batch_size: int,\
                in_size: int,\
                out_size: int,\
                no_of_reservoirs: int=None,\
                resSize: int = 450,\
                xn: list = [0, 0.4, -0.4], \
                pn: list = [0.9875, 0.00625, 0.00625], \
                random_state: float = None, \
                null_state_init: bool = True,\
                custom_initState: np.ndarray = None,\
                **kwargs)


    **Parameters**
        
        :``batch_size``: Specify the batch size.
        :``in_size``: Specify the input length.
        :``out_size``: Specify the output length.
        :``no_of_reservoirs``: Specify the number of reservoirs in the ensemble.
        :``resSize``: Number of units (nodes) in the reservoir.
        :``xn`` , ``pn``: User can provide custom random variable to control the connectivity of the reservoir. ``xn`` are the values and ``pn`` are the corresponding probabilities.
        :``random_state``: Fix random state. If provided, ``np.random.seed`` and ``torch.manual_seed`` are called.
        :``null_state_init``: If ``True``, starts the reservoir from null state. If ``False``, initializes randomly. Default is ``True``.
        :``custom_initState``: User can give custom initial reservoir state.


    **Keyword Arguments**
            
        :``verbose``: Mute the initialization message.
        :``f``: User can provide custom activation function of the reservoir. Default is identity.
                Functions in the pytorch or numpy libraries are accepted, including functions defined with ``np.vectorize``.
                Some functions can also be given as strings. Accepted strings are:

                    - ``'tanh'``
                    - ``'sigmoid'``
                    - ``'relu'``
                    - ``'leaky_{slope}'``: e.g. ``'leaky_0.5'`` for LeakyReLU with slope equal to `0.5`.
                    - ``'softmax'``
                    - ``'id'``: Identity.
        :``f_out``: User can provide custom output activation. Default is identity.
        :``leak_rate``: Leak parameter in Leaky Integrator ESN (LiESN). Default is `1`.
        :``leak_version``: Give ``0`` for `Jaeger's recursion formula`_, give ``1`` for recursion formula in `ESNRLS paper`_. Default is `0`.
        :``bias``: Set strength of bias in the input, reservoir and readout connections. Disabled by default.
        :``W`` , ``Win`` , ``Wout`` , ``Wback``: User can provide custom reservoir, input, output, feedback matrices.
        :``use_torch``: Use pytorch instead of numpy. Will use cuda if available.
        :``device``: Give ``'cpu'`` if ``use_torch`` is ``True``, CUDA is available on your device but you want to use CPU.
        :``dtype``: Data type of reservoir. Default is ``float64``.


------------
ESNN.forward
------------

Forward pass of the network.

.. Note:: Output feedback is not supported yet.

#. `Updates reservoir layer.`_
#. Returns `(\textbf W_{out}^T \cdot \textbf U)^T`, where `\textbf{U}` is the extended input `[x^T;\textbf H^T;\textbf 1]^T` and `\textbf H` is `the expanded reservoir layer`_.


    .. method:: forward(x:torch.Tensor) -> torch.Tensor



    **Parameters**

        :``x``: Input.