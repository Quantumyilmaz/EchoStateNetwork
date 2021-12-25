.. default-domain::py
.. default-role:: math

.. _Jaeger's recursion formula: https://www.researchgate.net/publication/215385037_The_echo_state_approach_to_analysing_and_training_recurrent_neural_networks-with_an_erratum_note'
.. _ESNRLS paper: https://ieeexplore.ieee.org/document/9458984
.. _set the reservoir layer mode: https://echostatenetwork.readthedocs.io/en/latest/ESN.html#esn-set-reservoir-layer-mode
.. _ESN: https://echostatenetwork.readthedocs.io/en/latest/ESN.html

====
ESNX
====

Echo State Network X

ESN for multitasking such as when using (mini)batches. It will `set the reservoir layer mode`_ to ``batch``.

.. Note:: ESNX_ inherits all methods from `ESN`_.

\ \

    .. class:: ESNX(  \
                batch_size: int, \
                W: np.ndarray = None,  \
                resSize: int = 450,  \
                xn: list = [0,0.4,-0.4],  \
                pn: list = [0.9875, 0.00625, 0.00625],  \
                random_state: float = None,  \
                null_state_init: bool = True, \
                custom_initState: np.ndarray = None, \
                **kwargs)


    **Parameters**

        :``batch_size``: Specify the batch size.
        :``W``: User can provide custom reservoir matrix.
        :``resSize``: Number of units (nodes) in the reservoir.
        :``xn`` , ``pn``: User can provide custom random variable to control the connectivity of the reservoir. ``xn`` are the values and ``pn`` are the corresponding probabilities.
        :``random_state``: Fix random state. If provided, ``np.random.seed`` and ``torch.manual_seed`` are called.
        :``null_state_init``: If ``True``, starts the reservoir from null state. If ``False``, initializes randomly. Default is ``True``.
        :``custom_initState``: User can give custom initial reservoir state.

    **Keyword Arguments**
            
        :``verbose``: Mute the initialization message.
        :``f``: User can provide custom activation function of the reservoir. 
                Functions in the pytorch or numpy libraries are accepted, including functions defined with ``np.vectorize``.
                Some functions can also be given as strings. Accepted strings are:

                    - ``'tanh'``
                    - ``'sigmoid'``
                    - ``'relu'``
                    - ``'leaky_{slope}'``: e.g. ``'leaky_0.5'`` for LeakyReLU with slope equal to `0.5`.
                    - ``'softmax'``
                    - ``'id'``: Identity.
        :``leak_rate``: Leak parameter in Leaky Integrator ESN (LiESN).
        :``leak_version``: Give ``0`` for `Jaeger's recursion formula`_, give ``1`` for recursion formula in `ESNRLS paper`_.
        :``bias``: Set strength of bias in the input, reservoir and readout connections.
        :``Win`` , ``Wout`` , ``Wback``: User can provide custom input, output, feedback matrices.
        :``use_torch``: Use pytorch instead of numpy. Will use cuda if available.