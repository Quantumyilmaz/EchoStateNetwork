
.. default-domain::py
.. default-role:: math


.. _Jaeger's recursion formula: https://www.researchgate.net/publication/215385037_The_echo_state_approach_to_analysing_and_training_recurrent_neural_networks-with_an_erratum_note'
.. _ESNRLS paper: https://ieeexplore.ieee.org/document/9458984
.. _scikit documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html?highlight=ridge#sklearn.linear_model.Ridge
.. _See: https://echostatenetwork.readthedocs.io/en/latest/ESN.html#ESN
.. _ESNX: ENSX.rst
.. _ESNS: ENSS.rst
.. _ESNN: ENSN.rst


===
ESN
===

Echo State Network


    .. class:: ESN( \
                            resSize: Optional[int]=400, \
                            xn: Optional[list[float]]=[0,0.4,-0.4], \
                            pn: Optional[list[float]]=[0.9875, 0.00625, 0.00625], \
                            random_state: Optional[float]=None, \
                            null_state_init: bool=True, \
                            custom_initState: Optional[np.ndarray]=None, \
                            **kwargs)


    **Parameters**

        :``resSize``: Number of units (nodes) in the reservoir.
        :``xn`` , ``pn``: User can provide custom random variable to control the connectivity of the reservoir. ``xn`` are the values and ``pn`` are the corresponding probabilities.
        :``random_state``: Fix random state. If provided, ``np.random.seed`` and ``torch.manual_seed`` are called.
        :``null_state_init``: If ``True``, starts the reservoir from null state. If ``False``, initializes randomly. Default is ``True``.
        :``custom_initState``: User can give custom initial reservoir state.


    **Keyword Arguments**
            
        :``verbose``: Give ``False`` to mute the messages.
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
        :``device``: Give ``'cpu'`` if ``use_torch`` is ``True`` and CUDA is available on your device but you want to use CPU.
        :``dtype``: Data type of reservoir. Default is ``float64``.
    
    **Attributes**

        :``bias``: Strength of bias in the input layer. *float*
        :``leak_rate``: Leak parameter in Leaky Integrator ESN (LiESN). *float*
        :``leak_version``: Give ``0`` for `Jaeger's recursion formula`_, give ``1`` for recursion formula in `ESNRLS paper`_. *int*
        :``f``: Activation function of the reservoir. *Callable*
        :``f_out``: Output activation function of the reservoir. *Callable*
        :``resSize``: Size of the reservoir. *int*
        :``reservoir_layer``: Reservoir layer. *np.ndarray | torch.Tensor*
        :``spectral_radius``: Spectral radius of the reservoir matrix. *float*
        :``spectral_norm``: Spectral norm of the reservoir matrix. *float*
        :``device``: Device, which the network computes on. *str*
        :``dtype``: Data type of the network. *str*
        :``W`` , ``Win`` , ``Wout`` , ``Wback``: Reservoir, input, output, feedback matrices. *np.ndarray | torch.Tensor*


---------------------------
ESN.scale_reservoir_weights
---------------------------

Scales the reservoir connection matrix to have certain spectral norm or radius. One can directly assign the desired value to ``spectral_radius`` or ``spectral_norm`` attributes instead of using this method.


    .. method:: scale_reservoir_weights(desired_scaling: float, reference: str) -> None


    **Parameters**

        :``desired_scaling``: Scales the reservoir matrix to have the desired spectral norm or radius.
        :``reference``: Give ``'ev'`` (eigenvalue) to choose spectral radius, ``'sv'`` (singular value) to choose spectral norm as reference.


---------------------------
ESN.reconnect_reservoir
---------------------------

Assigns new matrix to the reservoir with redefined connectivity.


    .. method:: reconnect_reservoir(xn: list[float],pn: list[float],**kwargs) -> None


    **Parameters**

        :``xn`` , ``pn``: User can provide random variable to alter the connectivity of the reservoir. ``xn`` are the values and ``pn`` are the corresponding probabilities of the random variable.
        :``verbose``: Set to False to mute the messages.

    **Keyword Arguments**

        :``verbose``: Give ``False`` to mute the messages.


----------
ESN.excite
----------

Time series data is used to update the reservoir nodes according to the formula:

`x_{n+1} = (1-\alpha) \cdot x_n + \alpha \cdot f(\textbf W_{in} \cdot u_{n+1} + \textbf W \cdot x_n + \textbf W_{back} \cdot y_n)`


, where `x` are the reservoir nodes, `u` inputs, `y` labels, `\alpha` leaking rate, `f` activation function.
This formula is for when both ``u`` and ``y`` are provided.

.. Note:: The appropriate update formula is automatically recognized from the given parameters.

After initial transient, updated `x` are registered at each iteration and can be called via ``reg_X`` attribute (history of states used in regression).

    .. method:: excite(  \
                    u: Optional[np.ndarray]=None,  \
                    y: Optional[np.ndarray]=None,  \
                    initLen: Optional[int]=None,   \
                    trainLen: Optional[int]=None,  \
                    initTrainLen_ratio: Optional[float]=None,  \
                    wobble: bool=False,  \
                    wobbler: Optional[np.ndarray]=None,  \
                    **kwargs) -> None


    **Parameters**

        :``u``: Input. Has shape [...,time].
        :``y``: To be forecast. Has shape [...,time].
        :``initLen``: Number of timesteps to be taken as initial transient tolarance. Will override initTrainLen_ratio. Will be set to an eighth of the training length if not provided.
        :``trainLen``: Total number of training steps. Will be set to the length of input data.
        :``initTrainLen_ratio``: Alternative to initLen, the user can provide the initialization period as ratio of the training length. The input ``8`` would mean that the initialization period will be an eighth of the training length.
        :``wobble``: For enabling noise to be added to ``y``.
        :``wobbler``: User can provide custom noise. Default is ``np.random.uniform(-1,1)/10000``.


    **Keyword Arguments**
                    
        :``validation_mode``: Set to ``True`` to use ``excite`` in validation mode to prepare the reservoir for validation.
            
            .. Note:: To use this feature, ``excite`` must be called in training mode first.

-----------
ESN.fit
-----------

Does a regression to ``y`` using the registered reservoir updates, which can be obtained from attribute ``reg_X`` (history of states used in regression):
`\text W_{out} = argmin_{w} ||y - Xw||^2_2 + \eta * ||w||^2_2`

    .. method:: fit( \
                    y: np.ndarray, \
                    f_out_inverse: Optional[Callable]=None, \
                    regr: Optional[Callable]=None, \
                    reg_type: str="ridge", \
                    ridge_param: float=1e-8, \
                    solver: str="auto", \
                    error_measure: str="mse", \
                    **kwargs) -> np.ndarray

    **Parameters**

        :``y``: Data to fit.
        :``f_out_inverse``: User can give custom output activation. Please give the INVERSE of the activation function. No activation is used by default.
        :``regr``: User can give custom regressor. Overrides other settings if provided. If not provided, will be set to scikit-learn's regressor.
        :``reg_type``: Regression type. Can be ``'ridge'``, ``'linear'`` or ``'pinv'`` (Moore-Penrose pseudo inverse). Default is ``'ridge'``.
        :``ridge_param``: Regularization factor in ridge regression.
        :``solver``: See `scikit documentation`_.
        :``error_measure``: Type of error to be displayed. Can be ``'mse'`` (Mean Squared Error) or ``'mape'`` (Mean Absolute Percentage Error).

    **Keyword Arguments**

        :``verbose``: For the error message. 

        :``reg_X``: Lets you overwrite ``reg_X`` attribute (history of states used in regression) with a custom one of your choice. \
                            
            .. tip:: 

                For online training purposes, i.e. you call ``excite`` up to a certain point in your data and do a forecast at that point and repeat it at later points in your data.




------------
ESN.validate
------------

Returns forecast.

    .. method:: validate( \
                    u: Optional[np.ndarray]=None, \
                    y: Optional[np.ndarray]=None, \
                    valLen: Optional[int]=None, \
                    **kwargs) -> np.ndarray


    **Parameters**

        :``u``: Input series. Has shape [...,time].

        :``y``: To be forecast. Has shape [...,time].

        :``valLen``: Validation length. 
        
            .. Note:: If ``u`` or ``y`` is provided it is not needed to be set. Mostly necessary for when neither ``u`` nor ``y`` is present.

    **Keyword Arguments**

        :``wobble``: For enabling random noise. Default is False.
        :``wobbler``: User can provide custom noise. Disabled per default.



-----------
ESN.session
-----------

Executes a whole training/validation session by calling the methods ``excite``, ``train`` and ``validate``. Returns the forecasts.

    .. method:: session( \
                            X_t: Optional[np.ndarray]=None, \
                            y_t: Optional[np.ndarray]=None, \
                            X_v: Optional[np.ndarray]=None, \
                            y_v: Optional[np.ndarray]=None, \
                            training_data: Optional[np.ndarray]=None, \
                            f_out_inverse: Optional[Callable]=None, \
                            initLen: Optional[int]=None,  \
                            initTrainLen_ratio: Optional[float]=None, \
                            trainLen: Optional[int]=None, \
                            valLen: Optional[int]=None, \
                            wobble_train: bool=False, \
                            wobbler_train: Optional[np.ndarray]=None, \
                            null_state_init: bool=True, \
                            custom_initState: Optional[np.ndarray]=None, \
                            regr: Optional[Callable]=None, \
                            reg_type: str="ridge", \
                            ridge_param: float=1e-8, \
                            solver: str="auto", \
                            error_measure: str="mse", \
                            **kwargs \
                            ) -> np.ndarray


    **Parameters**

        :``X_t``: Training inputs. Has shape [...,time].
        :``y_t``: Training targets. Has shape [...,time].
        :``X_v``: Validation inputs. Has shape [...,time].
        :``y_v``: Validation targets. Has shape [...,time].
        :``training_data``: Data to fit to in regression. It will be set to ``y_t`` automatically if it is not provided. Either way, ``y_t`` will be used when calling ``excite``.
        :``f_out_inverse``: Please give the INVERSE output activation function. No activation is used by default.
        :``initLen``: No of timesteps to initialize the reservoir. Will override initTrainLen_ratio. Will be set to an eighth of the training length if not provided.
        :``initTrainLen_ratio``: Alternative to initLen, the user can provide the initialization period as ratio of the training length. An input of 8 would mean that the initialization period will be an eighth of the training length.
        :``trainLen``: Total no of training steps. Will be set to the length of input data, if not provided.
        :``valLen``: Total no of validation steps. Will be set to the length of input data, if not provided.
        :``wobble_train``: For enabling noise to be added to ``y_t``.
        :``wobbler_train``: User can provide custom noise. Default is ``np.random.uniform(-1,1)/10000``.
        :``null_state_init``: If ``True``, starts the reservoir from null state. If ``False``, initializes randomly. Default is ``True``.
        :``custom_initState``: User can give custom initial reservoir state.
        :``regr``: User can give custom regressor. Overrides other settings if provided. If not provided, will be set to scikit-learn's regressor.
        :``reg_type``: Regression type. Can be ``'ridge'``, ``'linear'`` or ``'pinv'`` (Moore-Penrose pseudo inverse). Default is ``'ridge'``.
        :``ridge_param``: Regularization factor in ridge regression.
        :``solver``: See `scikit documentation`_.
        :``error_measure``: Type of error to be displayed. Can be ``'mse'`` (Mean Squared Error) or ``'mape'`` (Mean Absolute Percentage Error).

    **Keyword Arguments**

        :``wobble_val``: For enabling noise to be added to ``y_val`` during validation. Default is False.
        :``wobbler_val``: User can provide custom noise to be added to ``y_val``. Disabled per default.
        :``train_only``: Set to True to perform a training session only, i.e. no validation is done.
        :``verbose``: Mute the training error messages.



--------------------------
ESN.update_reservoir_layer
--------------------------

Applies one-step update to the reservoir layer using:
`x_{n+1} = (1-\alpha) \cdot x_n + \alpha \cdot f(\textbf W_{in} \cdot u_{n+1} + \textbf W \cdot x_n + \textbf W_{back} \cdot y_n)`
, where `x` are the reservoir nodes, `u` inputs, `y` labels, `\alpha` leaking rate, `f` activation function.
This formula is for when both ``u`` and ``y`` are provided.

.. Note:: The appropriate update formula is automatically recognized from the given parameters.

\ \

    .. method::   update_reservoir_layer( \
                    in_: Optional[np.ndarray | torch.Tensor]=None  \
                    ,out_: Optional[np.ndarray | torch.Tensor]=None  \
                    ,mode:Optional[str]=None) -> None



    **Parameters**

        :``in_``: Input array.
        :``out_``: Output array.
        :``mode``: Optional. Set to ``'train'`` if you are updating the reservoir layer for training purposes. Set to ``'val'`` if you are doing so for validation purposes. \
                This will allow the reservoir object to name the training/validation modes, which can be accessed from ``'training_type'`` and ``'val_type'`` attributes.

------------------------------------
ESN.update_reservoir_layers_serially
------------------------------------

.. warning:: Resets reservoir layer. See `ESN.reset_reservoir_layer`_.

When using the reservoir in ``batch`` or ``ensemble`` mode, the reservoir layer will  be updated using
`x^k = (1-\alpha)x^{k-1} + \alpha \cdot f(\textbf W_{in} \cdot u^{k} + \textbf W \cdot x^{k-1})` \
, where `1\leq k \leq` the batch size, `u^k` the `k^{th}` data point in the batch and `x^0` is the initial reservoir layer before any updates.

`x` here is a matrix with shape:

    (reservoir size,batch size) if in ``batch`` mode.

    (number of reservoirs,reservoir size,batch size) if in ``ensemble`` mode.

.. Note:: Reservoir can be set to ``batch`` or ``ensemble`` mode by using `ESN.set_reservoir_layer_mode`_ or one of the following:

    - `ESNX`_
    - `ESNS`_
    - `ESNN`_

\ \

    .. method:: update_reservoir_layers_serially( \
        , in_: Optional[np.ndarray | torch.Tensor] = None \
        , out_: Optional[np.ndarray | torch.Tensor] = None \
        , mode: Optional[str] = None \
        ,init_size: int = 0) -> None

    **Parameters**

        :``in_``: Input with shape 
                    
                    - (data point length,batch size + initialization length) (see ``init_size``) if in ``batch`` mode.
                    - (number of reservoirs,data point length,batch size + initialization length) if in ``ensemble`` mode.
        :``out_``: **Not supported. WIP.**
        :``leak_version``: Give ``0`` for `Jaeger's recursion formula`_, give ``1`` for recursion formula in `ESNRLS paper`_.
        :``leak_rate``: Leak parameter in Leaky Integrator ESN (LiESN).
        :``mode``: Optional. Set to ``'train'`` if you are updating the reservoir layer for training purposes. Set to ``'val'`` if you are doing so for validation purposes. \
                This will allow the reservoir object to name the training/validation modes, which can be accessed from ``'training_type'`` and ``'val_type'`` attributes.
        :``init_size``: The first ``init_size`` data points are expended for initial transient to pass.

-------------------------
ESN.reset_reservoir_layer
-------------------------

Resets reservoir layer, i.e. sets the reservoir nodes back to their initial state.

    .. method:: reset_reservoir_layer() -> None

----------------------------
ESN.set_reservoir_layer_mode
----------------------------

.. warning:: Resets reservoir layer. See `ESN.reset_reservoir_layer`_.

Sets the reservoir mode to ``single``, ``batch`` or ``ensemble`` by expanding or collapsing the reservoir layer (see shapes below).
Changes the shape of the reservoir layer, which can be obtained from ``reservoir_layer`` attribute.

    - ``single``: reservoir layer has shape (reservoir size,1)
    - ``batch``: reservoir layer has shape (reservoir size,batch size)
    - ``ensemble``: reservoir layer has shape (number of reservoirs,reservoir size,batch size)
  
  \ \

    .. method:: set_reservoir_layer_mode(mode: str,batch_size: Optional[int]=None,no_of_reservoirs: Optional[int]=None) -> None

    **Parameters**

        :``mode``: Set to ``single``, ``batch`` or ``ensemble``.
        :``batch_size``: Necessary if using ``batch`` or ``ensemble``. If not provided ``batch_size`` which was specified at initialization will be used.
        :``no_of_reservoirs``: Necessary if using ``ensemble``. If not provided ``no_of_reservoirs`` which was specified at initialization will be used.


-------------
ESN.copy_from
-------------

Copies the reservoir properties to the current reservoir.

    .. method:: copy_from(reservoir:Self,bind:bool=False,**kwargs) -> None

    **Parameters**

        :``reservoir``: Reservoir to copy from.
        :``bind``: Shares the same memory with the reservoir that is copied from.
    
    **Keyword Arguments**
            
        :``verbose``: Give ``False`` to mute the messages.

-------------------------
ESN.copy_connections_from
-------------------------

Similar to `ESN.copy_from`_ but copies only the connection matrices.

    .. method:: copy_connections_from(reservoir:Self,bind:bool=False,weights_list: Optional[list[str]]=None,**kwargs) -> None

    **Parameters**

        :``reservoir``: Reservoir to copy from.
        :``bind``: Shares the same memory with the reservoir that is copied from.
        :``weights_list``: Give a sublist of the list ``['Wout','W','Win','Wback']`` if you do not want to copy all the connections.

    **Keyword Arguments**
            
        :``verbose``: Give ``False`` to mute the messages.

-------------------
ESN.make_connection
-------------------

Creates the desired connection of the network.

    .. method:: make_connection(w_name:str,inplace:bool=False,**kwargs) -> Optional[np.ndarray | torch.Tensor]

    **Parameters**

        :``w_name``: Name of the connection: ``'Win'``, ``'W'`` or ``'Wback'``.
        :``inplace``: Whether to overwrite the connection.

    **Keyword Arguments**
            
        :``size``: User should provide information on the size associated with the corresponding connection matrix: input size for ``Win``, output size for ``Wback``.
        :``verbose``: Give ``False`` to mute the messages.

---------------------
ESN.delete_connection
---------------------

Deletes the undesired connection of the network.

    .. method:: delete_connection(w_name:str,**kwargs) -> None

    **Parameters**

        :``w_name``: Name of the connection: ``'Win'``, ``'W'`` or ``'Wback'``.

    **Keyword Arguments**
            
        :``verbose``: Give ``False`` to mute the messages.

-------
ESN.cpu
-------

Sends the reservoir to cpu device.

    .. method:: cpu() -> None


--------
ESN.save
--------

Pickles the reservoir to the provided path. Save path example: ``'./saved_reservoir.pkl'``

    .. method:: save(save_path:str) -> None

    **Parameters**

        :``save_path``: Path to pickle the reservoir to.

--------
ESN.load
--------

Loads the reservoir from the provided path. Load path example: ``'./saved_reservoir.pkl'``

    .. method:: load(load_path:str) -> None

    **Parameters**

        :``load_path``: Path to load the reservoir from.

---------------------
ESN.mute
---------------------

Toggles the verbosity of the network. 

    .. method:: mute(verbose:Optional[bool]=None) -> None

    **Parameters**

        :``verbose``: Use this parameter to force (un)verbosity by giving ``True`` or ``False``. If not given, the method toggles the verbosity.


---------------------------
ESN.forward
---------------------------

.. warning:: Updates reservoir layer. See `ESN.update_reservoir_layer`_.

One step forward pass through the whole network for given input and/or output.

    .. method::  forward(in_:Optional[np.ndarray | torch.Tensor]=None,out_:Optional[np.ndarray | torch.Tensor]=None) -> np.ndarray | torch.Tensor

    **Parameters**

        :``in_``: Input.
        :``out_``: Output.


---------------------------
ESN.__call__
---------------------------
Passes a given input `\textbf X` through the readout: `f_{out}(\textbf W_{out} \cdot \textbf X)`

    .. method:: __call__(x:np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor

    **Parameters**

        :``x``: Input.





.. .. code-block::
..    :caption: A cool example

..        The output of this line starts with four spaces.


