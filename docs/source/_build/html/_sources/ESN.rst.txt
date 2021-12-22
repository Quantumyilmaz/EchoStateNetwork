
.. default-domain::py
.. default-role:: math


.. _Jaeger's recursion formula: https://www.researchgate.net/publication/215385037_The_echo_state_approach_to_analysing_and_training_recurrent_neural_networks-with_an_erratum_note'
.. _ESNRLS paper: https://ieeexplore.ieee.org/document/9458984
.. _scikit documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html?highlight=ridge#sklearn.linear_model.Ridge
.. _ESNX: ENSX.rst
.. _ESNS: ENSS.rst
.. _ESNN: ENSN.rst


===
ESN
===

Echo State Network


    .. class:: ESN(self, \
                            W: np.ndarray=None, \
                            resSize: int=400, \
                            xn: list=[0,4,-4], \
                            pn: list=[0.9875, 0.00625, 0.00625], \
                            random_state: float=None, \
                            null_state_init: bool=True, \
                            custom_initState: np.ndarray=None, \
                            **kwargs) 


    |


    **Parameters**


        :``W``: User can provide custom reservoir matrix.
        :``resSize``: Number of units (nodes) in the reservoir.
        :``xn`` , pn: User can provide custom random variable to control the connectivity of the reservoir. ``xn`` are the values and ``pn`` are the corresponding probabilities.
        :``random_state``: Fix random state. If provided, ``np.random.seed`` and ``torch.manual_seed`` are called.
        :``null_state_init``: If ``True``, starts the reservoir from null state. If ``False``, initializes randomly. Default is ``True``.
        :``custom_initState``: User can give custom initial reservoir state.

    |


    **Keyword Arguments**
            
        :``verbose``: Mute the initialization message.
        :``f``: User can provide custom activation function of the reservoir.
        :``leak_rate``: Leak parameter in Leaky Integrator ESN (LiESN).
        :``leak_version``: Give ``0`` for `Jaeger's recursion formula`_, give ``1`` for recursion formula in `ESNRLS paper`_.
        :``bias``: Set strength of bias in the input, reservoir and readout connections.
        :``Win`` , Wout , Wback: User can provide custom input, output, feedback matrices.
        :``use_torch``: Use pytorch instead of numpy. Will use cuda if available.


---------------------------
ESN.scale_reservoir_weights
---------------------------

Scales the reservoir connection matrix to have certain spectral norm or radius.


    .. function:: scale_reservoir_weights(self,desired_scaling: float, reference='ev') -> None


    |

    **Parameters**

        :``desired_scaling``: Scales the reservoir matrix to have the desired spectral norm or radius.
        :``reference``: Give ``'ev'`` (eigenvalue) to choose spectral radius, ``'sv'`` (singular value) to choose spectral norm as reference.


----------
ESN.excite
----------

Time series data is used to update the reservoir nodes according to the formula:

`x(n+1) = (1-\alpha) \cdot x(n) + \alpha \cdot f(W_{in} \cdot u(n+1) + W \cdot x(n) + Wback \cdot y(n))`

, where `x` are the reservoir nodes, `u` inputs, `y` labels, `\alpha` leaking rate, `f` activation function.
This formula is for when both ``u`` and ``y`` are provided.

.. Note:: The appropriate update formula is automatically recognized from the given parameters.

After initial transient, updated `x` are registered at each iteration and can be called via ``reg_X`` attribute.

    .. function:: excite(self,  \
                    u: np.ndarray=None,  \
                    y: np.ndarray=None,  \
                    bias: Union[int,float]=None,  \
                    f: Union[str,Any]=None,  \
                    leak_rate: Union[int,float]=None,  \
                    initLen: int=None,   \
                    trainLen: int=None,  \
                    initTrainLen_ratio: float=None,  \
                    wobble: bool=False,  \
                    wobbler: np.ndarray=None,  \
                    leak_version: int =0,  \
                    **kwargs) -> None

    |

    **Parameters**

        :``u``: Input. Has shape [...,time].
        :``y``: To be forecast. Has shape [...,time].
        :``bias``: Set strength of bias in the input, reservoir and readout connections.
        :``f``: User can provide custom activation function. Default is None. Available activations: ``'tanh'``, ``'relu'``, ``'sigmoid'``. For leaky relu activation, write ``'leaky_{leaky rate}'``, e.g. ``'leaky_0.5'``.
        :``leak_rate``: Leak parameter in Leaky Integrator ESN (LiESN).
        :``leak_version``: Give ``0`` for `Jaeger's recursion formula`_, give ``1`` for recursion formula in `ESNRLS paper`_.
        :``initLen``: Number of timesteps to be taken as initial transient tolarance. Will override initTrainLen_ratio. Will be set to an eighth of the training length if not provided.
        :``trainLen``: Total number of training steps. Will be set to the length of input data.
        :``initTrainLen_ratio``: Alternative to initLen, the user can provide the initialization period as ratio of the training length. The input ``8`` would mean that the initialization period will be an eighth of the training length.
        :``wobble``: For enabling noise to be added to ``y``.
        :``wobbler``: User can provide custom noise. Default is ``np.random.uniform(-1,1)/10000``.

    |

    **Keyword Arguments**
                    
        :``Win``: Custom input weights.
        :``Wback``: Custom feedback weights.
        :``validation_mode``: Set to ``True`` to use ``excite`` in validation mode to prepare the reservoir for validation.
            
            .. Note:: To use this feature, ``excite`` must be called in training mode first.

-----------
ESN.reg_fit
-----------

Does a regression to ``y`` using the registered reservoir updates, which can be obtained from attribute ``reg_X``:
`W^{*} = argmin_{w} ||y - Xw||^2_2 + \eta * ||w||^2_2`

    .. function:: reg_fit(self, \
                    y: np.ndarray, \
                    f_out_inverse=None, \
                    regr=None, \
                    reg_type: str="ridge", \
                    ridge_param: float=1e-8, \
                    solver: str="auto", \
                    error_measure: str="mse", \
                    **kwargs) -> np.ndarray

    |

    **Parameters**

        :``y``: Data to fit.
        :``f_out_inverse``: User can give custom output activation. Please give the INVERSE of the activation function. No activation is used by default.
        :``regr``: User can give custom regressor. Overrides other settings if provided. If not provided, will be set to scikit-learn's regressor.
        :``reg_type``: Regression type. Can be ``ridge`` or ``linear``. Default is ``linear``.
        :``ridge_param``: Regularization factor in ridge regression.
        :``solver``: See `scikit documentation`_.
        :``error_measure``: Type of error to be displayed. Can be ``'mse'`` (Mean Squared Error) or ``'mape'`` (Mean Absolute Percentage Error).

    **Keyword Arguments**

        :``verbose``: For the error message. 

        :``reg_X``: Lets you overwrite ``self.reg_X`` (matrix used in regression) with a custom one of your choice. \
                            
            .. tip:: 

                For online training purposes, i.e. you call ``excite`` up to a certain point in your data and do a forecast at that point and repeat it at later points in your data. Instead of "exciting" reservoir states and doing regression multiple times up to these forecasts at varying points, which is inefficient since you perform same calculations repeatedly, you can excite using all data and use partial excitations, i.e. just the part of self.reg_X relevant and required for the regression.




------------
ESN.validate
------------

Returns forecast.

    .. function:: validate(self, \
                    u: np.ndarray=None, \
                    y: np.ndarray=None, \
                    valLen: int=None, \
                    f_out=lambda x: x, \
                    output_transformer=lambda x:x, \
                    **kwargs) -> np.ndarray

    |

    **Parameters**

        :``u``: Input series. Has shape [...,time].

        :``y``: To be forecast. Has shape [...,time].

        :``valLen``: Validation length. 
        
            .. Note:: If ``u`` or ``y`` is provided it is not needed to be set. Mostly necessary for when neither ``u`` nor ``y`` is present.
        
        :``f_out``: Custom output activation. Default is identity.

        :``output_transformer``: Transforms the reservoir outputs at the very end. Default is identity.

    **Keyword Arguments**

        :``bias``: Set strength of bias in the input, reservoir and readout connections. Default is the one used in training.
        :``f``: User can provide custom reservoir activation function. Default is the one used in training.
        :``leak_rate``: Leak parameter in Leaky Integrator ESN (LiESN). Default is the ``leak_rate`` used in training.
        :``wobble``: For enabling random noise. Default is False.
        :``wobbler``: User can provide custom noise. Disabled per default.




-----------
ESN.session
-----------

Executes a whole training/validation session by calling the methods ``excite``, ``train`` and ``validate``. Returns the forecasts.

    .. function:: session(self, \
                            X_t: np.ndarray=None, \
                            y_t: np.ndarray=None, \
                            X_v: np.ndarray=None, \
                            y_v: np.ndarray=None, \
                            training_data: np.ndarray=None, \
                            bias: int=None, \
                            f=None, \
                            f_out_inverse=None, \
                            f_out=lambda x:x, \
                            output_transformer=lambda x:x, \
                            initLen: int=None,  \
                            initTrainLen_ratio: float=None, \
                            trainLen: int=None, \
                            valLen: int=None, \
                            wobble_train: bool=False, \
                            wobbler_train: np.ndarray=None, \
                            null_state_init: bool=True, \
                            custom_initState: np.ndarray=None, \
                            regr=None, \
                            reg_type: str="ridge", \
                            ridge_param: float=1e-8, \
                            solver: str="auto", \
                            error_measure: str="mse", \
                            **kwargs \
                            ) -> np.ndarray

    |

    **Parameters**

        :``X_t``: Training inputs. Has shape [...,time].
        :``y_t``: Training targets. Has shape [...,time].
        :``X_v``: Validation inputs. Has shape [...,time].
        :``y_v``: Validation targets. Has shape [...,time].
        :``training_data``: Data to fit to in regression. It will be set to ``y_t`` automatically if it is not provided. Either way, ``y_t`` will be used when calling ``excite``.
        :``f_out_inverse``: Please give the INVERSE activation function. User can give custom output activation. No activation is used by default.
        :``f_out``: Custom output activation. Default is identity.
        :``output_transformer``: Transforms the reservoir outputs at the very end. Default is identity.
        :``initLen``: No of timesteps to initialize the reservoir. Will override initTrainLen_ratio. Will be set to an eighth of the training length if not provided.
        :``initTrainLen_ratio``: Alternative to initLen, the user can provide the initialization period as ratio of the training length. An input of 8 would mean that the initialization period will be an eighth of the training length.
        :``trainLen``: Total no of training steps. Will be set to the length of input data, if not provided.
        :``valLen``: Total no of validation steps. Will be set to the length of input data, if not provided.
        :``wobble_train``: For enabling noise to be added to ``y_t``.
        :``wobbler_train``: User can provide custom noise. Default is ``np.random.uniform(-1,1)/10000``.
        :``null_state_init``: If ``True``, starts the reservoir from null state. If ``False``, initializes randomly. Default is ``True``.
        :``custom_initState``: User can give custom initial reservoir state.

    **Keyword Arguments**

        :``Win`` , ``Wback``: User can provide custom input, feedback matrices.
        :``f``: User can provide custom activation function of the reservoir.
        :``bias``: Set strength of bias in the input, reservoir and readout connections.
        :``bias_val``: Set strength of bias in the input, reservoir and readout connections during validation. Default is bias used in training.
        :``f_val``: User can provide custom reservoir activation function to be used during validation. Default is activation used in training.
        :``leak_rate``: Leak parameter in Leaky Integrator ESN (LiESN).
        :``leak_rate_val``: Leak parameter in Leaky Integrator ESN (LiESN) during validation.
        :``leak_version``: Give ``0`` for `Jaeger's recursion formula`_, give ``1`` for recursion formula in `ESNRLS paper`_.
        :``leak_version_val``: Leaking version for validation. Default is the one used in training.
        :``wobble_val``: For enabling noise to be added to ``y_val`` during validation. Default is False.
        :``wobbler_val``: User can provide custom noise to be added to ``y_val``. Disabled per default.
        :``train_only``: Set to True to perform a training session only, i.e. no validation is done.
        :``verbose``: Mute the training error messages.



--------------------------
ESN.update_reservoir_layer
--------------------------

Applies one-step update to the reservoir layer using:
`x_{n+1} = (1-\alpha) \cdot x_n + \alpha \cdot f(W_{in} \cdot u_{n+1} + W \cdot x_n + Wback \cdot y_n)`
, where `x` are the reservoir nodes, `u` inputs, `y` labels, `\alpha` leaking rate, `f` activation function.
This formula is for when both ``u`` and ``y`` are provided.

.. Note:: The appropriate update formula is automatically recognized from the given parameters.

\ \

    .. function::   update_reservoir_layer(self, \
                    in_:Union[np.ndarray,torch.Tensor,NoneType]=None  \
                    ,out_:Union[np.ndarray,torch.Tensor,NoneType]=None  \
                    ,leak_version:int = 0  \
                    ,leak_rate=1.  \
                    ,mode:Optional[str]=None) -> None



    |

    **Parameters**

        :``in_``: Input array.
        :``out_``: Output array.
        :``leak_version``: Give ``0`` for `Jaeger's recursion formula`_, give ``1`` for recursion formula in `ESNRLS paper`_.
        :``leak_rate``: Leak parameter in Leaky Integrator ESN (LiESN).
        :``mode``: Optional. Set to ``'train'`` if you are updating the reservoir layer for training purposes. Set to ``'val'`` if you are doing so for validation purposes. \
                This will allow the reservoir object to name the training/validation modes, which can be accessed from ``'training_type'`` and ``'val_type'`` attributes.

------------------------------------
ESN.update_reservoir_layers_serially
------------------------------------

.. warning:: Resets reservoir layer. See `ESN.reset_reservoir_layer`_.

When using the reservoir in ``batch`` or ``ensemble`` mode, the reservoir layer will  be updated using
`x^k = (1-\alpha)x^{k-1} + f(W_{in}\cdot u^{k-1} + W \cdot x^{k-1})` \
, where `1\leq k \leq` the batch size, `u^k` the `k^{th}` data point in the batch and `x^1` is the initial reservoir layer before any updates. \
`x^1` is updated according to `x^1 = (1-\alpha)x^1 + f(W_{in}\cdot u^1 + W \cdot x^1)`.

`x` here is a matrix with shape:

    (reservoir size,batch size) if in ``batch`` mode.

    (number of reservoirs,reservoir size,batch size) if in ``ensemble`` mode.

.. Note:: Reservoir can be set to ``batch`` or ``ensemble`` mode by using `ESN.set_reservoir_layer_mode`_ or one of the following:

    - `ESNX`_
    - `ESNS`_
    - `ESNN`_

\ \

    .. function:: update_reservoir_layers_serially(self \
        , in_: Union[np.ndarray, torch.Tensor, NoneType] = None \
        , out_: Union[np.ndarray, torch.Tensor, NoneType] = None \
        , leak_version: int = 0 \
        , leak_rate: float=1    \
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

    .. function:: reset_reservoir_layer(self) -> None

----------------------------
ESN.set_reservoir_layer_mode
----------------------------

.. warning:: Resets reservoir layer. See `ESN.reset_reservoir_layer`_.

Sets the reservoir mode to ``single``, ``batch`` or ``ensemble``. Changes the shape of the reservoir layer, which can be obtained from ``reservoir_layer`` attribute.

    - ``single``: reservoir layer has shape (reservoir size,1)
    - ``batch``: reservoir layer has shape (reservoir size,batch size)
    - ``ensemble``: reservoir layer has shape (number of reservoirs,reservoir size,batch size)
  
  \ \

    .. function:: set_reservoir_layer_mode(self,mode: str,batch_size: int=None,no_of_reservoirs :int=None) -> None

    **Parameters**

        :``mode``: Set to ``single``, ``batch`` or ``ensemble``.
        :``batch_size``: Necessary if using ``batch`` or ``ensemble``. If not provided ``batch_size`` which was specified at initialization will be used.
        :``no_of_reservoirs``: Necessary if using ``ensemble``. If not provided ``no_of_reservoirs`` which was specified at initialization will be used.


-------------
ESN.copy_from
-------------

Copies the reservoir properties to the current reservoir.

    .. function:: copy_from(self,reservoir,bind=False) -> None

    **Parameters**

        :``reservoir``: Reservoir to copy from.
        :``bind``: Shares the memory with the reservoir that is copied from, i.e. changes to one reservoir will affect the other. \
                    By default the properties from the reservoir that is copied from will be written to separate memory.

-------------------------
ESN.copy_connections_from
-------------------------

Similar to `ESN.copy_from`_ but copies only the connection matrices.

    .. function:: copy_connections_from(self,reservoir,bind=False,weights_list=None) -> None

    **Parameters**

        :``reservoir``: Reservoir to copy from.
        :``bind``: Shares the memory with the reservoir that is copied from, i.e. changes to one reservoir's connection matrices will affect the other's. \
                    By default the connection matrices from the reservoir that is copied from will be written to separate memory.
        :``weights_list``: Give a sublist of the list ``['Wout','W','Win','Wback']`` if you do not want to copy all the connections.


-------
ESN.cpu
-------

Sends the reservoir to cpu device.

    .. function:: cpu(self) -> None


--------
ESN.save
--------

Pickles the reservoir to the provided path. Save path example: ``./saved_reservoir.pkl``

    .. function:: save(self,save_path:str) -> None

    **Parameters**

        :``save_path``: Path to pickle the reservoir to.

--------
ESN.load
--------

Loads the reservoir from the provided path. Load path example: ``./saved_reservoir.pkl``

    .. function:: load(self,load_path:str) -> None

    **Parameters**

        :``load_path``: Path to load the reservoir from.




.. .. code-block::
..    :caption: A cool example

..        The output of this line starts with four spaces.


