
.. default-domain::py
.. default-role:: math

===
ESN
===

|

------------
ESN.__init__
------------

Initializes reservoir computer.

.. function:: __init__(self, \
                        W: np.ndarray=None, \
                        resSize: int=400, \
                        xn: list=[0,4,-4], \
                        pn: list=[0.9875, 0.00625, 0.00625], \
                        random_state: float=None, \
                        null_state_init: bool=True, \
                        custom_initState: np.ndarray=None, \
                        **kwargs) -> None 


|

:Parameters:  :W: User can provide custom reservoir matrix.
              :resSize: Number of units (nodes) in the reservoir.
              :xn , pn: User can provide custom random variable to control the connectivity of the reservoir. ``xn`` are the values and ``pn`` are the corresponding probabilities.
              :random_state: Fix random state. If provided, ``np.random.seed`` and ``torch.manual_seed`` are called.
              :null_state_init: If ``True``, starts the reservoir from null state. If ``False``, initializes randomly. Default is ``True``.
              :custom_initState: User can give custom initial reservoir state.

|

:Keyword Arguments:
                
                :verbose: Mute the initialization message.
                :f: User can provide custom activation function of the reservoir.
                :leak_rate: Leak parameter in Leaky Integrator ESN (LiESN).
                :leak_version: Give ``0`` for `Jaeger's recursion formula`_, give 1 for recursion formula in `ESNRLS paper`_.
                :bias: Set strength of bias in the input, reservoir and readout connections.
                :Win , Wout , Wback: User can provide custom input, output, feedback matrices.
                :use_torch: Use pytorch instead of numpy. Will use cuda if available.


.. _Jaeger's recursion formula: https://www.researchgate.net/publication/215385037_The_echo_state_approach_to_analysing_and_training_recurrent_neural_networks-with_an_erratum_note'
.. _ESNRLS paper: https://ieeexplore.ieee.org/document/9458984

|

---------------------------
ESN.scale_reservoir_weights
---------------------------

Scales the reservoir connection matrix to have certain spectral norm or radius.


.. function:: scale_reservoir_weights(self,desired_scaling: float, reference='ev') -> None


|

:Parameters:  :desired_scaling: Scales the reservoir matrix to have the desired spectral norm or radius.
              :reference: Give ``'ev'`` (eigenvalue) to choose spectral radius, ``'sv'`` (singular value) to choose spectral norm as reference.


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

:Parameters:  :u: Input. Has shape [...,time].
              :y: To be forecast. Has shape [...,time].
              :bias: Set strength of bias in the input, reservoir and readout connections.
              :f: User can provide custom activation function. Default is None. Available activations: ``'tanh'``, ``'relu'``, ``'sigmoid'``. For leaky relu activation, write ``'leaky_{leaky rate}'``, e.g. ``'leaky_0.5'``.
              :leak_rate: Leak parameter in Leaky Integrator ESN (LiESN).
              :leak_version: Give ``0`` for `Jaeger's recursion formula`_, give 1 for recursion formula in `ESNRLS paper`_.
              :initLen: Number of timesteps to be taken as initial transient tolarance. Will override initTrainLen_ratio. Will be set to an eighth of the training length if not provided.
              :trainLen: Total number of training steps. Will be set to the length of input data.
              :initTrainLen_ratio: Alternative to initLen, the user can provide the initialization period as ratio of the training length. The input ``8`` would mean that the initialization period will be an eighth of the training length.
              :wobble: For enabling noise to be added to ``y``.
              :wobbler: User can provide custom noise. Default is ``np.random.uniform(-1,1)/10000``.

|

:Keyword Arguments:
                
                     :Win: Custom input weights.
                     :Wback: Custom feedback weights.



|

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

:Parameters:  :y: Data to fit.
              :f_out_inverse: User can give custom output activation. Please give the INVERSE of the activation function. No activation is used by default.
              :regr: User can give custom regressor. Overrides other settings if provided. If not provided, will be set to scikit-learn's regressor.
              :reg_type: Regression type. Can be ``ridge`` or ``linear``. Default is ``linear``.
              :ridge_param: Regularization factor in ridge regression.
              :solver: See `scikit documentation`_.
              :error_measure: Type of error to be displayed. Can be ``'mse'`` (Mean Squared Error) or ``'mape'`` (Mean Absolute Percentage Error).

:Keyword Arguments:  :verbose: For the error message. 

                     :reg_X: Lets you overwrite ``self.reg_X`` (matrix used in regression) with a custom one of your choice. \
                        
                        .. tip:: 

                            For online training purposes, i.e. you call ``excite`` up to a certain point in your data and do a forecast at that point and repeat it at later points in your data. Instead of "exciting" reservoir states and doing regression multiple times up to these forecasts at varying points, which is inefficient since you perform same calculations repeatedly, you can excite using all data and use partial excitations, i.e. just the part of self.reg_X relevant and required for the regression.


.. _scikit documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html?highlight=ridge#sklearn.linear_model.Ridge

|

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


:Parameters:
                :u: Input series. Has shape [...,time].

                :y: To be forecast. Has shape [...,time].

                :valLen: Validation length. 
                .. note:: If ``u`` or ``y`` is provided it is not needed to be set. Mostly necessary for when neither u nor y is present.
                :f_out: Custom output activation. Default is identity.

                :output_transformer: Transforms the reservoir outputs at the very end. Default is identity.



|

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

:Parameters:
            :X_t: Training inputs. Has shape [...,time].
            :y_t: Training targets. Has shape [...,time].
            :X_v: Validation inputs. Has shape [...,time].
            :y_v: Validation targets. Has shape [...,time].
            :training_data: Data to fit to in regression. It will be set to ``y_t`` automatically if it is not provided. Either way, ``y_t`` will be used when calling ``excite``.
            :f_out_inverse: Please give the INVERSE activation function. User can give custom output activation. No activation is used by default.
            :f_out: Custom output activation. Default is identity.
            :output_transformer: Transforms the reservoir outputs at the very end. Default is identity.
            :initLen: No of timesteps to initialize the reservoir. Will override initTrainLen_ratio. Will be set to an eighth of the training length if not provided.
            :initTrainLen_ratio: Alternative to initLen, the user can provide the initialization period as ratio of the training length. An input of 8 would mean that the initialization period will be an eighth of the training length.
            :trainLen: Total no of training steps. Will be set to the length of input data, if not provided.
            :valLen: Total no of validation steps. Will be set to the length of input data, if not provided.
            :wobble_train: For enabling noise to be added to ``y_t``.
            :wobbler_train: User can provide custom noise. Default is ``np.random.uniform(-1,1)/10000``.
            :null_state_init: If ``True``, starts the reservoir from null state. If ``False``, initializes randomly. Default is ``True``.
            :custom_initState: User can give custom initial reservoir state.

:Keyword Arguments:
                    :Win , Wback: User can provide custom input, feedback matrices.
                    :f: User can provide custom activation function of the reservoir.
                    :bias: Set strength of bias in the input, reservoir and readout connections.
                    :bias_val: Set strength of bias in the input, reservoir and readout connections during validation. Default is bias used in training.
                    :f_val: User can provide custom reservoir activation function to be used during validation. Default is activation used in training.
                    :leak_rate: Leak parameter in Leaky Integrator ESN (LiESN).
                    :leak_rate_val: Leak parameter in Leaky Integrator ESN (LiESN) during validation.
                    :leak_version: Give ``0`` for `Jaeger's recursion formula`_, give 1 for recursion formula in `ESNRLS paper`_.
                    :leak_version_val: Leaking version for validation. Default is the one used in training.
                    :wobble_val: For enabling noise to be added to ``y_val`` during validation. Default is False.
                    :wobbler_val: User can provide custom noise to be added to ``y_val``. Disabled per default.
                    :train_only: Set to True to perform a training session only, i.e. no validation is done.
                    :verbose: Mute the training error messages.



.. .. code-block::
..    :caption: A cool example

..        The output of this line starts with four spaces.

