# Author: Ahmet Ege Yilmaz
# Year: 2021
# Title: Echo State Network framework

# Documentation: https://echostatenetwork.readthedocs.io/

import numpy as np
from sklearn.linear_model import Ridge,LinearRegression
import warnings
from typing import Callable, Optional
import torch
import pandas as pd
# from functools import reduce

sigmoid = lambda k: 1 / (1 + np.exp(-k))

# leaky_relu = lambda a: np.vectorize(lambda x: x if x>=0 else a*x,otypes=[np.float32,np.float64])

def leaky_relu(slope):
    def f(x):
        y1 = ((x > 0) * x)                                                 
        y2 = ((x <= 0) * x * slope)                                         
        return y1 + y2 
    return f

softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)

mse = lambda k,l: np.square(k-l).mean()

mape = lambda k,l: np.abs(1-l/k).mean()

training_type_dict = {0:"Self Feedback",1:"Output Feedback/Teacher Forced",2:"Regular/Input Driven",3:"Regular/Teacher Forced"}

validation_rule_dict = {
                        0:
                            {0:"Self-Feedback"},
                        1:
                            {0:"Output Feedback/Autonomous", 1:"Output Feedback/Teacher Forced"},
                        2:
                            {2:"Regular/Input Driven",0:"Input Feedback"},
                        3:
                            {2:"Regular/Generative",3:"Regular/Predictive"}
                        }

layer_hierarchy_dict = {'single':0,'batch':1,'ensemble':2}

# KEEP IT UPDATED
echo_state_networks_list = {'ESN','ESNX','ESNS','ESNN'}

weight_dict = {'Win':'input','W':'reservoir','Wout':'output','Wback':'feedback'}
def weight_message(weight_name:str,action:str):
    message = f'{weight_name} ({weight_dict[weight_name]} weights)'
    print(message,action+'.')
size_dict = {key:val for key,val in zip(weight_dict,['_inSize','_resSize','_outSize','_outSize'])}

settables = {'dtype':
                {'name':'_dtype','default':'float64'},
            'device':
                {'name':'_device','default':'cpu'}, 
            'bias':
                {'name':'_bias','default':None},
            'leak_rate':
                {'name':'_leak_rate','default':1},
            'leak_version':
                {'name':'_leak_version','default':0},
            'f':    
                {'name':'_f','default':'id'},
            'f_out':
                {'name':'_f_out','default':'id'},
            'resSize':
                {'name':'_resSize','default':None},
            'reservoir_layer':
                {'name':'_reservoir_layer','default':None},
            'spectral_radius':
                {'name':'_spectral_radius','default':None},
            'spectral_norm':
                {'name':'_spectral_norm','default':None},
            'W':
                {'name':'_W','default':None},
            'Win':
                {'name':'_Win','default':None},
            'Wout':
                {'name':'_Wout','default':None},
            'Wback':
                {'name':'_Wback','default':None},
            }


def esnpropertyget(fname):
    def get(esn):
        return getattr(esn,esn._properties[fname]['name'])
    return get

def esnpropertyset(fname):
    def set(esn,val):
        esn._set(fname,val=val)
    return set

class esnproperty(property):
    def __init__(self, fget, fset=None, fdel=None, doc=None):
        super().__init__(esnpropertyget(fget.__name__),esnpropertyset(fget.__name__),fdel,doc)

def at_least_2d(arr):
    if arr is None:
        return arr
    if len(arr.shape)==1:
        return arr[:,None]
    elif len(arr.shape)==2:
        return arr
    else:
        raise Exception(f"Unsupported array shape: {arr.shape}.")

def at_least_3d(arr):
    if len(arr.shape)==2:
        return arr[:,:,None] if isinstance(arr,torch.Tensor) else arr[:,:,None]
    elif len(arr.shape)==3 or arr is None:
        return arr if isinstance(arr,torch.Tensor) else arr
    else:
        raise Exception(f"Unsupported array shape: {arr.shape}.")

def Id(x):
    return x

def is_normal(x):
    if isinstance(x,torch.Tensor):
        return torch.all(torch.matmul(x,x.T.conj())==torch.matmul(x.T.conj(),x))
    elif isinstance(x,np.ndarray):
        return np.all(np.matmul(x,x.T.conj())==np.matmul(x.T.conj(),x))
    else:
        raise Exception(f"Unsupported array type.")

# class Mat(list):
#     def __matmul__(self, B):
#         A = self
#         return Mat([[sum(A[i][k]*B[k][j] for k in range(len(B)))
#                     for j in range(len(B[0])) ] for i in range(len(A))])

# A = Mat([[1,3],[7,5]])
# B = Mat([[6,8],[4,2]])

# print(A @ B)


# def updates_reservoir_layer(func):
    #     def wrapper(self,*args,**kwargs):
    #         print("Something is happening before the function is called.")
    #         func(self,*args,**kwargs)
    #         print("Something is happening after the function is called.")
    #     return wrapper

class ESN:
    
    states = None
    training_type = None
    validation_type = None
    X = None

    no_of_reservoirs = None
    batch_size = None

    _Win = None
    _W = None
    _Wout = None
    _Wback = None
    
    _get_update = None # Wont be None after initialization.
    _initLen = None
    _update_rule_id_train = None
    _update_rule_id_val = None
    _U = None

    _properties = settables

    __name = 'ESN'
    __bias = None
    __wobbler_val = None
    __y_train_last = None
    __X_train_last = None
    __bypass_rules = False
    __safe_update = True

    __stt = np.ndarray | torch.Tensor #SupportedTensorTypes
    __stt_strdict = {'numpy':'numpy array','torch':'pytorch tensor'}
    __change_unsupported = ['dtype','device','resSize']
    __os_type_dict = {'numpy':np.ndarray,'torch':torch.Tensor}

    @esnproperty
    def dtype(self):pass
    @esnproperty
    def device(self):pass
    @esnproperty
    def bias(self):pass
    @esnproperty
    def leak_rate(self):pass
    @esnproperty
    def leak_version(self):pass
    @esnproperty
    def f(self):pass
    @esnproperty
    def f_out(self):pass
    @esnproperty
    def resSize(self):pass
    @esnproperty
    def reservoir_layer(self):pass
    @esnproperty
    def spectral_radius(self):pass
    @esnproperty
    def spectral_norm(self):pass
    @esnproperty
    def W(self):pass
    @esnproperty
    def Win(self):pass
    @esnproperty
    def Wback(self):pass
    @esnproperty
    def Wout(self):pass

    """
    DOCUMENTATION
    -

    Author: Ege Yilmaz
    Year: 2021
    Title: Echo State Network class for master's thesis at ETH Zurich.
    
    Documentation: https://echostatenetwork.readthedocs.io/
    
    _
        - TRAIN: Excite the reservoir states and train the linear readout via regression.

            - Regular: With inputs. Either Teacher Forced or Input Driven.

                - Teacher Forced: Uses past version of the data to be fit during training.
                x(n+1) = (1-leak_rate) * x(n) + leak_rate*f(Win * u(n+1) + W * x(n) + Wback * y(n))

                - Input Driven: Not teacher forced. Just inputs.
                x(n+1) = (1-leak_rate) * x(n) + leak_rate*f(Win * u(n+1) + W * x(n))

            - Output Feedback: No inputs, reservoir and outputs fed back to the reservoir. Either Teacher Forced or Autonomous.
            
                - Teacher Forced: Feedback is past versions of the data to be predicted.
                x(n+1) = (1-leak_rate) * x(n) + leak_rate*f(W * x(n) + Wback * y(n))

            - Self Feedback: No inputs, no outputs. Reservoir dynamics depends on itself.
                x(n+1) = (1-leak_rate) * x(n) + leak_rate*f(W * x(n)).


        - PREDICT: Use validation input data and/or output data, which will be fed back to the reservoir as input, to do forecasts.
        Can be Generative, Predictive or with Output Feedback.

            - Regular: With Inputs.

                - Generative: Reservoir states are updated by using reservoir's outputs.
                    x(n+1) = (1-leak_rate) * x(n) + leak_rate*f(Win * u(n+1) + W * x(n) + Wback * Wout * (1;u(n);y_predicted(n-1);x(n)))

                - Predictive: Reservoir states are updated by using past versions of the data to be predicted.
                    x(n+1) = (1-leak_rate) * x(n) + leak_rate*f(Win * u(n+1) + W * x(n) + Wback * y(n))

                - Input Driven: Inputs are the only external contributers to the dynamics of the reservoir.
                    x(n+1) = (1-leak_rate) * x(n) + leak_rate*f(Win * u(n+1) + W * x(n) )
                    
            - Output Feedback: No inputs, outputs are fed back to the reservoir.

                - Teacher Forced: Feedback is the past version of the data to be predicted.
                    x(n+1) = (1-leak_rate) * x(n) + leak_rate*f(W * x(n) + Wback * y(n))

                - Autonomous: Feedback is reservoir's predictions of the past version of the data to be predicted.
                    x(n+1) = (1-leak_rate) * x(n) + leak_rate*f(W * x(n) + Wback * Wout * (1;y_predicted(n-1);x(n)))

        - TEST: Additional mode for testing training success. Reservoir is initialized from zero state with the help of teacher output for a given period 
        and then left running freely. To be implemented.

    """

    # TODO her yerde bi suru verbose var gerekliliini cek et
    def __init__(self,
                resSize: Optional[int]=400,
                xn: Optional[list[float]]=[0,0.4,-0.4],
                pn: Optional[list[float]]=[0.9875, 0.00625, 0.00625],
                random_state: Optional[float]=None,
                null_state_init: bool=True,
                custom_initState: Optional[np.ndarray]=None,
                **kwargs) -> None:
        """ 

        Description
        -
        Initialize reservoir computer.

        Variables
        -

            - resSize: Number of units in the reservoir.

            - xn , pn: User can provide custom random variable that generates sparse reservoir matrix.
            xn are the values and pn are the corresponding probabilities.

            - random_state: Fix random state.

            - null_state_init: If True, starts the reservoir from null state. If False, initializes randomly. Default is True.

            - custom_initState: User can give custom initial reservoir state x(0).

            - keyword agruments:
                
                - verbose: Mute the initialization message.
                - f: Custom activation function of the reservoir. Default is identity.
                - f_out: Custom output activation. Default is identity.
                - leak_rate: Leak parameter in Leaky Integrator ESN (LiESN). Default is 1.
                - leak_version: Give 0 for Jaeger's recursion formula, give 1 for recursion formula in ESNRLS paper.
                - bias: Strength of bias. Disabled per default.
                - W,Win,Wout,Wback
                - use_torch: Use pytorch instead of numpy. Will use cuda if available.
                - device: Give 'cpu' if use_torch is True and CUDA is available on your device but you want to use CPU.
                - dtype: Data type of reservoir. Default is float64.
        """

        W = kwargs.get('W')
        assert W is not None or isinstance(resSize,int), f"Please provide W ({weight_dict['W']} matrix) or resSize."

        self._random_state = random_state
        if self._random_state is not None:
            np.random.seed(int(random_state))

        self._verbose = True
        verbose = kwargs.get("verbose",True)
        use_torch = kwargs.get("use_torch",False)

        self.__os = 'numpy'
        self._mm = np.matmul if not hasattr(self,"_mm") else self._mm  #matrix multiplier function. diger classlarin farkli _mm lerini overridelamamak icin
        self._layer_mode = 'single' #batch, ensemble
        self._atleastND = at_least_2d
        self._xn = xn if xn is not None else [0,0.4,-0.4]
        self._pn = pn if pn is not None else [0.9875, 0.00625, 0.00625]

        
        __init__set = [to_be_set for to_be_set in self._properties if not to_be_set in ['W','resSize','dtype','spectral_radius','spectral_norm']]
        self._dtype = kwargs.get('dtype',self._properties['dtype']['default'])
        if W is None:
            self._set('W', self.make_connection('W',verbose=verbose,size=resSize),verbose=verbose)
        else:
            self._set('W',W,verbose=verbose)
            assert self.resSize == resSize, 'Specified reservoir size and the reservoir matrix are incompatible.'

        for settable in __init__set:
            default_value = self._properties[settable]['default']
            self._set(settable, kwargs.get(settable,default_value),verbose=verbose)

        if custom_initState is None:
            self._core_nodes = np.zeros((self._resSize,1),dtype=self._dtype) if null_state_init else np.random.rand(self._resSize,1).astype(self._dtype)
            self._reservoir_layer = self._core_nodes.copy() # self._core_nodes never gets changed
        else:
            assert custom_initState.shape == (self._resSize,1),f"Please give custom initial state with shape ({self._resSize},1)."
            self._core_nodes = custom_initState.copy()
            self._reservoir_layer = custom_initState.copy() # self._core_nodes never gets changed
        
        self._reservoir_layer_init = self._reservoir_layer.copy()

        if use_torch:
            self._device = kwargs.get('device',"cuda") if torch.cuda.is_available() else "cpu"
            self.__torchify()

        self.__set_updater()
        
        if verbose:
            print(f'{self.__name} generated. Number of units: {self._resSize} Spectral Radius: {self.spectral_radius}')

    def scale_reservoir_weights(self,desired_scaling: float, reference:str) -> None:

        """ 
        Description
        -
        Scales the reservoir matrix to have the desired spectral radius.

        Variables
        - desired_scaling: Desired spectral radius or spectral norm depending on the chosen reference.
        - reference: Set to 'ev' or 'sv' to scale the reservoir matrix by taking spectral radius or spectral norm as reference.

        """

        assert isinstance(desired_scaling,float | int)
        
        print(f"Scaling reservoir matrix to have spectral {bool(reference=='ev')*'radius'}{bool(reference=='sv')*'norm'} {desired_scaling}...")

        if self.__get_tensor_device(self._W) != 'cpu':
            self._W = self._W.cpu()

        if reference=='ev':
            spectral_ = self.spectral_radius
        elif reference=='sv':
            spectral_ = self.spectral_norm
        else:
            raise Exception('{reference} is unsupported.')
        
        self._W *= desired_scaling / spectral_
        self._spectral_radius = self.__spectral_radius()
        self._spectral_norm = self.__spectral_norm()
        self._W = self.__send_tensor_to_device(self._W)
        print(f'Done: spectral radius= {self.spectral_radius}, spectral norm= {self.spectral_norm}.')

    def mc(self,u,delay,initLen,trainLen,reg_type='pinv',**kwargs):

        training_data = u[:,initLen-delay:trainLen-delay]

        self.clear()
        self.evolve(u=u[:,:trainLen],initLen=initLen)
        self.fit(y=training_data,reg_type=reg_type,f_out_inverse=kwargs.get('f_out_inverse'))
        forecasts = self.predict(predLen=u.shape[1]-trainLen,u=u[:,trainLen:]).ravel()
        target = u[:,trainLen-delay:-delay].ravel()
        self.clear()
        return np.corrcoef(forecasts,target)[0,1]**2

    def reconnect_reservoir(self,xn: list[float],pn: list[float],**kwargs) -> None:
        self._xn = xn
        self._pn = pn
        self.make_connection('W',inplace=True,verbose=False)
        if kwargs.get('verbose',self._verbose):
            print('Reservoir reconnected.')

    def evolve(self,
            u: Optional[np.ndarray|torch.Tensor]=None,
            y: Optional[np.ndarray|torch.Tensor]=None,
            initLen: Optional[int|float]=None, 
            totLen: Optional[int]=None,
            wobble: bool|np.ndarray|torch.Tensor=False,
            validation: bool=False
            )-> np.ndarray|torch.Tensor:

        """

        Description
        -
        Feed the reservoir either with given inputs and/or outputs and optionally make a fit to data. Returns the excited states.

        Variables
        -
            - u: Input. Has shape [...,time].

            - y: To be predicted. Has shape [...,time].


            - initLen: No of timesteps to initialize the reservoir. Will override initTrainLen_ratio. 
                        Will be set to an eighth of the training length if not provided. 
                        Alternatively, the user can provide the initialization period as ratio of the training length.

            - trainLen: Total no of training steps. Will be set to the length of input data if not provided.

            - wobble: For enabling random noise. User can provide custom noise. Default is np.random.uniform(-1,1)/10000.

            - See 'fit' method's documentation for other (keyword) arguments.

            - validation: You can use this method after training to prepare the reservoir for validation.
        """

        X = self.__excite(u=u,
                        y=y,
                        initLen=initLen,
                        trainLen=totLen,
                        validation=validation,
                        wobble=wobble)
        
        states = X[self._inSize+self._outSize+self.__bias:,:]
        assert states.shape[0] == self._resSize
        self.X = X if self.X is None else self._cat([self.X,X],axis=1)
        self.states = states if self.states is None else self._cat([self.states,states],axis=1)
        
        return X
    
    def fit(self,
            y: np.ndarray|torch.Tensor,
            f_out_inverse:Optional[Callable]=None,
            regr: Optional[Callable]=None,
            reg_type: str="ridge",
            ridge_param: float=1e-8,
            solver: str="auto",
            error_measure: str="mse",
            **kwargs) -> np.ndarray:

        """ 
        
        Description
        -

        Trains the readout via linear or ridge regression from scikit-learn:
        - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge
        
        Returns the error of selected type.

        Variables
        -

        - y: Data to fit.

        - f_out_inverse: Please give the INVERSE activation function. User can give custom output activation. No activation is used by default.

        - regr: User can give custom regressor. Overrides other settings if provided. If not provided, will be set to scikit-learn's regressor.

        - reg_type: Regression type. Can be ridge or linear. Default is linear.

        - ridge_param: Regularization factor in ridge regression.

        - solver: See scikit documentation.

        - error_measure: Can be 'mse' or 'mape'.

        - keyword arguments:

                - verbose: For the error message.
                
                - reg_X: Lets you choose a custom matrix to be used in regression fit. For online training purposes, i.e. you "evolve" up to a certain point in your data and do a forecast at that point and continue doing this at later points in your data.
                Instead of "exciting" reservoir states multiple times up to these forecasts at varying points, which is inefficient since you perform same calculations repeatedly, you can evolve using all data and use partial excitations, i.e. just the part
                of self.X relevant and required for the regression.

        """

        assert isinstance(y,self.__os_type_dict[self.__os]), f'Please give {self.__stt_strdict[self.__os]}. type(y):{type(y)}'

        if f_out_inverse is None:
            assert self.__f_out_name == 'id', 'It seems that you are using an output activation, which is not the identity. Please provide the inverse of the activation, which is needed for the regression. In case you are using the identity function, please avoid passing it in as argument. Output activation is by default the identity.'
        else:
            assert self.__f_out_name != 'id', 'You have passed in the inverse of your output activation, yet no output activation was specified!'
        y_ = y if f_out_inverse is None else self._fn_interpreter(f_out_inverse)(y)

        reg_type_ = reg_type.lower()
        if regr is None:
            if reg_type_ == "ridge":
                regr = Ridge(ridge_param, fit_intercept=False,solver=solver)
            elif reg_type_ == "pinv":
                regr = np.linalg.pinv
            else:
                regr = LinearRegression(fit_intercept=False)

        reg_X = kwargs.get('reg_X',self.X)
        assert reg_X is not None, 'No history of reservoir layer was registered. It can be manually given using reg_X keyword argument.'
        if reg_type_ == "pinv":
            self._Wout = self._mm(y_,self.__tensor(regr(reg_X)))
        else:
            regr.fit(reg_X.transpose(-1,-2),y_.transpose(-1,-2))
            self._Wout = self.__tensor(regr.coef_)

        if error_measure == "mse":
            error = mse(y_,self._mm( self._Wout , reg_X))
        elif error_measure == "mape":
            error = mape(y_,self._mm( self._Wout , reg_X))
        else:
            raise NotImplementedError("Unknown error measure type.")
        
        if kwargs.get("verbose",self._verbose):
            print("Training ",error_measure.upper(),": ",error)

        return error

    def predict(self,
            u: Optional[np.ndarray|torch.Tensor]=None,
            y: Optional[np.ndarray|torch.Tensor]=None,
            predLen: Optional[int]=None,
            wobble: bool|np.ndarray|torch.Tensor=False,
            )-> np.ndarray|torch.Tensor:
        
        assert self._Wout is not None
        
        return self.__call__(self.evolve(u=u,
                                        y=y,
                                        totLen=predLen,
                                        wobble = wobble,
                                        validation=True
                                        )
                            )

    # TODO: ridge_param,solver etc. kwarg olmali sanki
    def session(self,
                X_t: Optional[np.ndarray|torch.Tensor]=None,
                y_t: Optional[np.ndarray|torch.Tensor]=None,
                X_v: Optional[np.ndarray|torch.Tensor]=None,
                y_v: Optional[np.ndarray|torch.Tensor]=None,
                fit_data: Optional[np.ndarray|torch.Tensor]=None,
                f_out_inverse: Optional[Callable]=None,
                initLen: Optional[int]=None, 
                trainLen: Optional[int]=None,
                predLen: Optional[int]=None,
                wobble: bool | np.ndarray | torch.Tensor=False,
                wobble_val: bool | np.ndarray | torch.Tensor=False,
                regr: Optional[Callable]=None,
                reg_type: str="ridge",
                ridge_param: float=1e-8,
                solver: str="auto",
                error_measure: str="mse",
                **kwargs
                ) -> np.ndarray|torch.Tensor:
        
        """

        Description
        -
        Executes the class methods evolve, train and validate. Returns predictions.

        Variables
        -

            - X_t: Training inputs. Has shape [...,time].

            - y_t: Training targets. Has shape [...,time].

            - X_v: Validation inputs. Has shape [...,time].

            - y_v: Validation targets. Has shape [...,time].

            - training_data: Data to be fit. It will be set to y_t automatically if it is not provided.
            
            - f_out_inverse: Please give the INVERSE activation function. User can give custom output activation. No activation is used by default.
            

            - initLen: No of timesteps to initialize the reservoir. Will override initTrainLen_ratio. 
            Will be set to an eighth of the training length if not provided.

            - initTrainLen_ratio: Alternative to initLen, the user can provide the initialization period as ratio of the training length. 
            An input of 8 would mean that the initialization period will be an eighth of the training length.

            - trainLen: Total no of training steps. Will be set to the length of input data.

            - predLen: Total no of validation steps. Will be set to the length of input data.

            - wobble: For enabling random noise.

            - wobbler: User can provide custom noise. Default is np.random.uniform(-1,1)/10000.

            - wobble_val: For enabling random noise. Default is False.
                
            - wobbler_val: User can provide custom noise. Disabled per default.

            - keyword arguments:

                - train_only: Set to True to perform a training session only, i.e. no validation is done.

                - verbose: For the training error messages.


        """
        assert y_t is not None or fit_data is not None

        self.clear()

        self.evolve(u=X_t,
                    y=y_t,
                    initLen=initLen,
                    totLen=trainLen,
                    wobble=wobble,
                    validation=False
                    )
        
        # one-step ahead prediction
        if fit_data is None:
            fit_data = y_t[:,self._initLen:]
        
        self.fit(y=fit_data,
                    f_out_inverse=f_out_inverse,
                    regr=regr,
                    reg_type=reg_type,
                    ridge_param=ridge_param,
                    solver=solver,
                    error_measure=error_measure,
                    verbose=kwargs.get("verbose",self._verbose)
                    )

        if kwargs.get("train_only"):
            return self.__call__()
        
        return self.predict(u=X_v,
                            y=y_v,
                            predLen=predLen,
                            wobble = wobble_val
                            )

    def test(self,initLen,teacherLen,autonomLen,y_test):...
    #     assert self._layer_mode == 'single'
    #     forecasts = np.zeros((y_test.shape[0],teacherLen-initLen+autonomLen)).astype(self.dtype)
    #     for t in range(1,initLen):
    #         self.update_reservoir_layer(None,y_test[:,t-1])
    #     for t in range(initLen,teacherLen):
    #         self.update_reservoir_layer(None,y_test[:,t-1])
    #         forecasts[:,t-initLen] = self.__call__(self._pack_internal_state(None,y_test[:,t-1]))
    #     for t in range(teacherLen,teacherLen+autonomLen):
    #         self.update_reservoir_layer(None,forecasts[:,t-1-initLen])
    #         forecasts[:,t-initLen] = self.__call__(self._pack_internal_state(None,forecasts[:,t-1-initLen]))
    #     return forecasts

    def update_reservoir_layer(self,
        in_:Optional[np.ndarray | torch.Tensor]=None,
        out_:Optional[np.ndarray | torch.Tensor]=None,
        get_x:Optional[bool]=False
        ) -> None:
        """
        - in_: input array
        - out_: output array
        - get_x: If set to True, it returns the resulting reservoir state with bias and output feedback (if present).
        - mode: Optional. Set to 'train' if you are updating the reservoir layer for training purposes. Set to 'val' if you are doing so for validation purposes. \
                This will allow the ESN to name the training/validation modes, which can be accessed from 'training_type' and 'val_type' attributes.
        """

        self._reservoir_layer = self._get_update(self._reservoir_layer,in_=in_,out_=out_)

        if self.__safe_update:
            self.__check_reservoir_layer()

        if get_x:
            return self._pack_internal_state(in_=in_,out_=out_)

    def update_reservoir_layers_serially(self
        , in_: Optional[np.ndarray | torch.Tensor] = None
        , out_: Optional[np.ndarray | torch.Tensor] = None
        , mode: Optional[str] = None
        ,init_size: int = 0) -> None:

        """
        WARNING: RESETS RESERVOIR LAYERS!
        """

        assert self._layer_mode != 'single', "Single reservoir layer cannot be updated serially."

        if out_ is not None:
            raise NotImplementedError

        layer_mode = self._layer_mode

        batch_size = self.batch_size

        # TODO: Make it work for randomly initialized non-null reservoir initial state.

        if layer_mode == 'batch':
            self.set_reservoir_layer_mode('single')  #(resSize,1)
            res_layer_temp = self.__send_tensor_to_device(self.__tensor(np.zeros((self._resSize,self.batch_size+init_size),dtype=self._dtype)))  #(resSize,batch_size)

        # TODO: Make it work for randomly initialized non-null reservoir initial state.
        elif self._layer_mode == 'ensemble':
            self.set_reservoir_layer_mode('single')  #(resSize,1)
            res_layer_temp = self.__send_tensor_to_device(self.__tensor(np.zeros((self.no_of_reservoirs,self._resSize,self.batch_size+init_size),dtype=self._dtype)))  #(no_of_reservoirs,resSize,batch_size)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.set_reservoir_layer_mode('ensemble',batch_size=1)  #(no_of_reservoirs,resSize,1)
            assert self._atleastND(in_).shape[0] == self.no_of_reservoirs, [in_.shape,self.no_of_reservoirs]
            
        else:
            raise Exception(f"Unsupported reservoir layer mode: '{self._layer_mode}'.")
            
        assert self._atleastND(in_).shape[-1] == batch_size + init_size, [in_.shape,batch_size,init_size]

        res_layer_temp[...,0] = self._get_update(self._reservoir_layer,self._atleastND(in_)[...,0],out_)[...,-1]
        
        for i in range(1,self.batch_size + init_size):
            res_layer_temp[...,i] = self._get_update(res_layer_temp[...,i-1:i],self._atleastND(in_)[...,i],out_)[...,-1]

        if self._layer_mode == 'ensemble':
            self.set_reservoir_layer_mode('single') #(resSize,1)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.set_reservoir_layer_mode(layer_mode,batch_size=batch_size) #(no_of_reservoirs,resSize,batch_size) or #(resSize,batch_size)

        self._reservoir_layer = res_layer_temp[...,init_size:]

    def reset_reservoir_layer(self) -> None:
        self._reservoir_layer = self.__copy(self._reservoir_layer_init)

    def set_reservoir_layer_mode(self,mode: str,batch_size: Optional[int]=None,no_of_reservoirs : Optional[int]=None):

        """
        WARNING: RESETS RESERVOIR LAYERS!
        """

        current_mode_level = layer_hierarchy_dict[self._layer_mode]
        desired_mode_level = layer_hierarchy_dict[mode]

        if desired_mode_level: #if not 'single'
            if self.batch_size is None:
                assert batch_size is not None
            else:
                if batch_size is not None and self.batch_size!=batch_size:
                    warnings.warn(f"You are changing your reservoir's batch size from {self.batch_size} to {batch_size}.")
                    self.batch_size = batch_size

        if desired_mode_level>1: #if not 'single' and not 'batch'
            if self.no_of_reservoirs is None:
                assert no_of_reservoirs is not None
                self.no_of_reservoirs = no_of_reservoirs
            else:
                if no_of_reservoirs is not None and self.no_of_reservoirs!=no_of_reservoirs:
                    warnings.warn(f"You are changing your number of reservoirs from {self.no_of_reservoirs} to {no_of_reservoirs}.")
                    self.no_of_reservoirs = no_of_reservoirs


        if current_mode_level < desired_mode_level:
            while self._layer_mode != mode:
                self._expand_reservoir_layer()
        elif current_mode_level > desired_mode_level:
            while self._layer_mode != mode:
                self._collapse_reservoir_layers()
        else:
            raise Exception(f"Current reservoir mode is already '{self._layer_mode}'.")

    def copy_from(self,reservoir,bind:bool=False,**kwargs) -> None:
        assert isinstance(reservoir,self.__class__)
        assert self.__os == reservoir._os, f'Reservoirs do not have same OS: {self.__os} != {reservoir._os}.'

        # Skipping layer_mode bcs batch and ensemble layer modes are not supported for ESN.
        self.__bypass_rules = True
        verbose = kwargs.get('verbose',True)

        for prop in self._properties:
            if not prop in ['spectral_radius','spectral_norm']:
                other_prop = getattr(reservoir,prop)
                other_prop = self.__copy(other_prop) if not bind and isinstance(other_prop, np.ndarray| torch.Tensor) else other_prop
                self._set(prop, other_prop,verbose=verbose)

        for prop in ['_verbose','_xn','_pn','_core_nodes','_reservoir_layer_init','_update_rule_id_train','_initLen','_wobbler',
                                                '_update_rule_id_val','training_type','validation_type','states','val_states','_y_train_last','reg_X','_X_val']:
            other_prop = getattr(reservoir,prop)
            other_prop = self.__copy(other_prop) if not bind and isinstance(other_prop, np.ndarray| torch.Tensor) else other_prop
            self.__setattr__(prop, other_prop)

        self.__bypass_rules = False
        self.__check_connections()
    
    def copy_connections_from(self,reservoir,bind:bool=False,weights_list: Optional[list[str]]=None,**kwargs) -> None:
        assert isinstance(reservoir,(ESN,ESNX,ESNS,ESNN))

        verbose = kwargs.get('verbose',True)

        if weights_list is None:
            weights_list = ['Wout','W','Win','Wback']
        else:
            assert isinstance(weights_list,(list,tuple))

        for w in weights_list:
            other_w = getattr(reservoir,w)
            if other_w is not None:
                other_w = other_w if bind else self.__copy(other_w)
            self._set(w, other_w,verbose=verbose)

    def make_connection(self,w_name:str,inplace:bool=False,**kwargs) -> Optional[np.ndarray | torch.Tensor]:
        w = self.__generate_weight(w_name=w_name,size=kwargs.get('size'))

        if kwargs.get('verbose',self._verbose):
            weight_message(w_name,{False:'generated',True:'got replaced'}[inplace])

        if inplace:
            self.__setattr__(w_name,w)
        else:
            return w
    
    def delete_connection(self,w_name:str,**kwargs) -> None:
        
        self.__setattr__(w_name,None)
        self.__setattr__(size_dict[w_name],None)

        if kwargs.get('verbose',self._verbose):
            print(f'{weight_dict[w_name].capitalize()} got deleted.')

    def cpu(self) -> None:
        self._device = 'cpu'
        for val in self.__dict__.values():
            if hasattr(val,'cpu'):
                val = val.cpu()
            elif hasattr(val,'to'):
                val = self.__send_tensor_to_device(val)
    
    def save(self,save_path:str) -> None:
        """
        Save path example: ./saved_reservoir.pkl
        """
        vals = [val if not hasattr(val,'cpu') else val.cpu() for val in self.__dict__.values()]
        temp = pd.Series(vals,index=self.__dict__.keys())
        temp['__class__'] = self.__class__ # str(self.__class__)[:-2].split('.')[-1]
        temp.to_pickle(save_path)
        save_file_name = save_path.split("/")[-1]
        save_loc = "/".join(save_path.split("/")[:-1]) + "/"
        print(f"{save_file_name} saved to {save_loc}.")
    
    def load(self,load_path:str) -> None:
        """
        Load path example: ./saved_reservoir.pkl
        """
        temp = pd.read_pickle(load_path)
        if not isinstance(self,temp['__class__']):
            warnings.warn(f"Loading from {temp.pop('__class__')} to {self.__class__}.")
        for attr_name,attr in temp.items():
            self.__setattr__(attr_name,attr)
        print(f"Model loaded from {load_path}.")

    def mute(self,verbose:Optional[bool]=None):
        assert isinstance(verbose,Optional[bool]) 
        if verbose is None:
            self._verbose = not self._verbose
        else:
            self._verbose = not verbose

    def clear(self):
        self._initLen = None
        self.__y_train_last = None
        self.__X_train_last = None
        self.X = None
        self.states = None
        self.training_type = None
        self.validation_type = None
        self._update_rule_id_train = None
        self._update_rule_id_val = None
        self._U = None
        self.reset_reservoir_layer()
        
    def forward(self,in_:Optional[np.ndarray | torch.Tensor]=None,out_:Optional[np.ndarray | torch.Tensor]=None) -> np.ndarray | torch.Tensor:
        return self.__call__(self.update_reservoir_layer(in_,out_,get_x=True))

    def __call__(self, x:np.ndarray | torch.Tensor=None) -> np.ndarray | torch.Tensor:
        if x is None:
            return self._f_out(self._mm(self._Wout,self.X))
        else:
            return self._f_out(self._mm(self._Wout,x))

    def _set(self,prop:str,val:float | Callable | bool,**kwargs) -> None:

        verbose = kwargs.get('verbose',self._verbose)

        # prop = prop.lower()
        assert prop in self._properties, f'{prop} is not settable.'
        _prop = self._properties[prop]['name']
        
        hasattribute = hasattr(self,_prop)
        # Setting the attribute for the first time in __init__ or brute force setting.
        if not hasattribute:
            warn = False
        elif self.__bypass_rules:
            warn = None
        # Setting the attribute NOT the first time. warn is True if attempting to change the attribute from a not None value.
        else:
            old_val = getattr(self,_prop)
            if hasattr(old_val,'shape'):
                assert old_val.shape == val.shape, f'The new {prop} must have the same shape as the old one.'
            warn = False if old_val is None else True
            assert prop not in self.__change_unsupported or warn is False, f'Changing {prop} is not supported.'

        #### Checks and warning ###
        if prop == 'f' or prop == 'f_out':
            val = self._fn_interpreter(val)
        elif prop == 'leak_rate':
            assert 0<=val<=1, 'Leak rate must lie in the interval [0,1].'
        elif prop == 'leak_version':
            assert [0,1].count(val), 'Leak version must be 0 or 1.'
        elif prop == 'bias':
            if warn is False:
                assert not hasattribute or val is None, 'Reservoir was initialized without bias. Please reinitialize to use bias.'

        assert val is not None or not warn, f'{prop} cannot be set to None.'
        ##########################

        ### Setting stuff ###
        if prop == 'spectral_radius':
            if val is not None:
                self.scale_reservoir_weights(desired_scaling=val,reference='ev')
        elif prop == 'spectral_norm':
            if val is not None:
                self.scale_reservoir_weights(desired_scaling=val,reference='sv')
        elif prop in weight_dict:
            size_str = size_dict[prop]
            size_ = 0

            if val is not None:
                assert len(val.shape)==2 and isinstance(val,self.__os_type_dict[self.__os]), 'Connection matrices must be 2D.'
                assert self.__compare_dtype(val), f"Data type of the {weight_dict[prop]} connection matrix provided by the user does not match the reservoir's data type: {self._dtype} vs. {val.dtype}.\
                                                                    To change reservoir's data type use keyword argument 'dtype' during initialization."

                size_ = val.shape[1] if prop != 'Wout' else val.shape[0]
                if prop == 'Win':
                    size_ -= self.__bias

                self.__setattr__(_prop,val)

                if prop == 'W':
                    self._spectral_radius = self.__spectral_radius()
                    self._spectral_norm = self.__spectral_norm()
            try:
                self.__setattr__(size_str,size_)
                self.__check_connections()
            except AssertionError as asserr:
                if not self.__bypass_rules:
                    print(asserr,'Assignment unsuccessful.')
                    if old_val is not None:
                        size_ = old_val.shape[1] if prop != 'Wout' else old_val.shape[0]
                        self.__setattr__(size_str,size_)
                        self.__setattr__(_prop,old_val)
                    else:
                        self.__setattr__(size_str,None)
                        self.__setattr__(_prop,None)
        else:
            self.__setattr__(_prop,val)
            if self._verbose:
                if warn:
                    warnings.warn(f"You have already been using {old_val} as {prop}. It has been changed to {val}.")
                elif verbose:
                    if val is not None:
                        print(f'{prop} has been set to {val}.')
        ####################

        ### Changing other things dependent on this thing ###
        if prop == 'bias':
            if self._verbose and self._bias == 0:
                warnings.warn("You have set the bias to zero.")
            self.__bias = False if self._bias is None else True
            self._make_bias_vec()
            if self._get_update is not None:
                self.__set_updater()
        elif prop == 'leak_version':
            if self._get_update is not None:
                self.__set_updater()
        #####################################################
    
    def _make_bias_vec(self):
        # assert self._bias != 0,'Bias equal to zero is forbidden.'

        if not self.__bias:
            self._bias_vec = None
        else:
            if self._layer_mode == 'single':
                self._bias_vec  = np.ones((1,1),dtype=self._dtype)*self._bias
            elif self._layer_mode == 'batch':
                self._bias_vec  = np.ones((1,self.batch_size),dtype=self._dtype)*self._bias
            elif self._layer_mode == 'ensemble':
                self._bias_vec  = np.ones((self.no_of_reservoirs,1,self.batch_size),dtype=self._dtype)*self._bias
            else:
                raise Exception(f"Unknown layer mode: {self._layer_mode}.")
            
            self._bias_vec = self.__send_tensor_to_device(self.__tensor(self._bias_vec))
 
    def _make_reservoir_layer(self):

        """
        WARNING: RESETS RESERVOIR LAYERS!
        """
        core_nodes = self._core_nodes.copy()
        if self._layer_mode == 'single':
            self._reservoir_layer  = core_nodes
        elif self._layer_mode == 'batch':
            self._reservoir_layer  = np.hstack(self.batch_size*[core_nodes])
        elif self._layer_mode == 'ensemble':
            self._reservoir_layer  = np.stack(self.no_of_reservoirs*[np.hstack(self.batch_size*[core_nodes])])
        else:
            raise Exception(f"Unknown layer mode: {self._layer_mode}.")

        self._reservoir_layer = self.__send_tensor_to_device(self.__tensor(self._reservoir_layer))
        self._reservoir_layer_init = self.__copy(self._reservoir_layer)

    def _collapse_reservoir_layers(self):

        """
        WARNING: RESETS RESERVOIR LAYERS!
        """

        if self._layer_mode == 'single':
            raise Exception(f'Your reservoir layer has shape {self._reservoir_layer.shape}, which cannot be collapsed further in dimension.')

        if self._layer_mode == 'batch':
            self._layer_mode = 'single'

        elif self._layer_mode == 'ensemble':
            self._layer_mode = 'batch'
            self._vstack = torch.vstack
            self._hstack = torch.hstack
            self._atleastND = at_least_2d
        else:
            raise Exception(f"Unknown layer mode: {self._layer_mode}. Needs to be one of the following: {','.join(layer_hierarchy_dict.keys())}.")
        
        self._make_bias_vec()
        self._make_reservoir_layer()
    
    def _expand_reservoir_layer(self):

        """
        WARNING: RESETS RESERVOIR LAYERS!
        """

        if self._layer_mode == 'ensemble':
            raise Exception(f'Your reservoir layer has shape {self._reservoir_layer.shape}, which cannot be expanded further in dimension.')

        if self._layer_mode == 'single':
            self._layer_mode = 'batch'

        elif self._layer_mode == 'batch':
            assert self.__os == 'torch', "To use ensemble mode, please pass in keyword argument 'use_torch=True' when initializing your Echo State Network."
            self._layer_mode = 'ensemble'
            self._vstack = lambda x: torch.cat(x,1)
            self._hstack = lambda x: torch.cat(x,2)
            self._atleastND = at_least_3d
        
        else:
            raise Exception(f"Unknown layer mode: {self._layer_mode}. Needs to be one of the following: {','.join(layer_hierarchy_dict.keys())}.")

        self._make_bias_vec()
        self._make_reservoir_layer()

    def _pack_internal_state(self,in_=None,out_=None):

        in_ = self._atleastND(in_)
        out_ = self._atleastND(out_)

        result = self.__copy(self._reservoir_layer)

        if out_ is not None:
            result = self._vstack((out_,result))
        if in_ is not None:
            result = self._vstack((in_,result))
        if self.__bias:
            result = self._vstack((self._bias_vec,result))

        return result

        # no u, no y
        if in_ is None and out_ is None:
            return self._cat((self._bias_vec,self._reservoir_layer)).ravel()

        # no u, yes y
        elif in_ is None and out_ is not None:
            return self._cat((self._bias_vec,out_,self._reservoir_layer)).ravel()

        # yes u, no y
        elif in_ is not None and out_ is None:
            return self._cat((self._bias_vec,in_,self._reservoir_layer)).ravel()

        # yes u, yes y
        elif in_ is not None and out_ is not None:
            return self._cat((self._bias_vec,in_,out_,self._reservoir_layer)).ravel()
    
    def _get_update_rule_id(self,in_=None,out_=None):
        return min(3,((in_ is not None) + 1)*((in_ is not None)+(out_ is not None)))
    
    def _fn_interpreter(self,f):
        if isinstance(f,str):
            self.__f_out_name = f
            if f.lower()=="tanh":
                return np.tanh if self.__os == 'numpy' else torch.tanh
            elif f.lower()=="sigmoid":
                return sigmoid if self.__os == 'numpy' else torch.sigmoid
            elif f.lower()=="relu":
                return leaky_relu(0) if self.__os == 'numpy' else torch.relu
            elif f.lower().startswith('leaky'):
                neg_slope = float(f.split('_')[-1])
                return leaky_relu(neg_slope) if self.__os == 'numpy' else lambda x: torch.nn.functional.leaky_relu(x,neg_slope)
            elif f.lower()=="softmax":
                return softmax if self.__os == 'numpy' else lambda x: torch.softmax(x,0,dtype=self._reservoir_layer.dtype)
            elif f.lower()=="id":
                return Id
            else:
                raise Exception("The specified activation function is not a registered one.")
        else:
            if f is None:
                return self._fn_interpreter('id')
            else:
                self.__f_out_name = 'custom'
            return f

    def _vstack(self,x,*args,**kwargs):
        if self.__os == 'numpy':
            return np.vstack(x,*args,**kwargs)
        else:
            return torch.vstack(x,*args,**kwargs)

    def _hstack(self,x,*args,**kwargs):
        if self.__os == 'numpy':
            return np.hstack(x,*args,**kwargs)
        else:
            return torch.hstack(x,*args,**kwargs)
    
    def _cat(self,x,*args,**kwargs):
        if self.__os == 'numpy':
            return np.concatenate(x,*args,**kwargs)
        else:
            return torch.cat(x,*args,**kwargs)

    def _column_stack(self,x,*args,**kwargs):
        if self.__os == 'numpy':
            return np.column_stack(x,*args,**kwargs)
        else:
            return torch.column_stack(x,*args,**kwargs )

    def __spectral_radius(self):
        if self.__os == 'numpy':
            return float(abs(np.linalg.eigvals(self._W)).max().real) #abs(linalg.eig(self._W)[0]).max()
        elif self.__os == 'torch':
            return float(torch.linalg.eigvals(self._W.cpu()).abs().max().item())
        else:
            raise Exception("Unknown os type.")

    def __spectral_norm(self):
        if self.__os == 'numpy':
            return float(np.linalg.svd(self._W,compute_uv=False).max())
        elif self.__os == 'torch':
            return float(torch.linalg.svdvals(self._W.cpu()).max().item())
        else:
            raise Exception("Unknown os type.")

    def __excite(self,
                validation: bool,
                u: Optional[np.ndarray]=None,
                y: Optional[np.ndarray]=None,
                initLen: Optional[int|float]=None, 
                trainLen: Optional[int]=None,
                wobble: bool | np.ndarray | torch.Tensor=False
                ) -> None:
        """

        Description
        -
        Stimulate reservoir states either with given inputs and/or outputs or let it evolve itself without input and output.

        Variables
        -
            - validation: You can use this method after training to prepare the reservoir for validation.

            - u: Input. Has shape [...,time].

            - y: To be predicted. Has shape [...,time].


            - initLen: No of timesteps to initialize the reservoir. Will override initTrainLen_ratio. 
                        Will be set to an eighth of the training length if not provided. 
                        Alternatively, the user can provide the initialization period as ratio of the training length.

            - trainLen: Total no of training steps. Will be set to the length of input data if not provided.
        """

        update_rule_id,initLen,trainLen,y_ = self.__prepare_for_update(in_=u,out_=y,initLen=initLen,trainLen=trainLen,validation=validation,wobble=wobble)
        assert not initLen or not validation
        # Exciting the reservoir states
        X = self.__tensor(np.zeros((self.__bias + self._resSize + self._inSize + self._outSize,trainLen-initLen),dtype=self._dtype))
        # self.__safe_update = False
        # no u, no y
        if update_rule_id == 0:
            if validation:
                if self._update_rule_id_train == 1:
                    # training was with no u, yes y. now validation with no u, yes y_pred
                    y_temp = self.__call__(self.__X_train_last) + self.__wobbler_val[:,-1]
                    for t in range(trainLen):
                        X[:,t] =  self.update_reservoir_layer(out_=y_temp,get_x=True).ravel()
                        y_temp = self.__call__(X[:,t]) + self.__wobbler_val[:,t]
                else:
                    # training was with no u, no y
                    for t in range(trainLen):
                        X[:,t] = self.update_reservoir_layer(get_x=True).ravel()
            else:
                for t in range(1,initLen):
                    self.update_reservoir_layer()
                for t in range(initLen,trainLen):
                    X[:,t-initLen] = self.update_reservoir_layer(get_x=True).ravel()
        # no u, yes y
        elif update_rule_id == 1:
            if validation:
                # no u, yes y
                y_temp = self.__y_train_last
                for t in range(trainLen):
                    X[:,t] = self.update_reservoir_layer(None,y_temp,get_x=True).ravel()
                    y_temp = y[:,t]  + self.__wobbler_val[:,t]
            else:
                for t in range(1,initLen):
                    self.update_reservoir_layer(out_=y_[:,t-1])
                for t in range(initLen,trainLen):
                    X[:,t-initLen] = self.update_reservoir_layer(out_=y_[:,t-1],get_x=True).ravel()
                self.__y_train_last = y[:,-1]
                self.__X_train_last = X[:,-1]
        # yes u, no y
        elif update_rule_id == 2:
            if validation:
                if self._update_rule_id_train == 3:
                    # yes u, yes y_pred (generative)
                    y_temp = self.__call__(self.__X_train_last) + self.__wobbler_val[:,-1]
                    for t in range(trainLen):
                        X[:,t] = self.update_reservoir_layer(in_=u[:,t],out_=y_temp,get_x=True).ravel()
                        y_temp = self.__call__(X[:,t]) + self.__wobbler_val[:,t]
                else:
                    # yes u, no y
                    for t in range(trainLen):
                        X[:,t] = self.update_reservoir_layer(in_=u[:,t],get_x=True).ravel()
            else:
                for t in range(1,initLen):
                    self.update_reservoir_layer(in_=u[:,t])
                for t in range(initLen,trainLen):
                    X[:,t-initLen] = self.update_reservoir_layer(in_=u[:,t],get_x=True).ravel()
        # yes u, yes y
        elif update_rule_id == 3:
            assert u.shape[-1] == y.shape[-1], "Inputs and outputs must have same shape at the last axis (time axis)."
            if validation:
                y_temp = self.__y_train_last + self.__wobbler_val[:,-1]
                for t in range(trainLen):
                    X[:,t] = self.update_reservoir_layer(in_=u[:,t],out_=y_temp,get_x=True).ravel()
                    y_temp = y[:,t] + self.__wobbler_val[:,t]
            else:
                for t in range(1,initLen):
                    self.update_reservoir_layer(in_=u[:,t],out_=y_[:,t-1])
                for t in range(initLen,trainLen):
                    X[:,t-initLen] = self.update_reservoir_layer(in_=u[:,t],out_=y_[:,t-1],get_x=True).ravel()
                self.__y_train_last = y[:,-1]
                self.__X_train_last = X[:,-1]
        #?
        else:
            raise NotImplementedError("Could not find a case for this training.")       

        # self.__safe_update = True
        self.__check_reservoir_layer()

        return X
    
    def __prepare_for_update(self
                    ,in_:Optional[np.ndarray | torch.Tensor]=None
                    ,out_:Optional[np.ndarray | torch.Tensor]=None
                    ,initLen: Optional[int|float]=None
                    ,trainLen: Optional[int]=None
                    ,wobble: bool | np.ndarray | torch.Tensor=False
                    ,validation: bool=False
                    ):
        
        # Update rule recognition based on I/O
        update_rule_id = self._get_update_rule_id(in_,out_)
        assert trainLen is not None or update_rule_id, 'trainLen must be provided in the case of Self-Feedback.'
        assert isinstance(in_,Optional[self.__os_type_dict[self.__os]]) and isinstance(out_,Optional[self.__os_type_dict[self.__os]]), f'Please give {Optional[self.__os_type_dict[self.__os]]}. type(u):{type(in_)} and type(y):{type(out_)}'
        assert isinstance(wobble,bool) or isinstance(wobble,self.__os_type_dict[self.__os])
        wobbler_ = 0 # This is needed such that this method can return wobbler, which is expected by __excite
        y_=None

        """
        0: both no
        1: no u yes y
        2: yes u no y
        3: both yes
        """

        #Handling I/O
        if in_ is not None:
            assert len(in_.shape) == 2
            trainLen = in_.shape[-1] if trainLen is None else trainLen
            assert self.__get_tensor_device(in_) == self._device, (self._device,in_)
            if self._Win is None:
                self.make_connection('Win',inplace=True,size=self._atleastND(in_).shape[-2])
            assert self._Win.shape[-1] == self._atleastND(in_).shape[-2]+self.__bias,[self._Win.shape,in_.shape]
        else:
            assert not self._inSize,'Please initialize the reservoir with Win only if you are going to use it.'

        if out_ is not None:
            assert len(out_.shape) == 2
            trainLen = out_.shape[-1] if trainLen is None else trainLen # Here we do not force the user to have same number of time points in u and y. Also trainLen does not have to be equal to number of time points.
            assert self.__get_tensor_device(out_) == self._device, (self._device,out_)
            if self._Wback is None:
                self.make_connection('Wback',inplace=True,size=self._atleastND(out_).shape[0])
            assert self._Wback.shape[1]==self._atleastND(out_).shape[0], "Please give input consistent with training output."
        
        # Set update_rule_ids/names for training or validation
        # Set wobbler
        if validation:
            # Check training/validation synergies
            if self._update_rule_id_train == 2 and update_rule_id == 0:
                raise Exception('Consider teacher forced training and autonomous validation.')
            assert self._update_rule_id_train == update_rule_id or self._update_rule_id_train - update_rule_id == 1 and not update_rule_id%2 \
                ,f"You trained the network in {self.training_type} mode but trying to forecast in {self.validation_type} mode."
            
            # Set update_rule_ids/names for validation
            new_val_type = validation_rule_dict[self._update_rule_id_train][update_rule_id]
            if self._update_rule_id_val is not None and self._verbose:
                warnings.warn(f"You have already performed validation of type {self.validation_type} with this reservoir. Now you are doing validation of type {new_val_type}.")
            self._update_rule_id_val = update_rule_id
            self.validation_type = new_val_type

            if self.validation_type == validation_rule_dict[0][0] and self._verbose:
                warnings.warn(f"You are forecasting in {validation_rule_dict} mode!")

            # Wobble
            assert self._update_rule_id_train % 2 or not wobble, "Wobble states are desired only in teacher forced settings."
            if wobble is True:
                self.__wobbler_val = self.__tensor(np.random.uniform(-1,1,size=(self._Wout.shape[0],trainLen)).astype(self._dtype)/10000)
            elif isinstance(wobble,self.__os_type_dict[self.__os]):
                assert wobble.shape == (self._Wout.shape[0],trainLen)
                self.__wobbler_val = self.__copy(wobble)
            else:
                self.__wobbler_val = self.__tensor(np.zeros(shape=(self._Wout.shape[0],trainLen),dtype=self._dtype))
        else:
            # Set update_rule_ids/names for training
            if self._update_rule_id_train is None:
                self._update_rule_id_train = update_rule_id
            else:
                assert self._update_rule_id_train == update_rule_id
            self.training_type = training_type_dict[update_rule_id]

            # Wobble
            assert out_ is not None or wobble is None or wobble is False ,"Wobble states are desired only in teacher forced settings."
            if wobble is True:
                wobbler_ = self.__tensor(np.random.uniform(-1,1,size=out_.shape).astype(self._dtype)/10000)
            elif isinstance(wobble,self.__os_type_dict[self.__os]):
                assert out_.shape == wobble.shape, f"Wobbler must have shape same as the output: {out_.shape} != {wobble.shape}."
                wobbler_ = self.__copy(wobble)
            y_ = None if out_ is None else out_ + wobbler_

        # Initialization and Training Lengths
        assert isinstance(trainLen,int), f"Training length must be integer. {trainLen} is given." # Yukarda trainLen'i setlemeye calistik.
        assert isinstance(initLen,Optional[int|float]),'initLen can be float between 0 and 1, a positive integer or None.'
        if initLen is None:
            initLen = 0 if validation else trainLen//8 # defaults for initLen
        elif 0<initLen<1:
            initLen *= trainLen
        else:
            assert initLen>=1 or initLen==0 and validation

        self._initLen = int(initLen)

        self.__check_connections()

        return update_rule_id,initLen,trainLen,y_
    
    def __generate_weight(self,w_name,size):
        supported = ['Win','W','Wback']
        if w_name in supported:
            size_str = size_dict[w_name]
        elif w_name in weight_dict:
            raise NotImplementedError(f'Generation of {w_name} is not supported at the moment.')
        else:
            raise Exception(f"{w_name} is not a valid weight name. Valid names are {', '.join(weight_dict.keys())}.")

        if size is None:
            size = getattr(self,size_str)
            assert size is not None

        if w_name == 'Win':
            w = np.random.rand(self._resSize,self.__bias+size) - 0.5
            # Win = np.random.uniform(size=(self._resSize,inSize+bias))<0.5
            # self._Win = np.where(Win==0, -1, Win)
        elif w_name == 'W':
            w = np.random.choice(self._xn, p=self._pn,size=(size,size))
        elif w_name == 'Wback':
            w = np.random.uniform(-2,2,size=(self._resSize,size))

        return self.__send_tensor_to_device(self.__tensor(w.astype(self._dtype)))

    def __check_connections(self):
        assert self._W.shape[0] == self._W.shape[1], f'Reservoir matrix has to be square matrix: {self._W.shape[0]} != {self._W.shape[1]}.'
        assert self._W.shape[1] == self._resSize, f'Mismatched reservoir size and reservoir weights: {self._W.shape[1]} != {self._resSize}.'
        if self._Win is not None:
            assert self._Win.shape[0] == self._W.shape[0], f'Input matrix has shape, which is inconsistent with the reservoir matrix: {self._Win.shape[0]} != {self._W.shape[0]}.'
            assert self._Win.shape[0] == self._resSize, f'Mismatched reservoir size and input weights: {self._Win.shape[0]} != {self._resSize}.'
            assert self._Win.shape[1] == self._inSize + self.__bias, f'Mismatched input size and input weights: {self._Win.shape[1]} != {self._inSize + self.__bias}.'
            if self._Wout is not None:
                assert self._Wout.shape[1] ==self._W.shape[0] + self._Win.shape[1] + self._outSize,f'Output matrix has shape, which is inconsistent with the reservoir and input matrices: {self._Wout.shape[1]} != {self._W.shape[0] + self._Win.shape[1] + self._outSize}.'
                # assert self._Wout.shape[0] == self._outSize, f'Mismatched output size and output weights: {self._Wout.shape[0]} != {self._outSize}.'
        if self._Wback is not None:
            assert self._Wback.shape[0] == self._W.shape[0],f'Feedback matrix has shape, which is inconsistent with the reservoir matrix: {self._Wback.shape[0]} != {self._W.shape[0]}.'
            assert self._Wback.shape[0] == self._resSize, f'Mismatched reservoir size and feedback weights: {self._Wback.shape[0]} != {self._resSize}.'
            assert self._Wback.shape[1] == self._outSize, f'Mismatched output size and feedback weights: {self._Wback.shape[1]} != {self._outSize}.'
            if self._Wout is not None:
                assert self._Wout.shape[0] == self._Wback.shape[1], f'Feedback matrix has shape, which is inconsistent with the output matrix: {self._Wout.shape[0]} != {self._Wback.shape[1]}.'
                # assert self._Wout.shape[0] == self._outSize, f'Mismatched output size and output weights: {self._Wout.shape[0]} != {self._outSize}.'
    
    def __check_reservoir_layer(self):

        if self.__os == 'numpy':
            _hasnan =  np.any(np.isnan(self._reservoir_layer))
        elif self.__os == 'torch':
            _hasnan = torch.any(torch.isnan(self._reservoir_layer)).item()
        else:
            raise Exception("Unknown os type.")
        
        _correctshape = self._reservoir_layer_init.shape == self._reservoir_layer.shape

        assert not _hasnan, 'NaN value encountered in reservoir layer!'
        assert _correctshape, 'Reservoir layer has changed shape!'

        # return _hasnan,_correctshape

    def __set_updater(self):

        if self.__bias:
            __updater_inPart = lambda in_: self._mm(self._Win, self._vstack((self._bias_vec,self._atleastND(in_))))
        else:
            __updater_inPart = lambda in_: self._mm(self._Win, self._atleastND(in_))

        if self._leak_version == 1:
            __updater_fullPart = lambda inPart,resPart,outPart: self._f(self._leak_rate*resPart + inPart + outPart)
        elif self._leak_version == 0:
            __updater_fullPart = lambda inPart,resPart,outPart: self._leak_rate * self._f(resPart + inPart + outPart)
        else:
            raise Exception('Unknown leak version.')

        def _updater(x:np.ndarray | torch.Tensor,
                      in_:Optional[np.ndarray | torch.Tensor]=None,
                      out_:Optional[np.ndarray | torch.Tensor]=None
                        ):

            resPart = self._mm( self._W, x )
            inPart = 0
            outPart = 0

            if in_ is not None:
                inPart = __updater_inPart(in_=in_)
            if out_ is not None:
                outPart = self._mm(self._Wback, self._atleastND(out_))
            
            return (1-self._leak_rate)*x + __updater_fullPart(inPart=inPart,resPart=resPart,outPart=outPart)
        
        self._get_update = _updater
    
    def __compare_dtype(self,x):
        return x.dtype == self._dtype or str(x.dtype).split('.')[-1] == self._dtype

    def __copy(self,x) -> np.ndarray | torch.Tensor:
        if self.__os == 'numpy':
            return x.copy()
        elif self.__os == 'torch':
            return x.clone()
        else:
            raise Exception('Unknown os.')

    def __torchify(self):

        self.__os = 'torch'

        self._mm = torch.matmul if self.__os == 'numpy' else self._mm #Dont change this

        for W_str in ['Wout','W','Win','Wback']:
            W_ = self.__getattribute__(W_str)
            if W_ is not None:
                self.__setattr__(W_str,self.__tensor(W_).to(self._device))

            if self._bias_vec is not None:
                self._bias_vec = self.__tensor(self._bias_vec).to(self._device)
            self._reservoir_layer = self.__tensor(self._reservoir_layer).to(self._device)
            self._reservoir_layer_init = self.__tensor(self._reservoir_layer_init).to(self._device)
            self._vstack = torch.vstack
            self._hstack = torch.hstack
            self._cat = torch.cat

        if self._random_state is not None:
            torch.manual_seed(int(self._random_state))

    def __tensor(self,x) -> Optional[np.ndarray | torch.Tensor]:
        assert self.__compare_dtype(x)
        if self.__os == 'numpy':
            if isinstance(x,Optional[np.ndarray]):
                return x
            elif isinstance(x,list):
                return np.array(x)
            elif isinstance(x,torch.Tensor):
                return x.cpu().numpy()
            else:
                raise NotImplementedError
        elif self.__os == 'torch':
            if isinstance(x,Optional[torch.Tensor]):
                return x
            elif isinstance(x,list):
                return torch.tensor(x)
            elif isinstance(x,np.ndarray):
                return torch.from_numpy(x)
            else:
                raise NotImplementedError

    def __get_tensor_device(self,x):
        if isinstance(x,np.ndarray):
            return 'cpu'
        elif isinstance(x,torch.Tensor):
            return x.device.type
        else:
            raise Exception("Unsupported Tensor/Array type!")

    def __send_tensor_to_device(self,x):
        if hasattr(x,'to'):
            return x.to(self._device)
        else:
            return x


class ESNX(ESN):
    """
    EchoStateNetwork X

    ESN for multitasking such as when using (mini)batches.
    """

    __name = 'ESNX'

    def __init__(self, 
                batch_size: int,
                resSize: int = 450, 
                xn: list = [0, 0.4, -0.4],
                pn: list = [0.9875, 0.00625, 0.00625], 
                random_state: float = None, 
                null_state_init: bool = True,
                custom_initState: np.ndarray = None,
                **kwargs):

        super().__init__(resSize=resSize, 
                        xn=xn, 
                        pn=pn, 
                        random_state=random_state, 
                        null_state_init=null_state_init, 
                        custom_initState=custom_initState, 
                        **kwargs)
        
        assert batch_size>1

        self.batch_size = batch_size

        self.set_reservoir_layer_mode('batch')


    def _pack_internal_state(self,in_=None,out_=None):
        raise NotImplementedError
    def __excite(self, u: np.ndarray = None, y: np.ndarray = None, bias: float = None, f: str | Callable = None, leak_rate: float = None, initLen: int = None, trainLen: int = None, initTrainLen_ratio: float = None, wobble: bool = False, wobbler: np.ndarray = None, leak_version=0, **kwargs) -> None:
        raise NotImplementedError


class ESNS(ESN):
    """
    EchoStateNetwork S

    Ensemble of ESNs for training with multiple environments using (mini)batches.
    Shape: (#Reservoirs, Vector Length, Batch Size)
    """
    
    __name = 'ESNS'

    def __init__(self, 
                no_of_reservoirs: int,
                batch_size: int,
                resSize: int = 450, 
                xn: list = [0, 0.4, -0.4],
                pn: list = [0.9875, 0.00625, 0.00625], 
                random_state: float = None, 
                null_state_init: bool = True,
                custom_initState: np.ndarray = None,
                **kwargs):


        super().__init__(resSize=resSize, 
                        xn=xn, 
                        pn=pn, 
                        random_state=random_state, 
                        null_state_init=null_state_init, 
                        custom_initState=custom_initState, 
                        batch_size=batch_size,
                        use_torch=True,
                        **kwargs)
        
        assert no_of_reservoirs>1,"Use ESNX or ESN instead."


        self.no_of_reservoirs = no_of_reservoirs
        self.batch_size = batch_size

        self.set_reservoir_layer_mode('ensemble')


    def _pack_internal_state(self,in_=None,out_=None):
        raise NotImplementedError
    def __excite(self, u: np.ndarray = None, y: np.ndarray = None, bias: float = None, f: str | Callable = None, leak_rate: float = None, initLen: int = None, trainLen: int = None, initTrainLen_ratio: float = None, wobble: bool = False, wobbler: np.ndarray = None, leak_version=0, **kwargs) -> None:
        raise NotImplementedError


class ESNN(ESN,torch.nn.Module):

    # TODO: Add output feedback support to forward.

    """
    EchoStateNetwork N

    Echo State Network as Pytorch Neural Network. Useful to train Wout via gradients.

    """


    __name = 'ESNN'


    def __init__(self,
                batch_size: int,
                in_size: int,
                out_size: int,
                no_of_reservoirs: int=None,
                resSize: int = 450,
                xn: list = [0, 0.4, -0.4],
                pn: list = [0.9875, 0.00625, 0.00625], 
                random_state: float = None, 
                null_state_init: bool = True,
                custom_initState: np.ndarray = None,
                **kwargs):

        super().__init__(
                    resSize=resSize, 
                    xn=xn, 
                    pn=pn, 
                    random_state=random_state, 
                    null_state_init=null_state_init, 
                    custom_initState=custom_initState,
                    use_torch=True,
                    **kwargs)

        torch.nn.Module.__init__(self)

        assert batch_size>1

        self.batch_size = batch_size

        if no_of_reservoirs:
            self.no_of_reservoirs=no_of_reservoirs
            self.set_reservoir_layer_mode('ensemble')
        else:
            self.set_reservoir_layer_mode('batch')


        self._inSize = in_size
        self._outSize = out_size

        self.Wout_neural = torch.nn.Linear(in_size+self._resSize+(self._bias is not None), out_size,bias=False,device=self._device,dtype=self._reservoir_layer.dtype)

    def _mm(self,a,b):
        if hasattr(a,'in_features'):
            return a(b)
        else:
            return torch.matmul(a,b)

    def forward(self,x:torch.Tensor):
        with torch.no_grad():
            self.update_reservoir_layer(x.transpose(-2,-1))
        return self.__call__(x)

    def __call__(self, in_,init_size=0):

        """
        WARNING: DOES NOT UPDATE RESERVOIR LAYER(S)!
        """

        assert self.__get_tensor_device(in_) == self._device, (self._device,in_)

        # if self._update_rule_id_train is None:
        #     self._update_rule_id_train = 2
        # else:
        #     assert self._update_rule_id_train==2


        if self.__bias:
            self._U = self._vstack((self._atleastND(in_)[...,init_size:],self._reservoir_layer,self._atleastND(self._bias_vec)))
        else:
            self._U = self._vstack((self._atleastND(in_)[...,init_size:],self._reservoir_layer))

        return self.Wout_neural(self._U.transpose(-2,-1))

    
    def _pack_internal_state(self,in_=None,out_=None):
        raise NotImplementedError
    def __excite(self, u: np.ndarray = None, y: np.ndarray = None, bias: float = None, f: str | Callable = None, leak_rate: float = None, initLen: int = None, trainLen: int = None, initTrainLen_ratio: float = None, wobble: bool = False, wobbler: np.ndarray = None, leak_version=0, **kwargs) -> None:
        raise NotImplementedError
