# Author: Ahmet Ege Yilmaz
# Year: 2021
# Title: Echo State Network framework

# Documentation: https://echostatenetwork.readthedocs.io/

import numpy as np
from sklearn.linear_model import Ridge,LinearRegression
import warnings
from typing import Any, Callable, Optional, Union
import torch
import pandas as pd
# from functools import reduce

NoneType = type(None)

sigmoid = lambda k: 1 / (1 + np.exp(-k))

leaky_relu = lambda a: np.vectorize(lambda x: x if x>=0 else a*x,otypes=[np.float32,np.float64])

softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)

mse = lambda k,l: np.square(k-l).mean()

mape = lambda k,l: np.abs(1-l/k).mean()

training_type_dict = {0:"Self Feedback",1:"Output feedback/Teacher forced",2:"Regular/Input driven",3:"Regular/Teacher forced"}

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
size_dict = {key:val for key,val in zip(weight_dict,['_inSize','resSize','_outSize','_outSize'])}

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

class ESN:

    __settable = { 'bias':
                        {'name':'_bias','default':None},
                    'leak_rate':
                        {'name':'_leak_rate','default':1},
                    'leak_version':
                        {'name':'_leak_version','default':0},
                    'f':    
                        {'name':'_f','default':'id'},
                    'f_out':
                        {'name':'_f_out','default':'id'}
                    }

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

    # def updates_reservoir_layer(func):
    #     def wrapper(self,*args,**kwargs):
    #         print("Something is happening before the function is called.")
    #         func(self,*args,**kwargs)
    #         print("Something is happening after the function is called.")
    #     return wrapper

    def __init__(self,
                resSize: int=400,
                xn: list=[0,0.4,-0.4],
                pn: list=[0.9875, 0.00625, 0.00625],
                random_state: float=None,
                null_state_init: bool=True,
                custom_initState: np.ndarray=None,
                **kwargs) -> NoneType:
        
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
                - dtype: Data type of reservoir. Default is float64.
        """

        verbose = kwargs.get("verbose",True)
        use_torch = kwargs.get("use_torch",False)

        self.dtype = kwargs.get('dtype','float64')
        self.device = 'cpu'
        self._mm = np.matmul if not hasattr(self,"_mm") else self._mm  #matrix multiplier function. diger classlarin farkli _mm lerini overridelamamak icin
        self._U = None
        self._layer_mode = 'single' #batch, ensemble
        self._atleastND = at_least_2d
        self._os = 'numpy'
        self.__xn = xn
        self.__pn = pn
        if not hasattr(self,'_name'):
            self.__name = 'ESN'

        for settable in self.__settable:
            default_value = self.__settable[settable]['default']
            self.set(settable, kwargs.get(settable,default_value),verbose=verbose)

        self._random_state = random_state
        if self._random_state:
            np.random.seed(int(random_state))

        assert kwargs.get('W') is not None or isinstance(resSize,int), f"Please provide W ({weight_dict['W']} matrix) or resSize."

        for w_name in weight_dict:
            w = kwargs.get(w_name)
            self.__setattr__(w_name,w)
            size_str = size_dict[w_name]
            size_info = None

            if w is not None:
                assert len(w.shape)==2 and isinstance(w,np.ndarray)
                assert w.dtype == self.dtype, f"Data type of the {weight_dict[w_name]} connection matrix provided by the user does not match the reservoir's data type: {self.dtype} vs. {w.dtype}.\
                                                                    To change reservoir's data type use keyword argument 'dtype' during initialization."

                size_info = w.shape[1] if w_name != 'Wout' else w.shape[0]
                if w_name == 'Win':
                    size_info -= bool(self._bias)

            self.__setattr__(size_str,size_info)
        
        if self.W is None:
            self.resSize = resSize
            self.W = self.make_connection('W')
        else:
            assert self.resSize == resSize, 'Specified reservoir size and the reservoir matrix are incompatible.'

        self.__check_connections()

        if custom_initState is None:
            self._core_nodes = np.zeros((self.resSize,1),dtype=self.dtype) if null_state_init else np.random.rand(self.resSize,1).astype(self.dtype)
            self.reservoir_layer = self._core_nodes.copy() # self._core_nodes never gets changed
        else:
            assert custom_initState.shape == (self.resSize,1),f"Please give custom initial state with shape ({self.resSize},1)."
            self._core_nodes = custom_initState.copy()
            self.reservoir_layer = custom_initState.copy() # self._core_nodes never gets changed
        
        self._reservoir_layer_init = self.reservoir_layer.copy()

        if use_torch:
            # torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device = kwargs.get('device',"cuda") if torch.cuda.is_available() else "cpu"
            self._torchify()

        self._update_rule_id_train = None
        self._update_rule_id_val = None
        self._initLen = None
        self.states = None
        self.val_states = None
        self.training_type = None
        self.validation_type = None
        self.reg_X = None
        self._X_val = None

        self.spectral_radius = self._get_spectral_radius()
        self.spectral_norm = self._get_spectral_norm()
        if verbose:
            print(f'{self.__name} generated. Number of units: {self.resSize} Spectral Radius: {self.spectral_radius}')

        self.no_of_reservoirs = None
        self.batch_size = None

    def scale_reservoir_weights(self,desired_scaling: float, reference='ev') -> NoneType:

        """ 
        Description
        -
        Scales the reservoir matrix to have the desired spectral radius.

        Variables
        - desired_scaling: Desired spectral radius or spectral norm depending on the chosen reference.
        - reference: Set to 'ev' or 'sv' to scale the reservoir matrix by taking spectral radius or spectral norm as reference.

        """

        assert isinstance(desired_scaling,float)
        
        print(f"Scaling reservoir matrix to have spectral {bool(reference=='ev')*'radius'}{bool(reference=='sv')*'norm'} {desired_scaling}...")

        if self._get_tensor_device(self.W) != 'cpu':
            self.W = self.W.cpu()

        if reference=='ev':
            self.W *= desired_scaling / self.spectral_radius
            self.spectral_radius = self._get_spectral_radius()
            self.spectral_norm = self._get_spectral_norm()
            print(f'Done: {self.spectral_radius}')
        elif reference=='sv':
            self.W *= desired_scaling / self.spectral_norm
            self.spectral_norm = self._get_spectral_norm()
            self.spectral_radius = self._get_spectral_radius()
            print(f'Done: {self.spectral_norm}')
        else:
            raise Exception('{reference} is unsupported.')
        
        self.W = self._send_tensor_to_device(self.W)

    def set(self,prop:str,val:Union[int,float,Callable,bool],verbose:bool=True) -> None:
        
        prop = prop.lower()
        assert prop in self.__settable, f'{prop} is not settable.'
        _prop = self.__settable[prop]['name']
        if not hasattr(self,_prop):
            warn = False
        else:
            warn = False if getattr(self,_prop) is None else True

        if prop == 'f' or prop == 'f_out':
            val = self._fn_interpreter(val)
        elif prop == 'leak_rate':
            assert 0<=val<=1, 'Leak rate must lie in the interval [0,1].'
        elif prop == 'leak_version':
            assert [0,1].count(val), 'Leak version must be 0 or 1.'
        elif prop == 'bias':
            if warn is False:
                assert not hasattr(self,_prop) or val is None, 'Reservoir was initialized without bias. Please reinitialize to use bias.'

        if verbose:
            if warn:
                assert val is not None, f'{prop} cannot be set to None.'
                warnings.warn(f"You have already been using {getattr(self,_prop)} as {prop}. It has been changed to {val}.")
            else:
                if val is not None:
                    print(f'{prop} has been set to {val}.')

        self.__setattr__(_prop,val)

        if prop == 'bias':
            self._make_bias_vec()

    def get(self,prop:str):
        prop = prop.lower()
        assert prop in self.__settable, f'{prop} is not gettable.'
        return self.__getattribute__(prop)

    def reconnect_reservoir(self,xn: list[Union[int,float]],pn: list[Union[int,float]],verbose:bool=True) -> None:
        self.__xn = xn
        self.__pn = pn
        self.make_connection('W',inplace=True,verbose=0)
        if verbose:
            print('Reservoir reconnected.')

    def excite(self,
                u: np.ndarray=None,
                y: np.ndarray=None,
                initLen: int=None, 
                trainLen: int=None,
                initTrainLen_ratio: float=None,
                wobble: bool=False,
                wobbler: np.ndarray=None,
                **kwargs) -> NoneType:
        """

        Description
        -
        Stimulate reservoir states either with given inputs and/or outputs or let it excite itself without input and output.

        Variables
        -

            - u: Input. Has shape [...,time].

            - y: To be predicted. Has shape [...,time].


            - initLen: No of timesteps to initialize the reservoir. Will override initTrainLen_ratio. 
            Will be set to an eighth of the training length if not provided.

            - trainLen: Total no of training steps. Will be set to the length of input data.

            - initTrainLen_ratio: Alternative to initLen, the user can provide the initialization period as ratio of the training length. 
            An input of 8 would mean that the initialization period will be an eighth of the training length.

            - wobble: For enabling random noise.

            - wobbler: User can provide custom noise. Default is np.random.uniform(-1,1)/10000.

            - keyword arguments:

                - validation_mode: You can use this method in validation mode after calling this method to prepare the reservoir for validation.
        """

        # Some stuff needs checking right out the bat.

        validation_mode = kwargs.get("validation_mode",False)
        assert bool(initLen)+bool(initTrainLen_ratio) < 2, "Please give either initLen or initTrainLen_ratio."
        assert isinstance(u,(np.ndarray,torch.Tensor,NoneType)) and isinstance(y,(np.ndarray,torch.Tensor,NoneType)), f'Please give numpy arrays or torch tensors. type(u):{type(u)} and type(y):{type(y)}'
        
        # Update rule recognition based on function inputs

        update_rule_id = self._get_update_rule_id(u,y)
        
        """
        0: both no
        1: no u yes y
        2: yes u no y
        3: both yes
        """

        #Handling I/O

        inSize = 0
        outSize = 0

        if not validation_mode:
            if self._update_rule_id_train is None:
                self._update_rule_id_train = update_rule_id
            else:
                assert self._update_rule_id_train == update_rule_id

            self.training_type = training_type_dict[update_rule_id]

            if update_rule_id % 2 - 1: #if y is None
                assert wobbler is None and not wobble ,"Wobble states are desired only in the case of teacher forced setting."
            
            if update_rule_id > 1: #if u is not None:
                # u = u.copy()
                assert len(u.shape) == 2
                inSize = u.shape[0]
                trainLen = u.shape[-1] if trainLen is None else trainLen

            if update_rule_id % 2:  #if y is not None:
                assert len(y.shape) == 2
                outSize = y.shape[0]
                trainLen = y.shape[-1] if trainLen is None else trainLen

                """Wobbler"""
                assert isinstance(wobble,bool),"wobble parameter must be boolean."
                if wobbler is not None:
                    assert y.shape == wobbler, "Wobbler must have shape same as the output."
                    self._wobbler = wobbler
                elif wobble:
                    self._wobbler = self._tensor(np.random.uniform(-1,1,size=y.shape).astype(self.dtype)/10000)
                else:
                    self._wobbler = 0
                y_ = y + self._wobbler

        else:
            new_val_type = validation_rule_dict[self._update_rule_id_train][update_rule_id]
            if self._update_rule_id_val is None:
                self._update_rule_id_val = update_rule_id
            else:
                warnings.warn(f"You have already performed validation of type {self.validation_type} with this reservoir. Now you are switching to validation of type {new_val_type}.")
                #assert self._update_rule_id_val == update_rule_id
            
            self.validation_type = new_val_type

            if self.validation_type == validation_rule_dict[0][0]:
                warnings.warn(f"You are forecasting in {validation_rule_dict} mode!")

            if (self._update_rule_id_train - update_rule_id) == 1 and update_rule_id%2 == 0:
                pass
            elif self._update_rule_id_train == 2 and update_rule_id==0:
                pass
            else:
                assert self._update_rule_id_train == update_rule_id \
                    ,f"You trained the network in {self.training_type} mode but trying to forecast in {self.validation_type} mode."
            inSize = 0 if self._inSize is None else self._inSize
            outSize = 0 if self._outSize is None else self._outSize



        # Initialization and Training Lengths
        assert initTrainLen_ratio is None or initTrainLen_ratio >= 1, "initTrainLen_ratio must be larger equal than 1."
        if initLen is None:
            initLen = trainLen//initTrainLen_ratio if initTrainLen_ratio else trainLen//8
        assert initLen >= 1 or validation_mode
        self._initLen = initLen

        # Exciting the reservoir states
        assert isinstance(trainLen,int), f"Training length must be integer.{trainLen} is given."
        X = self._tensor(np.zeros((bool(self._bias)+self.resSize + inSize + outSize,trainLen-initLen),dtype=self.dtype))

        # no u, no y
        if update_rule_id == 0:
            if validation_mode:
                if self._update_rule_id_train == 1:
                    # training was with no u, yes y. now validation with no u, yes y_pred
                    y_temp = self._f_out(self._mm(self.Wout, self.reg_X[:,-1]))
                    for t in range(trainLen):
                        self.update_reservoir_layer(None,y_temp)
                        X[:,t] =  self._pack_internal_state(None,y_temp)
                        y_temp = self._f_out(self._mm(self.Wout, X[:,t])) + self._wobbler[:,t]
                elif self._update_rule_id_train == 2:
                    # training was with yes u, no y. now validation with yes u_pred, no y
                    # This is only useful when input data and output data differ by a phase.
                    u_temp = self._f_out(self._mm(self.Wout, self.reg_X[:,-1]))
                    for t in range(trainLen):
                        self.update_reservoir_layer(u_temp,None)
                        X[:,t] = self._pack_internal_state(u_temp)
                        u_temp = self._f_out(self._mm(self.Wout, X[:,t]))
                else:
                    # training was with no u, no y
                    for t in range(trainLen):
                        self.update_reservoir_layer()
                        X[:,t] = self._pack_internal_state()
            else:
                for t in range(1,trainLen):
                    self.update_reservoir_layer()
                    if t >= initLen:
                        X[:,t-initLen] = self._pack_internal_state()
        # no u, yes y
        elif update_rule_id == 1:
            if validation_mode:
                # no u, yes y
                y_temp = self._y_train_last
                for t in range(trainLen):
                    self.update_reservoir_layer(None,y_temp)
                    X[:,t-initLen] = self._pack_internal_state(None,y_temp)
                    y_temp = y[:,t]  + self._wobbler[:,t]
            else:
                for t in range(1,trainLen):
                    self.update_reservoir_layer(None,y_[:,t-1])
                    if t >= initLen:
                        X[:,t-initLen] = self._pack_internal_state(None,y_[:,t-1])
                self._outSize = outSize
                self._y_train_last = y_[:,-1]
        # yes u, no y
        elif update_rule_id == 2:
            if validation_mode:
                if self._update_rule_id_train == 3:
                    # yes u, yes y_pred (generative)
                    y_temp = self._f_out(self._mm(self.Wout, self.reg_X[:,-1]))
                    for t in range(trainLen):
                        self.update_reservoir_layer(u[:,t],y_temp)
                        X[:,t] = self._pack_internal_state(u[:,t],y_temp)
                        y_temp = self._f_out(self._mm(self.Wout, X[:,t])) + self._wobbler[:,t]
                else:
                    # yes u, no y
                    for t in range(trainLen):
                        self.update_reservoir_layer(u[:,t])
                        X[:,t] = self._pack_internal_state(u[:,t])
            else:
                for t in range(1,trainLen):
                    self.update_reservoir_layer(u[:,t],None)
                    if t >= initLen:
                        X[:,t-initLen] = self._pack_internal_state(u[:,t])
                self._inSize = inSize
        # yes u, yes y
        elif update_rule_id == 3:
            assert u.shape[-1] == y.shape[-1], "Inputs and outputs must have same shape at the last axis (time axis)."
            if validation_mode:
                y_temp = self._y_train_last
                for t in range(trainLen):
                    self.update_reservoir_layer(u[:,t],y_temp)
                    X[:,t] = self._pack_internal_state(u[:,t],y_temp)
                    y_temp = y[:,t] + self._wobbler[:,t]
            else:
                for t in range(1,trainLen):
                    self.update_reservoir_layer(u[:,t],y_[:,t-1])
                    if t >= initLen:
                        X[:,t-initLen] = self._pack_internal_state(u[:,t],y_[:,t-1])
                self._inSize = inSize
                self._outSize = outSize
                self._y_train_last = y_[:,-1]
        #?
        else:
            raise NotImplementedError("Could not find a case for this training.")       

        
        states = X[inSize+outSize+bool(self._bias):,:]
        assert states.shape[0] == self.resSize
        if validation_mode:
            self.val_states = states
            self._X_val = X
        else:
            self.reg_X = X if self.reg_X is None else self._cat([self.reg_X,X],axis=1)
            self.states = states if self.states is None else self._cat([self.states,states],axis=1)

    def reg_fit(self,
                y: np.ndarray,
                f_out_inverse=None,
                regr=None,
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
                
                - reg_X: Lets you overwrite self.reg_X (matrix used in regression fit) with a custom one of your choice. For online training purposes, i.e. you "excite" up to a certain point in your data and do a forecast at that point and continue doing this at later points in your data.
                Instead of "exciting" reservoir states multiple times up to these forecasts at varying points, which is inefficient since you perform same calculations repeatedly, you can excite using all data and use partial excitations, i.e. just the part
                of self.reg_X relevant and required for the regression.

        """

        assert isinstance(y,(np.ndarray,torch.Tensor)), f'Please give numpy array or torch tensor. type(y):{type(y)}'

        if regr is None:
            if reg_type.lower() == "ridge":
                regr = Ridge(ridge_param, fit_intercept=False,solver=solver)
            else:
                regr = LinearRegression(fit_intercept=False)
        
        if f_out_inverse is None:
            assert self.__f_out_name == 'id', 'It seems that you are using an output activation, which is not the identity. Please provide the inverse of the activation, which is needed for the regression. In case you are using the identity function, please avoid passing it in as argument. Output activation is by default the identity.'
        y_ = y if f_out_inverse is None else self._fn_interpreter(f_out_inverse)(y)

        self.reg_X = kwargs.get("reg_X",self.reg_X)
        assert self.reg_X is not None, 'No history of reservoir layer was registered. It can be manually given using reg_X keyword argument.'
        regr.fit(self.reg_X.transpose(-1,-2),y_.transpose(-1,-2))
        self.Wout = self._tensor(regr.coef_)

        if error_measure == "mse":
            error = mse(y_,self._mm( self.Wout , self.reg_X))
            self.mse_train = error
        elif error_measure == "mape":
            error = mape(y_,self._mm( self.Wout , self.reg_X))
            self.mape_train = error
        else:
            raise NotImplementedError("Unknown error measure type.")
        
        if kwargs.get("verbose",1):
            print("Training ",error_measure.upper(),": ",error)

        return error
   
    def validate(self,
                u: np.ndarray=None,
                y: np.ndarray=None,
                valLen: int=None,
                **kwargs) -> np.ndarray:

        """
        Returns prediction.


        -VARIABLES-

        u: input

        y: to be predicted

        - valLen: Validation length. If u or y is provided it is not needed to be set. Mostly necessary for when neither u nor y is present.

        - f_out: Custom output activation. Default is identity.
        
        - keyword arguments:


            - wobble: For enabling random noise. Default is False.

            - wobbler: User can provide custom noise. Disabled per default.

        """

        assert self.Wout is not None
        assert isinstance(u,(np.ndarray,torch.Tensor,NoneType)) and isinstance(y,(np.ndarray,torch.Tensor,NoneType)), f'Please give numpy arrays or torch tensors. type(u):{type(u)} and type(y):{type(y)}'

        assert u is not None or y is not None or valLen is not None, 'valLen is needed to know how many steps the reservoir should generate outputs.'
        
        if u is not None:
            assert self._inSize == u.shape[0], "Please give input consistent with training input."
        if y is not None:
            assert self._outSize == y.shape[0], "Please give output consistent with training output."

        if u is not None:
            valLen = u.shape[-1]
        elif y is not None:
            valLen = y.shape[-1]
        else:
            valLen=valLen

        # Wobbler
        wobble = kwargs.get("wobble",False)
        wobbler = kwargs.get("wobbler")
        assert self._update_rule_id_train % 2 or not wobble
        assert wobbler is None or wobble
        if wobble and wobbler is None:
           self._wobbler = np.random.uniform(-1,1,size=(self.Wout.shape[0],valLen)).astype(self.dtype)/10000
        elif wobbler is not None:
            self._wobbler = wobbler
        else:
            self._wobbler = np.zeros(shape=(self.Wout.shape[0],valLen),dtype=self.dtype)

        self.excite(u, y, initLen=0,trainLen=valLen,wobble=wobble,wobbler=self._wobbler,validation_mode=True)

        return self._f_out(self._mm(self.Wout, self._X_val))

    def session(self,
                X_t: np.ndarray=None,
                y_t: np.ndarray=None,
                X_v: np.ndarray=None,
                y_v: np.ndarray=None,
                training_data: np.ndarray=None,
                f_out_inverse=None,
                initLen: int=None, 
                initTrainLen_ratio: float=None,
                trainLen: int=None,
                valLen: int=None,
                wobble_train: bool=False,
                wobbler_train: np.ndarray=None,
                null_state_init: bool=True,
                custom_initState: np.ndarray=None,
                regr=None,
                reg_type: str="ridge",
                ridge_param: float=1e-8,
                solver: str="auto",
                error_measure: str="mse",
                **kwargs
                ) -> np.ndarray:
        
        """

        Description
        -
        Executes the class methods excite, train and validate. Returns predictions.

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

            - valLen: Total no of validation steps. Will be set to the length of input data.

            - wobble_train: For enabling random noise.

            - wobbler_train: User can provide custom noise. Default is np.random.uniform(-1,1)/10000.

            - null_state_init: If True, starts the reservoir from null state. If False, starts them randomly. Default is True.

            - custom_initState: User can give custom initial reservoir state x(0).

            - keyword arguments:

                - wobble_val: For enabling random noise. Default is False.
                
                - wobbler_val: User can provide custom noise. Disabled per default.

                - train_only: Set to True to perform a training session only, i.e. no validation is done.

                - verbose: For the training error messages.


        """
        assert y_t is not None or training_data is not None

        training_data = y_t if y_t is not None else training_data

        self._inSize = None
        self._outSize = None

        self.reg_X = None
        self._X_val = None
        self.Wout = None
        self.states = None
        self.val_states = None
        self._update_rule_id_train = None
        self._update_rule_id_val = None

        if custom_initState is None:
            self.reservoir_layer = np.zeros((self.resSize,1),dtype=self.dtype) if null_state_init else np.random.rand(self.resSize,1).astype(self.dtype)
        else:
            assert custom_initState.shape == (self.resSize,1),f"Please give custom initial state with shape ({self.resSize},1)."
            self.reservoir_layer = custom_initState

        if self._mm != np.matmul:
            self._torchify()

        self.excite(u=X_t,
                    y=y_t,
                    initLen=initLen,
                    trainLen=trainLen,
                    initTrainLen_ratio=initTrainLen_ratio,
                    wobble=wobble_train,
                    wobbler=wobbler_train
                    )
        
        self.reg_fit(y=training_data[:,self._initLen:],
                    f_out_inverse=f_out_inverse,
                    regr=regr,
                    reg_type=reg_type,
                    ridge_param=ridge_param,
                    solver=solver,
                    error_measure=error_measure,
                    verbose=kwargs.get("verbose",1)
                    )

        if kwargs.get("train_only"):
            return self._mm( self.Wout , self.reg_X)

        pred = self.validate(u=X_v,
                    y=y_v,
                    valLen=valLen,
                    wobble = kwargs.get("wobble_val",False),
                    wobbler = kwargs.get("wobbler_val")
                    )
        
        return pred

    def test(self):
        "TBD"
        pass

    def update_reservoir_layer(
        self,in_:Union[np.ndarray,torch.Tensor,NoneType]=None
        ,out_:Union[np.ndarray,torch.Tensor,NoneType]=None
        ,mode:Optional[str]=None) -> NoneType:
        """
        - in_: input array
        - out_: output array
        - mode: Optional. Set to 'train' if you are updating the reservoir layer for training purposes. Set to 'val' if you are doing so for validation purposes. \
                This will allow the ESN to name the training/validation modes, which can be accessed from 'training_type' and 'val_type' attributes.
        """
        
        self._update_rule_id_check(in_,out_,mode)

        self.reservoir_layer = self._get_update(self.reservoir_layer,in_=in_,out_=out_)

    def update_reservoir_layers_serially(self
        , in_: Union[np.ndarray, torch.Tensor, NoneType] = None
        , out_: Union[np.ndarray, torch.Tensor, NoneType] = None
        , mode: Optional[str] = None
        ,init_size: int = 0) -> NoneType:

        """
        WARNING: RESETS RESERVOIR LAYERS!
        """

        assert self._layer_mode != 'single', "Single reservoir layer cannot be updated serially."

        if out_ is not None:
            raise NotImplementedError

        self._update_rule_id_check(in_,out_,mode)

        layer_mode = self._layer_mode

        batch_size = self.batch_size

        # TODO: Make it work for randomly initialized non-null reservoir initial state.

        if layer_mode == 'batch':
            self.set_reservoir_layer_mode('single')  #(resSize,1)
            res_layer_temp = self._send_tensor_to_device(self._tensor(np.zeros((self.resSize,self.batch_size+init_size),dtype=self.dtype)))  #(resSize,batch_size)

        # TODO: Make it work for randomly initialized non-null reservoir initial state.
        elif self._layer_mode == 'ensemble':
            self.set_reservoir_layer_mode('single')  #(resSize,1)
            res_layer_temp = self._send_tensor_to_device(self._tensor(np.zeros((self.no_of_reservoirs,self.resSize,self.batch_size+init_size),dtype=self.dtype)))  #(no_of_reservoirs,resSize,batch_size)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.set_reservoir_layer_mode('ensemble',batch_size=1)  #(no_of_reservoirs,resSize,1)
            assert self._atleastND(in_).shape[0] == self.no_of_reservoirs, [in_.shape,self.no_of_reservoirs]
            
        else:
            raise Exception(f"Unsupported reservoir layer mode: '{self._layer_mode}'.")
            
        assert self._atleastND(in_).shape[-1] == batch_size + init_size, [in_.shape,batch_size,init_size]

        res_layer_temp[...,0] = self._get_update(self.reservoir_layer,self._atleastND(in_)[...,0],out_)[...,-1]
        
        for i in range(1,self.batch_size + init_size):
            res_layer_temp[...,i] = self._get_update(res_layer_temp[...,i-1:i],self._atleastND(in_)[...,i],out_)[...,-1]

        if self._layer_mode == 'ensemble':
            self.set_reservoir_layer_mode('single') #(resSize,1)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.set_reservoir_layer_mode(layer_mode,batch_size=batch_size) #(no_of_reservoirs,resSize,batch_size) or #(resSize,batch_size)

        self.reservoir_layer = res_layer_temp[...,init_size:]

    def reset_reservoir_layer(self) -> NoneType:
        if self._mm == np.matmul:
            self.reservoir_layer = self._reservoir_layer_init.copy()
        else:
            self.reservoir_layer = self._reservoir_layer_init.clone()

    def set_reservoir_layer_mode(self,mode: str,batch_size: int=None,no_of_reservoirs :int=None):

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

    def copy_from(self,reservoir,bind=False) -> NoneType:
        assert isinstance(reservoir,self.__class__)
        # assert reservoir._mm == self._mm, f"{reservoir} is using {str(reservoir._mm).split('.')[0]}, whereas {self} is using {str(self._mm).split('.')[0]}."
        for attr_name,attr in reservoir.__dict__.items():
            if isinstance(attr,(int,float,NoneType,np.ufunc,np.vectorize,type(sigmoid),type(torch.tanh),type(torch.nn.functional.leaky_relu))) or bind:
                self.__setattr__(attr_name,attr)
            else:
                if isinstance(attr,torch.Tensor):
                    self.__setattr__(attr_name,attr.clone())
                else:
                    self.__setattr__(attr_name,attr.copy())
    
    def copy_connections_from(self,reservoir,bind=False,weights_list=None) -> NoneType:
        assert isinstance(reservoir,(ESN,ESNX,ESNS,ESNN))

        if weights_list is None:
            weights_list = ['Wout','W','Win','Wback']
        else:
            assert isinstance(weights_list,(list,tuple))

        for attr_name,attr in reservoir.__dict__.items():
            if weights_list.count(attr_name):
                if isinstance(attr,NoneType) or bind:
                    self.__setattr__(attr_name,attr)
                else:
                    if reservoir._mm == np.matmul:
                        self.__setattr__(attr_name,self._tensor(attr.copy()))
                    else:
                        self.__setattr__(attr_name,self._tensor(attr.clone()))

    def make_connection(self,w_name:str,inplace:bool=False,verbose:bool=True,**kwargs) -> Union[np.ndarray,torch.tensor,NoneType]:
        w = self.__generate_weight(w_name=w_name,size=kwargs.get('size'))

        if verbose:
            weight_message(w_name,{False:'generated',True:'got replaced'}[inplace])

        if inplace:
            self.__setattr__(w_name,w)
        else:
            return w
    
    def delete_connection(self,w_name:str,verbose:bool=True) -> NoneType:
        
        self.__setattr__(w_name,None)
        self.__setattr__(size_dict[w_name],None)

        if verbose:
            print(f'{weight_dict[w_name].capitalize()} got deleted.')

    def cpu(self) -> NoneType:
        self.device = 'cpu'
        for val in self.__dict__.values():
            if hasattr(val,'cpu'):
                val = val.cpu()
            elif hasattr(val,'to'):
                val = self._send_tensor_to_device(val)
    
    def save(self,save_path:str) -> NoneType:
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
    
    def load(self,load_path:str) -> NoneType:
        """
        Load path example: ./saved_reservoir.pkl
        """
        temp = pd.read_pickle(load_path)
        if not isinstance(self,temp['__class__']):
            warnings.warn(f"Loading from {temp.pop('__class__')} to {self.__class__}.")
        for attr_name,attr in temp.items():
            self.__setattr__(attr_name,attr)
        print(f"Model loaded from {load_path}.")

    def _get_spectral_radius(self):
        if self._os == 'numpy':
            return abs(np.linalg.eigvals(self.W)).max().real #abs(linalg.eig(self.W)[0]).max()
        elif self._os == 'torch':
            return torch.linalg.eigvals(self.W.cpu()).abs().max().item()
        else:
            raise Exception("Something is terribly wrong.")

    def _get_spectral_norm(self):
        if self._os == 'numpy':
            return np.linalg.svd(self.W,compute_uv=False).max()
        elif self._os == 'torch':
            return torch.linalg.svdvals(self.W.cpu()).max().item()
        else:
            raise Exception("Something is terribly wrong.")

    def _get_update(self
                    ,x,in_:Union[np.ndarray,torch.tensor,NoneType]=None
                    ,out_:Union[np.ndarray,torch.tensor,NoneType]=None
                    ):

        if self._os == 'torch':
            assert isinstance(in_,(torch.Tensor,NoneType)) and isinstance(out_,(torch.Tensor,NoneType)), 'Please give pytorch tensors.'

        assert self.W.shape[-1]==x.shape[-2], [self.W.shape,x.shape]
        resPart = self._f(self._mm( self.W, x ))
        inPart = 0
        outPart = 0

        if in_ is not None:
            assert self._get_tensor_device(in_) == self.device, (self.device,in_)
            if self.Win is None:
                self.make_connection('Win',inplace=True,size=self._atleastND(in_).shape[-2])
            assert self.Win.shape[-1] == self._atleastND(in_).shape[-2]+bool(self._bias),[self.Win.shape,in_.shape]
            if self._bias is not None:
                inPart = self._mm(self.Win, self._vstack((self._bias_vec,self._atleastND(in_))))
            else:
                inPart = self._mm(self.Win, self._atleastND(in_))

        if out_ is not None:
            if self.Wback is None:
                self.make_connection('Wback',inplace=True,size=self._atleastND(out_).shape[0])

            assert self._get_tensor_device(out_) == self.device, (self.device,out_)
            
            assert self.Wback.shape[1]==self._atleastND(out_).shape[0]

            outPart = self._mm(self.Wback, self._atleastND(out_))


        if self._leak_version == 1:
            resPart *= self._leak_rate
            fullPart = self._f(resPart + inPart + outPart)
        elif self._leak_version == 0:
            fullPart = self._leak_rate * self._f(resPart + inPart + outPart)
        else:
            raise Exception('Unknown leak version.')
        
        return (1-self._leak_rate)*x + fullPart
    
    def _make_bias_vec(self):
        # assert self._bias != 0,'Bias equal to zero is forbidden.'

        if self._bias is None:
            self._bias_vec = None
        else:
            if self._layer_mode == 'single':
                self._bias_vec  = np.ones((1,1),dtype=self.dtype)*self._bias
            elif self._layer_mode == 'batch':
                self._bias_vec  = np.ones((1,self.batch_size),dtype=self.dtype)*self._bias
            elif self._layer_mode == 'ensemble':
                self._bias_vec  = np.ones((self.no_of_reservoirs,1,self.batch_size),dtype=self.dtype)*self._bias
            else:
                raise Exception(f"Unknown layer mode: {self._layer_mode}.")
            
            self._bias_vec = self._send_tensor_to_device(self._tensor(self._bias_vec))
 
    def _make_reservoir_layer(self):

        """
        WARNING: RESETS RESERVOIR LAYERS!
        """
        core_nodes = self._core_nodes.copy()
        if self._layer_mode == 'single':
            self.reservoir_layer  = core_nodes
        elif self._layer_mode == 'batch':
            self.reservoir_layer  = np.hstack(self.batch_size*[core_nodes])
        elif self._layer_mode == 'ensemble':
            self.reservoir_layer  = np.stack(self.no_of_reservoirs*[np.hstack(self.batch_size*[core_nodes])])
        else:
            raise Exception(f"Unknown layer mode: {self._layer_mode}.")

        self.reservoir_layer = self._send_tensor_to_device(self._tensor(self.reservoir_layer))
        self._reservoir_layer_init = self._get_clone(self.reservoir_layer)

    def _collapse_reservoir_layers(self):

        """
        WARNING: RESETS RESERVOIR LAYERS!
        """

        if self._layer_mode == 'single':
            raise Exception(f'Your reservoir layer has shape {self.reservoir_layer.shape}, which cannot be collapsed further in dimension.')

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
            raise Exception(f'Your reservoir layer has shape {self.reservoir_layer.shape}, which cannot be expanded further in dimension.')

        if self._layer_mode == 'single':
            self._layer_mode = 'batch'

        elif self._layer_mode == 'batch':
            assert self._os == 'torch', "To use ensemble mode, please pass in keyword argument 'use_torch=True' when initializing your Echo State Network."
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

        result = self.reservoir_layer.copy()

        if out_ is not None:
            result = self._cat((out_,result))
        if in_ is not None:
            result = self._cat((in_,result))
        if self._bias is not None:
            result = self._cat((self._bias_vec,result))

        return result.ravel()

        # no u, no y
        if in_ is None and out_ is None:
            return self._cat((self._bias_vec,self.reservoir_layer)).ravel()

        # no u, yes y
        elif in_ is None and out_ is not None:
            return self._cat((self._bias_vec,out_,self.reservoir_layer)).ravel()

        # yes u, no y
        elif in_ is not None and out_ is None:
            return self._cat((self._bias_vec,in_,self.reservoir_layer)).ravel()

        # yes u, yes y
        elif in_ is not None and out_ is not None:
            return self._cat((self._bias_vec,in_,out_,self.reservoir_layer)).ravel()
    
    def _get_update_rule_id(self,in_=None,out_=None):
        return min(3,(bool(in_ is not None) + 1)*(bool(in_ is not None)+bool(out_ is not None)))
    
    def _update_rule_id_check(self,in_,out_,mode):
        #assert len(self.reservoir_layer.shape)>1 and self.reservoir_layer.shape[1]==1,self.reservoir_layer.shape

        if mode == "train":
            update_rule_id = self._get_update_rule_id(in_,out_)
            if self._update_rule_id_train is None:
                self._update_rule_id_train = update_rule_id   
            else:
                assert update_rule_id == self._update_rule_id_train
        elif mode == "val":
            update_rule_id = self._get_update_rule_id(in_,out_)
            if self._update_rule_id_val is None:
                self._update_rule_id_val = update_rule_id   
            else:
                assert update_rule_id == self._update_rule_id_val
        else:
            assert mode is None,"You have given unsupported input for the 'mode' argument."

    def _fn_interpreter(self,f):
        if isinstance(f,str):
            self.__f_out_name = f
            if f.lower()=="tanh":
                return np.tanh if self._os == 'numpy' else torch.tanh
            elif f.lower()=="sigmoid":
                return sigmoid if self._os == 'numpy' else torch.sigmoid
            elif f.lower()=="relu":
                return leaky_relu(0) if self._os == 'numpy' else torch.relu
            elif f.lower().startswith('leaky'):
                neg_slope = float(f.split('_')[-1])
                return leaky_relu(neg_slope) if self._os == 'numpy' else lambda x: torch.nn.functional.leaky_relu(x,neg_slope)
            elif f.lower()=="softmax":
                return softmax if self._os == 'numpy' else lambda x: torch.softmax(x,0,dtype=self.reservoir_layer.dtype)
            elif f.lower()=="id":
                return Id
            else:
                raise Exception("The specified activation function is not a registered one.")
        else:
            self.__f_out_name = 'custom'
            return f

    def _vstack(self,x):
        if self._mm == np.matmul:
            return np.vstack(x)
        else:
            return torch.vstack(x)

    def _hstack(self,x):
        if self._mm == np.matmul:
            return np.hstack(x)
        else:
            return torch.hstack(x)
    
    def _cat(self,x):
        if self._mm == np.matmul:
            return np.concatenate(x)
        else:
            return torch.cat(x)

    def _column_stack(self,x):
        if self._mm == np.matmul:
            return np.column_stack(x)
        else:
            return torch.column_stack(x)            

    def _tensor(self,x):
        assert x.dtype == self.dtype or str(x.dtype).split('.')[-1] == self.dtype
        if self._mm == np.matmul:
            if isinstance(x,(np.ndarray,NoneType)):
                return x
            elif isinstance(x,list):
                return np.array(x)
            elif isinstance(x,torch.Tensor):
                return x.cpu().numpy()
            else:
                raise NotImplementedError
        else:
            if isinstance(x,(torch.Tensor,NoneType)):
                return x
            elif isinstance(x,list):
                return torch.tensor(x)
            elif isinstance(x,np.ndarray):
                return torch.from_numpy(x)
            else:
                raise NotImplementedError

    def _get_clone(self,x):
        if self._mm == np.matmul:
            return x.copy()
        else:
            return x.clone()

    def _torchify(self):

        self._mm = torch.matmul if self._mm == np.matmul else self._mm #Dont change this

        for W_str in ['Wout','W','Win','Wback']:
            W_ = self.__getattribute__(W_str)
            if W_ is not None:
                self.__setattr__(W_str,self._tensor(W_).to(self.device))

            if self._bias_vec is not None:
                self._bias_vec = self._tensor(self._bias_vec).to(self.device)
            self.reservoir_layer = self._tensor(self.reservoir_layer).to(self.device)
            self._reservoir_layer_init = self._tensor(self._reservoir_layer_init).to(self.device)
            self._vstack = torch.vstack
            self._hstack = torch.hstack
            self._cat = torch.cat

        if self._random_state is not None:
            torch.manual_seed(int(self._random_state))

        self._os = 'torch'

    def _get_tensor_device(self,x):
        if isinstance(x,np.ndarray):
            return 'cpu'
        elif isinstance(x,torch.Tensor):
            return x.device.type
        else:
            raise Exception("Unsupported Tensor/Array type!")

    def _send_tensor_to_device(self,x):
        if hasattr(x,'to'):
            return x.to(self.device)
        else:
            return x

    def __call__(self, in_):

        assert self._get_tensor_device(in_) == self.device, (self.device,in_)

        if self._update_rule_id_train is None:
            self._update_rule_id_train = 2
        else:
            assert self._update_rule_id_train==2

        if self._bias:
            self._U = self._hstack((self._atleastND(in_).transpose(-1,-2),self.reservoir_layer.transpose(-1,-2),self._atleastND(self._bias_vec).transpose(-1,-2))).transpose(-1,-2)
        else:
            self._U = self._hstack((self._atleastND(in_).transpose(-1,-2),self.reservoir_layer.transpose(-1,-2))).transpose(-1,-2)
        return self._mm(self.Wout,self._U)

    def __generate_weight(self,w_name,size):
        try:
            size_str = {'Win':'_inSize','W':'resSize','Wback':'_outSize'}[w_name]
        except KeyError:
            if w_name in weight_dict:        
                raise NotImplementedError(f'Generation of {w_name} is not supported at the moment.')
            else:
                raise Exception(f"{w_name} is not a valid weight name. Valid names are {', '.join(weight_dict.keys())}.")

        self_size = getattr(self,size_str)

        if self_size is None:
            self.__setattr__(size_str,size)
        elif size is None:
            assert self_size is not None
        else:
            assert self_size == size, f'Reservoir has {weight_dict[w_name]} size={self_size} but you have given {weight_dict[w_name]} size={size}.'
        
        _size = getattr(self,size_str)

        if w_name == 'Win':
            assert _size is not None, 'Please provide input size of the reservoir.'
            w = np.random.rand(self.resSize,bool(self._bias)+_size) - 0.5
            # Win = np.random.uniform(size=(self.resSize,inSize+bias))<0.5
            # self.Win = np.where(Win==0, -1, Win)
        elif w_name == 'W':
            w = np.random.choice(self.__xn, p=self.__pn,size=(self.resSize,self.resSize))
        elif w_name == 'Wback':
            assert _size is not None, 'Please provide output size of the reservoir.'
            w = np.random.uniform(-2,2,size=(self.resSize,_size))

        return self._send_tensor_to_device(self._tensor(w.astype(self.dtype)))

    def __check_connections(self):
        assert self.W.shape[0] == self.W.shape[1], f'Reservoir matrix has to be square matrix: {self.W.shape[0]} != {self.W.shape[1]}'
        assert self.W.shape[1] == self.resSize, f'Mismatched reservoir size and reservoir weights: {self.W.shape[1]} != {self.resSize}'
        if self.Win is not None:
            assert self.Win.shape[0] == self.W.shape[0], f'Input matrix has shape, which is inconsistent with the reservoir matrix: {self.Win.shape[0]} != {self.W.shape[0]}'
            assert self.Win.shape[0] == self.resSize, f'Mismatched reservoir size and input weights: {self.Win.shape[0]} != {self.resSize}'
            assert self.Win.shape[1] == self._inSize + bool(self._bias), f'Mismatched input size and input weights: {self.Win.shape[1]} != {self._inSize + bool(self._bias)}'
            if self.Wout is not None:
                assert self.Wout.shape[1] ==self.W.shape[0] + self.Win.shape[1],f'Output matrix has shape, which is inconsistent with the reservoir and input matrices: {self.Wout.shape[1]} != {self.W.shape[0] + self.Win.shape[1]}'
                assert self.Wout.shape[0] == self._outSize, f'Mismatched output size and output weights: {self.Wout.shape[0]} != {self._outSize}'
        if self.Wback is not None:
            assert self.Wback.shape[0] == self.W.shape[0],f'Feedback matrix has shape, which is inconsistent with the reservoir matrix: {self.Wback.shape[0]} != {self.W.shape[0]}'
            assert self.Wback.shape[0] == self.resSize, f'Mismatched reservoir size and feedback weights: {self.Wback.shape[0]} != {self.resSize}'
            assert self.Wback.shape[1] == self._outSize, f'Mismatched output size and feedback weights: {self.Wback.shape[1]} != {self._outSize}'
            if self.Wout is not None:
                assert self.Wout.shape[0] == self.Wback.shape[1], f'Feedback matrix has shape, which is inconsistent with the output matrix: {self.Wout.shape[0]} != {self.Wback.shape[1]}'
                assert self.Wout.shape[0] == self._outSize, f'Mismatched output size and output weights: {self.Wout.shape[0]} != {self._outSize}'


class ESNX(ESN):
    """
    EchoStateNetwork X

    ESN for multitasking such as when using (mini)batches.
    """
    def __init__(self, 
                batch_size: int,
                resSize: int = 450, 
                xn: list = [0, 0.4, -0.4],
                pn: list = [0.9875, 0.00625, 0.00625], 
                random_state: float = None, 
                null_state_init: bool = True,
                custom_initState: np.ndarray = None,
                **kwargs):
        
        self.__name = 'ESNX'

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
    def excite(self, u: np.ndarray = None, y: np.ndarray = None, bias: Union[int, float] = None, f: Union[str, Any] = None, leak_rate: Union[int, float] = None, initLen: int = None, trainLen: int = None, initTrainLen_ratio: float = None, wobble: bool = False, wobbler: np.ndarray = None, leak_version=0, **kwargs) -> NoneType:
        raise NotImplementedError



class ESNS(ESN):
    """
    EchoStateNetwork S

    Ensemble of ESNs for training with multiple environments using (mini)batches.
    Shape: (#Reservoirs, Vector Length, Batch Size)
    """
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

        self.__name = 'ESNS'

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
    def excite(self, u: np.ndarray = None, y: np.ndarray = None, bias: Union[int, float] = None, f: Union[str, Any] = None, leak_rate: Union[int, float] = None, initLen: int = None, trainLen: int = None, initTrainLen_ratio: float = None, wobble: bool = False, wobbler: np.ndarray = None, leak_version=0, **kwargs) -> NoneType:
        raise NotImplementedError



class ESNN(ESN,torch.nn.Module):

    # TODO: Add output feedback support to forward.

    """
    EchoStateNetwork N

    Echo State Network as Pytorch Neural Network. Useful to train Wout via gradients.

    """

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

        self.__name = 'ESNN'

        ESN.__init__(self,
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

        self.Wout = torch.nn.Linear(in_size+self.resSize+bool(self._bias), out_size,bias=False,device=self.device,dtype=self.reservoir_layer.dtype)

        self._inSize = in_size
        self._outSize = out_size

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

        assert self._get_tensor_device(in_) == self.device, (self.device,in_)

        # if self._update_rule_id_train is None:
        #     self._update_rule_id_train = 2
        # else:
        #     assert self._update_rule_id_train==2


        if self._bias:
            self._U = self._vstack((self._atleastND(in_)[...,init_size:],self.reservoir_layer,self._atleastND(self._bias_vec)))
        else:
            self._U = self._vstack((self._atleastND(in_)[...,init_size:],self.reservoir_layer))

        return self._mm(self.Wout,self._U.transpose(-2,-1))

    
    def _pack_internal_state(self,in_=None,out_=None):
        raise NotImplementedError
    def excite(self, u: np.ndarray = None, y: np.ndarray = None, bias: Union[int, float] = None, f: Union[str, Any] = None, leak_rate: Union[int, float] = None, initLen: int = None, trainLen: int = None, initTrainLen_ratio: float = None, wobble: bool = False, wobbler: np.ndarray = None, leak_version=0, **kwargs) -> NoneType:
        raise NotImplementedError
