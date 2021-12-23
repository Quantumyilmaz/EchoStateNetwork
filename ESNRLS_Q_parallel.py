import sys
import pandas as pd
import numpy as np
import numpy as np
from tqdm.notebook import tqdm,trange
# from tqdm import tqdm,trange
sys.path.append("./../../")
from itertools import count
import random,math
from utils.EchoStateNetwork import ESN,ESNS
import torch
from collections import deque

random.seed(42)
np.random.seed(42)
torch.manual_seed(0)

def train(envs
        ,resSize
        ,leak_rate
        ,leak_version
        ,bias
        ,max_episodes
        ,BATCH_SIZE
        ,T
        ,EPS_START
        ,EPS_END
        ,EPS_DECAY
        ,TARGET_UPDATE
        ,P_alpha
        ,GAMMA
        ,kappa
        ,approx_factor
        ,forgetting_factor
        ,omega
        ,**kwargs):

    no_of_reservoirs = envs.num_envs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu").type

    print(f"Using {device}")

    safe_mode = 0

    envs.seed(42)

    n_actions = envs.action_space.n


    Wout_0 = kwargs.get("Wout",np.stack(no_of_reservoirs*[np.random.rand(n_actions,envs.observation_space.shape[0]+resSize+bias)]))
    Win = kwargs.get("Win",np.random.rand(resSize,bias+envs.observation_space.shape[0]) - 0.5)

    activation_fn = kwargs.get("activation_fn","relu")

    keyword_args = dict(resSize=resSize,random_state=42,verbose=0,bias=bias,leak_rate=leak_rate,f=activation_fn,leak_version=leak_version)#,pn=[0.75, 0.125, 0.125])


    # Reservoir to be optimized. Shares Wout with policy_net.
    policy_net_x = ESNS(Wout=Wout_0.copy(),no_of_reservoirs=no_of_reservoirs,batch_size=BATCH_SIZE,**keyword_args, Win=Win.copy())
    # Reservoir to get next_state_values. Shares Wout with target_net
    target_net_x = ESNS(Wout=Wout_0.copy(),no_of_reservoirs=no_of_reservoirs,batch_size=BATCH_SIZE,**keyword_args, Win=Win.copy())

    # For creating replay.
    policy_net = ESNS(no_of_reservoirs=no_of_reservoirs,batch_size=1,**keyword_args)
    policy_net.copy_connections_from(policy_net_x,bind=True)

    assert policy_net_x.resSize==target_net_x.resSize==policy_net.resSize
    assert device == policy_net_x.device==target_net_x.device==policy_net.device

    # P from Sherman-Morrison formula
    P_0 = np.identity(envs.observation_space.shape[0]+resSize+1)*P_alpha
    P_0 = torch.from_numpy(np.concatenate(no_of_reservoirs*[P_0[None,:,:]]))
    P = P_0.clone().to(device)

    # Parameter to track no of ALL steps taken during the WHOLE training.
    steps_done = 0

    # Track episode durations
    episode_durations = {i:[] for i in range(no_of_reservoirs)}
    rewards = {i:[] for i in range(no_of_reservoirs)}

    # Initialize replay memory
    memory = ReplayMemoryParallel(no_of_reservoirs,kwargs.get("MEMORY_SIZE",int(1e5)),T)


    
    ts = np.array(no_of_reservoirs*[0])
    Rs = np.array(no_of_reservoirs*[0.])
    penalty = kwargs.get("penalty",False) # Currently only for CartPole-v0

    # Initialize the environment and state
    state = envs.reset()

    i_episode = tqdm(total=max_episodes,desc='No. of Episodes')
    collecting_samples = tqdm(total=BATCH_SIZE,desc='Collecting Samples')
    n_collecting_samples = 0 #pff
    t_step = tqdm(desc='Total Timesteps')
    ep_t = tqdm(desc='Episode Timesteps')
    optimized = False
    while i_episode.last_print_n < max_episodes:

        # assert policy_net_x.reservoir_layer.device.type == device
        # assert target_net_x.reservoir_layer.device.type == device
        # assert policy_net.reservoir_layer.device.type == device
        # assert policy_net_x._mm == torch.matmul
        # assert target_net_x._mm == torch.matmul
        # assert policy_net._mm == torch.matmul
        
        # if safe_mode:
        #     assert np.all(policy_net.Wout == Wout_0.copy())
        #     assert np.all(policy_net_x.Wout == Wout_0.copy())
        #     assert np.all(policy_net.reservoir_layer == np.zeros((resSize,1))); assert policy_net.reservoir_layer.shape == (resSize,1), policy_net.reservoir_layer.shape
        #     assert np.all(policy_net_x.reservoir_layer == np.zeros((resSize,BATCH_SIZE))); assert policy_net_x.reservoir_layer.shape == (resSize,BATCH_SIZE)
        #     assert np.all(target_net_x.reservoir_layer == np.zeros((resSize,BATCH_SIZE))); assert target_net_x.reservoir_layer.shape == (resSize,BATCH_SIZE)
        #     assert np.all(P_0 == np.identity(envs.observation_space.shape[0]+resSize+1)*P_alpha)
        #     assert np.all(P==P_0)

        assert torch.all(policy_net_x.Wout == policy_net.Wout)

        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        # Select and perform an action
        action = select_action(policy_net
                                ,state=torch.from_numpy(state).to(device)
                                ,no_of_reservoirs=no_of_reservoirs
                                ,n_actions=n_actions
                                ,eps_threshold=eps_threshold)
        envs.step_async(action)
        next_state, reward, done, _ = envs.step_wait()

        # Observe new state
        next_state[done] = np.zeros(envs.observation_space.shape[0],dtype=np.float32)

        if penalty:
            reward[done & (ts<199)] = -penalty

        # Store the transition in memory
        memory.push(state, action, next_state, reward, done)
        steps_done+=1
        
        if not optimized:
            n_temp = len(memory) - n_collecting_samples
            if n_temp:
                collecting_samples.update(n_temp)
                n_collecting_samples += n_temp
                if n_collecting_samples >= BATCH_SIZE:
                    collecting_samples.close()

        # Perform one step of the optimization (on the policy network)
        # v1
        optimized = optimize(envs=envs
                            ,memory=memory
                            ,policy_net_x=policy_net_x
                            ,target_net_x=target_net_x
                            ,T=T
                            ,P=P
                            ,GAMMA=GAMMA
                            ,kappa=kappa
                            ,approx_factor=approx_factor
                            ,forgetting_factor=forgetting_factor
                            ,omega=omega)
        if optimized:
            t_step.update()
            ep_t.update()
            ts += 1
            Rs += reward
            if i_episode.last_print_n % TARGET_UPDATE == 0:
                target_net_x.Wout = policy_net_x.Wout.clone()#.to(device)


        if np.any(done):
            # Move to the next state
            state[done]=envs.env_method("reset",indices=np.where(done)[0])
            state[~done] = next_state[~done]
            policy_net.reservoir_layer[done] = torch.zeros((no_of_reservoirs,resSize,1))[done].double().to(device)
            if optimized:
                print('#############')
                print(done)
                print(ts)
                print(Rs)
                for i,d_ in enumerate(done):
                    if d_:
                        episode_durations[i].append(ts[i])
                        rewards[i].append(Rs[i])
                ts[done] = 0
                Rs[done] = 0.
                i_episode.update()
                P[done] = P_0.clone()[done].to(device)
                policy_net_x.Wout[done] = torch.from_numpy(Wout_0).clone()[done].to(device)
                policy_net.Wout = policy_net_x.Wout
                ep_t.reset()
        else:
            # Move to the next state
            state=next_state.astype(np.float32)

    i_episode.close()
    t_step.close()
    ep_t.close()
    print('Complete')

    target_net = ESN(**keyword_args)
    target_net.copy_connections_from(target_net_x)

    return rewards, episode_durations, target_net


def optimize(envs,memory,policy_net_x,target_net_x,T,P,GAMMA,kappa,approx_factor,forgetting_factor,omega):
    
    BATCH_SIZE = policy_net_x.batch_size

    if len(memory) < BATCH_SIZE:
        return False

    no_of_reservoirs = policy_net_x.no_of_reservoirs
    device = policy_net_x.device

    batch = memory.sample(BATCH_SIZE)
    
    #ressize,batchsize,serieslength,veclength(e.g. 4 for action in cartpole)
    state_series_batch = torch.from_numpy(np.stack([[*map(lambda x: x.state,i)] for i in batch])).to(device)
    # action_series_batch = torch.from_numpy(np.stack([[*map(lambda x: x.action,i)] for i in batch])).to(device)
    reward_series_batch = torch.from_numpy(np.stack([[*map(lambda x: x.reward,i)] for i in batch])).to(device)
    next_state_series_batch = torch.from_numpy(np.stack([[*map(lambda x: x.next_state,i)] for i in batch])).to(device)
    done_series_batch = torch.from_numpy(np.stack([[*map(lambda x: x.done,i)] for i in batch])).to(device)
    
    q_diff = torch.zeros((no_of_reservoirs,envs.action_space.n,BATCH_SIZE)).to(device)
    u = torch.zeros((no_of_reservoirs,envs.observation_space.shape[0]+policy_net_x.resSize+policy_net_x.bias,BATCH_SIZE)).to(device)
    
    for i in range(T):
        
        #ressize,veclength,batchsize
        state_batch = state_series_batch[:,:,i,:].transpose(-2,-1)
        # action_batch = action_series_batch[:,:,i][:,None,:]
        reward_batch = reward_series_batch[:,:,i][:,None,:]
        next_state_batch = next_state_series_batch[:,:,i,:].transpose(-2,-1)
        done_batch = done_series_batch[:,:,i][:,None,:]

        policy_net_x.update_reservoir_layer(state_batch)
        state_action_values = policy_net_x(state_batch)
        u += policy_net_x._U
        
        target_net_x.update_reservoir_layer(next_state_batch)
        next_state_values = target_net_x(next_state_batch)

        next_state_values = next_state_values*(1-done_batch.to(torch.float64))
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        q_diff += expected_state_action_values - state_action_values
    
    ### UPDATE SECTION ###
    #Necessary for updates
    q_diff = (q_diff.sum(-1)[:,:,None]/BATCH_SIZE/T).double()
    u = (u.sum(-1)[:,:,None]/BATCH_SIZE/T).double()
    v = torch.matmul(P,u)
    g = v/(forgetting_factor+torch.matmul(v.transpose(-2,-1),u))
    
    # Updates
    policy_net_x.Wout += torch.matmul(q_diff,g.transpose(-2,-1)) - kappa*torch.matmul(torch.sign(policy_net_x.Wout),P.transpose(-2,-1))
    P -= torch.matmul(g,v.transpose(-2,-1))
    P/=forgetting_factor

    target_net_x.reset_reservoir_layer()
    policy_net_x.reset_reservoir_layer()

    return True

def select_action(policy_net,state,no_of_reservoirs,n_actions,eps_threshold):
    sample = random.random()
    if sample > eps_threshold:
        policy_net.update_reservoir_layer(state)
        action = torch.argmax(policy_net(state),1).cpu().numpy().ravel()
    else:
        action = torch.randint(0,n_actions,(no_of_reservoirs,1)).cpu().numpy().ravel()
    return action



class TransitionSeries:
    def __init__(self,max_len,keys):
        assert isinstance(max_len,int)
        self.max_len = max_len
        self.keys = tuple(keys)
        self.transition_series_dict = {key:[] for key in self.keys}
        for key in keys:
            assert isinstance(key,str)
            setattr(self,key,self.transition_series_dict[key])
        
    def __call__(self,*args):
        assert len(args) == len(self.keys),args
        assert len(self) < self.max_len
        for arg, key in zip(args,self.transition_series_dict):
            self.transition_series_dict[key].append(arg)
    
    def __len__(self):
        return len(getattr(self,self.keys[0]))
    
class ReplayMemory(object):

    def __init__(self, capacity,series_length):
        self.keys = ['state', 'action', 'next_state', 'reward', 'done']
        self.series_memory = TransitionSeries(series_length,self.keys)
        self.memory = deque([],maxlen=capacity)
        self.length = 0
        self.series_length = series_length

    def push(self, *args):
        """Save a transition"""
        if args[4]:
            while len(self.series_memory)<self.series_length:
                self.series_memory(*args)
                self.length +=1
        else:
            self.series_memory(*args)
            self.length +=1
        if len(self.series_memory)==self.series_length:
            self.memory.append(self.series_memory)
            self.series_memory = TransitionSeries(self.series_length,self.keys)
            

    def sample(self, batch_size):#, series_length=1):
        return random.sample(self.memory, batch_size)
         #temp = random.sample(self.length, batch_size)
         #return [self.memory[i:i+series_length] for i in temp]

    def __len__(self):
        return len(self.memory)
    
class ReplayMemoryParallel:
    def __init__(self, n_envs, capacity, series_length):
        self.memories = [ReplayMemory(capacity, series_length) for i in range(n_envs)]
        self.series_length = series_length
        
    def push(self, *args_of_args):
        for i,args in enumerate(zip(*args_of_args)):
            self.memories[i].push(*args)
            
    def sample(self, batch_size):#, series_length=1):
        return [*map(lambda mem: random.sample(mem.memory, batch_size),self.memories)]
    
    def __len__(self):
        return min(map(len,self.memories))
        