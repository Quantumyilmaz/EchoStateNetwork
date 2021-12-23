import sys,os
import pandas as pd
import numpy as np
import numpy as np
from torch._C import device
from tqdm.notebook import tqdm,trange
# from tqdm import tqdm,trange
from torchinfo import summary
sys.path.append("./../../")
from collections import namedtuple, deque
from itertools import count
import random,math
from utils.EchoStateNetwork import ESN,ESNX,at_least_2d
import torch

random.seed(42)
np.random.seed(42)

def train(env
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


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using {device}")

    use_torch=True

    safe_mode = 0

    env.seed(42)

    n_actions = env.action_space.n

    Wout_0 = kwargs.get("Wout",np.random.rand(n_actions,env.observation_space.shape[0]+resSize+bias))
    Win = kwargs.get("Win",np.random.rand(resSize,bias+env.observation_space.shape[0]) - 0.5)

    activation_fn = "relu"

    keyword_args = dict(resSize=resSize,random_state=42,verbose=0,bias=bias,leak_rate=leak_rate,f=activation_fn,leak_version=leak_version,use_torch=use_torch)#,pn=[0.75, 0.125, 0.125])


    # Reservoir to be optimized. Shares Wout with policy_net.
    policy_net_x = ESNX(Wout=Wout_0.copy(),batch_size=BATCH_SIZE,**keyword_args, Win=Win.copy())
    # Reservoir to get next_state_values. Shares Wout with target_net
    target_net_x = ESNX(Wout=Wout_0.copy(),batch_size=BATCH_SIZE,**keyword_args, Win=Win.copy())

    # For creating replay.
    policy_net = ESN(**keyword_args)
    policy_net.copy_connections_from(policy_net_x,bind=True)

    assert policy_net_x.resSize==target_net_x.resSize==policy_net.resSize

    # P from Sherman-Morrison formula
    P_0 = np.identity(env.observation_space.shape[0]+resSize+1)*P_alpha

    P = torch.from_numpy(P_0.copy())

    # Parameter to track no of ALL steps taken during the WHOLE training.
    steps_done = 0

    # Track episode durations
    episode_durations = []

    # Initialize replay memory
    memory = ReplayMemory(int(1e5),T)



    i_episode = tqdm(total=max_episodes)
    while i_episode.last_print_n < max_episodes:
        
        if safe_mode:
            assert torch.all(policy_net.Wout == Wout_0.copy())
            assert torch.all(policy_net_x.Wout == Wout_0.copy())
            assert torch.all(policy_net.reservoir_layer == np.zeros((resSize,1))); assert policy_net.reservoir_layer.shape == (resSize,1), policy_net.reservoir_layer.shape
            assert torch.all(policy_net_x.reservoir_layer == np.zeros((resSize,BATCH_SIZE))); assert policy_net_x.reservoir_layer.shape == (resSize,BATCH_SIZE)
            assert torch.all(target_net_x.reservoir_layer == np.zeros((resSize,BATCH_SIZE))); assert target_net_x.reservoir_layer.shape == (resSize,BATCH_SIZE)
            assert torch.all(P_0 == np.identity(env.observation_space.shape[0]+resSize+1)*P_alpha)
            assert torch.all(P==P_0)


        # Initialize the environment and state
        state = env.reset()
        for t in count():
            if safe_mode:
                assert torch.all(policy_net_x.Wout == policy_net.Wout)


            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
            # Select and perform an action
            action = select_action(policy_net=policy_net
                                    ,state=policy_net._tensor(state.astype(np.float64))
                                    ,n_actions=n_actions
                                    ,eps_threshold=eps_threshold)
            steps_done += 1
            next_state, reward, done, _ = env.step(action)

            # Observe new state
            if done:
                next_state = np.zeros(env.observation_space.shape[0],dtype=np.float32)
                if t<199: ### CHANGE THIS TO 499 FOR CARTPOLE_V1 ###
                    reward=-10

            # Store the transition in memory
            memory.push(state, action, next_state, reward, done)

            # Move to the next state
            state = next_state

            if safe_mode:
                P_check=P.copy()
                theta_check = policy_net_x.Wout.copy()

            optimized = optimize(env=env
                                ,memory=memory
                                ,BATCH_SIZE=BATCH_SIZE
                                ,policy_net_x=policy_net_x
                                ,target_net_x=target_net_x
                                ,T=T
                                ,P=P
                                ,GAMMA=GAMMA
                                ,kappa=kappa
                                ,approx_factor=approx_factor
                                ,forgetting_factor=forgetting_factor
                                ,omega=omega)


            if safe_mode:
                assert (P - P_check).sum()!=0 or not optimized,P
                assert (policy_net_x.Wout - theta_check).sum()!=0 or not optimized,P
                assert torch.all(policy_net_x.Wout == policy_net.Wout)# or not optimized,P

            if optimized:
                if i_episode.last_print_n % TARGET_UPDATE == 0:
                    target_net_x.copy_connections_from(policy_net_x,bind=False,weights_list=["Wout"])
                    assert torch.all(target_net_x.Wout == policy_net_x.Wout)
            
            if done:
                policy_net.reservoir_layer = policy_net._tensor(np.zeros((resSize,1)))
                if optimized:
                    episode_durations.append(t + 1)
                    print(episode_durations[-1])
                    i_episode.update()
                    P = torch.from_numpy(P_0.copy())
                    policy_net_x.Wout = torch.from_numpy(Wout_0.copy())
                    policy_net.Wout = policy_net_x.Wout
                break


    i_episode.close()
    print('Complete')

    target_net = ESN(**keyword_args)
    target_net.copy_connections_from(target_net_x)

    return episode_durations, target_net


def optimize(env,memory,BATCH_SIZE,policy_net_x,target_net_x,T,P,GAMMA,kappa,approx_factor,forgetting_factor,omega):
    if len(memory) < BATCH_SIZE:
        return False

    device = policy_net_x.device

    batch = memory.sample(BATCH_SIZE)
    
    #ressize,batchsize,serieslength,veclength(e.g. 4 for action in cartpole)
    state_series_batch = torch.from_numpy(np.stack([[*map(lambda x: x.state,i)] for i in batch])).to(device)
    # action_series_batch = torch.from_numpy(np.stack([[*map(lambda x: x.action,i)] for i in batch])).to(device)
    reward_series_batch = torch.from_numpy(np.stack([[*map(lambda x: x.reward,i)] for i in batch])).to(device)
    next_state_series_batch = torch.from_numpy(np.stack([[*map(lambda x: x.next_state,i)] for i in batch])).to(device)
    done_series_batch = torch.from_numpy(np.stack([[*map(lambda x: x.done,i)] for i in batch])).to(device)
    
    q_diff = torch.zeros((env.action_space.n,BATCH_SIZE)).double().to(device)
    u = torch.zeros((env.observation_space.shape[0]+policy_net_x.resSize+policy_net_x.bias,BATCH_SIZE)).double().to(device)
    
    for i in range(T):
        
        #ressize,veclength,batchsize
        state_batch = state_series_batch[:,i,:].T#.transpose(-2,-1)
        # action_batch = action_series_batch[:,i][None,:]
        reward_batch = reward_series_batch[:,i][None,:]
        next_state_batch = next_state_series_batch[:,i,:].T#.transpose(-2,-1)
        done_batch = done_series_batch[:,i][None,:]

        policy_net_x.update_reservoir_layer(state_batch)
        state_action_values = policy_net_x(state_batch)
        u += policy_net_x._U
        
        # v1
        target_net_x.update_reservoir_layer(next_state_batch)
        next_state_values = target_net_x(next_state_batch)

        next_state_values = next_state_values*(1-done_batch.to(torch.float64))
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

         # v1
        q_diff += expected_state_action_values - state_action_values
    
    ### UPDATE SECTION ###
    #Necessary for updates
    q_diff = at_least_2d(q_diff.sum(-1))/BATCH_SIZE/T
    u = at_least_2d(u.sum(-1))/BATCH_SIZE/T
    v = torch.matmul(P,u)
    g = v/(forgetting_factor+torch.matmul(v.T,u))
    
    # Updates
    policy_net_x.Wout += torch.matmul(q_diff,g.T) - kappa*torch.matmul(torch.sign(policy_net_x.Wout),P.T)
    P -= torch.matmul(g,v.T)
    P/=forgetting_factor

    target_net_x.reset_reservoir_layer()
    policy_net_x.reset_reservoir_layer()

    return True

def select_action(policy_net,state,n_actions,eps_threshold):
    sample = random.random()
    if sample > eps_threshold:
        policy_net.update_reservoir_layer(state)
        action = np.array(np.argmax(policy_net(state)))
    else:
        action = random.randrange(n_actions)
    return action

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):

    def __init__(self, capacity,series_length):
        self.series_memory = deque([],maxlen=series_length)
        self.memory = deque([],maxlen=capacity)
        self.length = 0
        self.series_length = series_length

    def push(self, *args):
        """Save a transition"""
#         if args[2] is None:
        if args[4]:
            while len(self.series_memory)<self.series_length:
                #v1
                self.series_memory.append(Transition(*args))
                self.length +=1
                #v2
#                 if len(self.series_memory)>0:
#                     self.series_memory.append(self.series_memory[-1])
#                     self.length +=1
#                 else:
#                     break
        else:
            self.series_memory.append(Transition(*args))
            self.length +=1
        if len(self.series_memory)==self.series_length:
            self.memory.append(self.series_memory)
            self.series_memory = deque([],maxlen=self.series_length)
            

    def sample(self, batch_size):#, series_length=1):
        return random.sample(self.memory, batch_size)
         #temp = random.sample(self.length, batch_size)
         #return [self.memory[i:i+series_length] for i in temp]

    def __len__(self):
        return len(self.memory)