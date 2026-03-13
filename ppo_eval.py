import torch
import numpy as np

from offlinemodels.offline_generator import OfflineDataGenerator
from AstroCraft.PettingZoo_MA.env.CaptureTheFlagMA import CTFENVMA, MAX_FUEL

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from train import *
from itertools import combinations_with_replacement

import pickle as pkl
import gymnasium

class DummyEnv(gymnasium.Env):
    """
    A wrapper for the CTFENVMA environment that allows
    the environment to be used with sb3's MaskablePPO
    """
    
    def __init__(self, team_size, win_rew, flag_rew):
        super().__init__()
        self._num_actions = 14 
        self._team_size = team_size
        self.action_space = gymnasium.spaces.MultiDiscrete(np.array([self._num_actions]*self._team_size, dtype=int))
        self.observation_space = gymnasium.spaces.Box(low=-3, high=MAX_FUEL, shape=(self._team_size*2+2, 9))
        
    def reset(self, seed=None, options=None):
        """
        Changes the reset method to return just the obs
        and info for player 0, for use with sb3
        """
        pass
    
    def step(self, action):
        """
        Modifies the step method to only take an action 
        for player 0. Player 1 will take no actions. 
        Action masks are not returned at the end.
        """
        pass

    def action_mask(self):
        """
        Returns the action mask for player 0 only. If
        there are multiple such masks (e.g. for NvN games)
        the masks are stacked.
        """
        pass
    
    def prox_rew(self):
        """
        Provides a reward for proximity to goal
        """
        pass

def ppo_v_bot(bot):
    # Reset environment
    state, info = env.reset()
    rew = {'player0': 0, 'player1': 0}
    term = {'player0': False, 'player1': False}
    trunc = {'player0': False, 'player1': False}

    # Continue playing until the game is over
    while True:
        action0 = ppo.predict(state['player0']['observation'], action_masks=state['player0']['action_mask'])
        action1 = bot.select_action(state['player1'], rew['player1'], term['player1'], trunc['player1'], info['player1'])

        action = {'player0': action0, 'player1': action1}
        state0 = state['player0']
        state1 = state['player1']
        state, rew, term, trunc, info = env.step(action)
        rew0 = rew['player0']
        rew1 = rew['player1']

        if term['player0'] or term['player1'] or (trunc['player0'] and trunc['player1']):
            return rew0
        
def ppo_v_model(model):
    # Reset environment
    state, info = env.reset()
    rew = {'player0': 0, 'player1': 0}
    term = {'player0': False, 'player1': False}
    trunc = {'player0': False, 'player1': False}
        
    (hn, cn) = (torch.zeros(1,128).detach().to(device), torch.zeros(1,128).detach().to(device))

    # Continue playing until the game is over
    while True:
        action1, (hn,cn) = model.select_action(state['player1'], rew['player1'], term['player1'], trunc['player1'], info['player1'], hn, cn)
        action0 = ppo.predict(state['player0']['observation'], action_masks=state['player0']['action_mask'])

        action = {'player0': action0, 'player1': action1}
        state0 = state['player0']
        state1 = state['player1']
        state, rew, term, trunc, info = env.step(action)
        rew0 = rew['player0']
        rew1 = rew['player1']

        if term['player0'] or term['player1'] or (trunc['player0'] and trunc['player1']):
            return rew0
        
outcomes = {}

dataset = None
model1 = CQLA2C(dataset)
env = CTFENVMA(1, 1, 0)
ppo = MaskablePPO(MaskableActorCriticPolicy, DummyEnv(1, 1, 0), seed=42, verbose=1)
ppo.load('./ppo/ppo_agent.zip')

# PPO v models
weights = ['0','4','8']
for w1 in weights:
    # Load weights
    if w1 == '8':
        domain = 3
        
    else:
        domain = None

    model1.load_weights("./weights/"+w1+".pth")
    model1.domain = domain
    m1 = []
    
    # Play 100 games
    for _ in range(100):
        print("({})\tgame:{}".format(w1,_), end='\r')
        
        s1 = ppo_v_model(model1)
        m1.append(s1)
        
    # Record W/D
    outcomes[("PPO",w1)] = {"W": m1.count(1) / 100, "D": m1.count(0) / 100}

# PPO vs Bots
m1 = []

# Play 100 games
for _ in range(100):

    # Build a bot
    p_capture_slow = np.random.uniform(0,.5)
    p_return_slow = np.random.uniform(.5,.7)
    p_capture_fast = np.random.uniform(.337,1)
    p_return_fast = np.random.uniform(.45,1)
    p_intercept_slow = np.random.uniform(.5,1)
    p_intercept_fast = np.random.uniform(.62,1)
    orb_norm = 0
    while abs(orb_norm - 1) > .1:
        p_orbital_1 = np.random.uniform(.1,.2)
        p_orbital_2 = np.random.uniform(.17,.2)
        p_orbital_3 = np.random.uniform(.12,.15)
        p_orbital_4 = np.random.uniform(0,.12)
        p_orbital_5 = np.random.uniform(0,.12)
        p_orbital_6 = np.random.uniform(.18,.27)
        p_orbital_7 = np.random.uniform(0,.17)
        orbitals = [p_orbital_1, p_orbital_2, p_orbital_3, p_orbital_4, p_orbital_5, p_orbital_6, p_orbital_7]
        orb_norm = sum(orbitals)

    orbitals = [x/orb_norm for x in orbitals]

    p_dodge = np.random.uniform(.48,1)
    p_random_traj_change = 0

    bot = OfflineDataGenerator(1, p_capture_slow, p_return_slow, p_capture_fast, p_return_fast, p_intercept_slow, p_intercept_fast, orbitals, p_dodge, p_random_traj_change)
    print("({})\tgame:{}".format(w1,_), end='\r')
    
    s1 = ppo_v_bot(bot)
    m1.append(s1)
    
# Record W/D
outcomes[('PPO','bot')] = {"W": m1.count(1) / 100, "D": m1.count(0) / 100}

with open("./ppo_outcomes.pkl", 'wb') as outfile:
    pkl.dump(outcomes, outfile)