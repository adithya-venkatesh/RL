# -*- coding: utf-8 -*-
"""main.py

Modified gridworld with CAR operator implementation.
Includes pause/resume functionality for long-running experiments.

Based on original gridworld.ipynb with additions for:
1. CAR operator from "Towards Optimal Adversarial Robust Q-learning with Bellman Infinity-error"
2. Comparison between Pessimistic Q-Learning and CAR operator
3. CSV logging for 500 experiments
4. Pause/Resume capability
5. Progress tracking and analysis

Nov-27-2025: Based on feedback, adding the following in this version:
1. Fixing bugs in the CAR operator implementation
2. Adding clearer comments and documentation
3: Add new learning function (greedy target)
4: Add new play function (off-policy)
5: Update main() to test this method
6: Run experiments and compare
"""

# Import all the libraries
import math
import time
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import copy
from datetime import datetime
import signal
import sys

# Changes to next two lines 
import stable_baselines3
from stable_baselines3 import SAC, PPO, TD3, DQN
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback

# define the linear schedule exponential function
def linear_schedule_exp(min_exp, ini_exp, total, current):
    return max(min_exp, ((total-current)*ini_exp)/total)

# Define the GridWorld Class
class GridWorld(gym.Env):
    ## Initialise starting data
    def __init__(self):
        # Set information about the gridworld
        self.height = 5
        self.width = 5
        self.grid = np.zeros(( self.height, self.width)) - 1
        self.total_step = 0

        # Set random start location for the agent
        self.current_location = [4, np.random.randint(0,5)]

        # Set locations for the bomb and the gold
        self.bomb_location = [4,3]
        self.gold_location = [0,3]
        self.terminal_states = [ self.bomb_location, self.gold_location]

        # Set grid rewards for special cells
        self.grid[ self.bomb_location[0], self.bomb_location[1]] = -10
        self.grid[ self.gold_location[0], self.gold_location[1]] = 10

        # Set available actions
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([5,5])


    ## Put methods here:
    def get_available_actions(self):
        """Returns possible actions"""
        return self.actions

    def agent_on_map(self):
        """Prints out current location of the agent on the grid (used for debugging)"""
        grid = np.zeros(( self.height, self.width))
        grid[ self.current_location[0], self.current_location[1]] = 1
        return grid

    def get_reward(self, new_location):
        """Returns the reward for an input position"""
        return self.grid[ new_location[0], new_location[1]]

    def seed(self, seed=None):
        if seed:
            np.random.seed(seed)


    def step(self, action):
        """Moves the agent in the specified direction. If agent is at a border, agent stays still
        but takes negative reward. Function returns the reward for the move."""
        # Store previous location
        last_location = self.current_location
        if action in [0,1,2,3]:
            true_action = self.actions[action]
        else:
            true_action = action

        # UP
        if true_action == 'UP':
            # If agent is at the top, stay still, collect reward
            if last_location[0] == 0:
                reward = self.get_reward(last_location)
            else:
                self.current_location = [self.current_location[0] - 1, self.current_location[1]]
                reward = self.get_reward(self.current_location)

        # DOWN
        elif true_action == 'DOWN':
                # If agent is at bottom, stay still, collect reward
                if last_location[0] == self.height - 1:
                    reward = self.get_reward(last_location)
                else:
                    self.current_location =  [self.current_location[0] + 1, self.current_location[1]]
                    reward = self.get_reward(self.current_location)

        # LEFT
        elif true_action == 'LEFT':
                # If agent is at the left, stay still, collect reward
                if last_location[1] == 0:
                    reward = self.get_reward(last_location)
                else:
                    self.current_location = [self.current_location[0], self.current_location[1] - 1]
                    reward = self.get_reward(self.current_location)

        # RIGHT
        elif true_action == 'RIGHT':
                # If agent is at the right, stay still, collect reward
                if last_location[1] == self.width - 1:
                    reward = self.get_reward(last_location)
                else:
                    self.current_location = [self.current_location[0], self.current_location[1] + 1]
                    reward = self.get_reward(self.current_location)
        self.total_step += 1
        # Changes here to return 5th parameter
        # Separated done into terminated (natural episode end) and truncated (time limit reached)
        # Changed return from 4 values to 5 values: (observation, reward, terminated, truncated, info)
      
        terminated = self.current_location in self.terminal_states
        truncated = self.total_step >= 1000

        return np.asarray(self.current_location), reward, terminated, truncated, {}

    def take_action(self, state, action):
        last_location = state
        if action in [0,1,2,3]:
            true_action = self.actions[action]
        else:
            true_action = action

        # UP
        if true_action == 'UP':
            # If agent is at the top, stay still, collect reward
            if last_location[0] == 0:
                state = state
            else:
                state = [state[0] - 1, state[1]]

        # DOWN
        elif true_action == 'DOWN':
                # If agent is at bottom, stay still, collect reward
                if last_location[0] == self.height - 1:
                    state = state
                else:
                    state =  [state[0] + 1, state[1]]
        # LEFT
        elif true_action == 'LEFT':
                # If agent is at the left, stay still, collect reward
                if last_location[1] == 0:
                    state = state
                else:
                    state = [state[0], state[1] - 1]

        # RIGHT
        elif true_action == 'RIGHT':
                # If agent is at the right, stay still, collect reward
                if last_location[1] == self.width - 1:
                    state = state
                else:
                    state = [state[0], state[1] + 1]

        return tuple(state)


    def check_state(self):
        if self.current_location in self.terminal_states:
            return 'TERMINAL'
    
    def reset(self, seed=None):
        self.total_step = 0
        if seed is not None:  
            np.random.seed(seed)
        
        self.current_location = [np.random.randint(0,5), np.random.randint(0,5)]
        
        while self.current_location in self.terminal_states:
            self.current_location = [np.random.randint(0,5), np.random.randint(0,5)]
        return np.asarray(self.current_location)

# Define the attack class
class GridWorld_RL_Attack(gym.Env):
    ## Initialise starting data
    def __init__(self, agentQ, agent_type='pessimistic'):
        """
        Initialize attack environment for training adversarial attackers.
        
        Args:
            agentQ: The agent to train against
            agent_type: 'pessimistic', 'car', or 'offpolicy'
        """
        self.agent_type = agent_type  # Store which type of agent we're attacking
        # Set information about the gridworld
        self.height = 5
        self.width = 5
        self.grid = np.zeros(( self.height, self.width)) - 1
        self.agentQ = agentQ

        # Set random start location for the agent
        self.current_location = [4, np.random.randint(0,5)]

        # Set locations for the bomb and the gold
        self.bomb_location = [4,3]
        self.gold_location = [0,3]
        self.terminal_states = [ self.bomb_location, self.gold_location]

        # Set grid rewards for special cells
        self.grid[ self.bomb_location[0], self.bomb_location[1]] = -10
        self.grid[ self.gold_location[0], self.gold_location[1]] = 10

        # Set available actions
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.MultiDiscrete([5,5])
        self.purturb_state = None
        self.belief = None


    ## Put methods here:
    def get_available_actions(self):
        """Returns possible actions"""
        return self.actions

    def agent_on_map(self):
        """Prints out current location of the agent on the grid (used for debugging)"""
        grid = np.zeros(( self.height, self.width))
        grid[ self.current_location[0], self.current_location[1]] = 1
        return grid

    def get_reward(self, new_location):
        """Returns the reward for an input position"""
        return self.grid[ new_location[0], new_location[1]]

    def seed(self, seed=None):
        if seed:
            np.random.seed(seed)


    def step(self, action):
        """Moves the agent in the specified direction. If agent is at a border, agent stays still
        but takes negative reward. Function returns the reward for the move."""
        # Store previous location
        last_location = self.current_location
        if action in [0,1,2,3,4]:
            true_action = self.actions[action]
        else:
            true_action = action
        # UP
        if true_action == 'UP':
            # If agent is at the top, stay still, collect reward
            if last_location[0] == 0:
                self.purturb_state = self.current_location
            else:
                self.purturb_state = [self.current_location[0] - 1, self.current_location[1]]
        # DOWN
        elif true_action == 'DOWN':
                # If agent is at bottom, stay still, collect reward
                if last_location[0] == self.height - 1:
                    self.purturb_state = self.current_location
                else:
                    self.purturb_state =  [self.current_location[0] + 1, self.current_location[1]]
        # LEFT
        elif true_action == 'LEFT':
                # If agent is at the left, stay still, collect reward
                if last_location[1] == 0:
                    self.purturb_state = self.current_location
                else:
                    self.purturb_state = [self.current_location[0], self.current_location[1] - 1]
        # RIGHT
        elif true_action == 'RIGHT':
                # If agent is at the right, stay still, collect reward
                if last_location[1] == self.width - 1:
                    self.purturb_state = self.current_location
                else:
                    self.purturb_state = [self.current_location[0], self.current_location[1] + 1]
        elif true_action == 'STAY':
            self.purturb_state = self.current_location


        # Agent action selection depends on agent type
        if self.agent_type == 'pessimistic':
            # PESSIMISTIC: Use belief-based policy
            agent_action = self.agentQ.choose_action_att(
                ['UP', 'DOWN', 'LEFT', 'RIGHT'], 
                tuple(self.purturb_state), 
                0
            )
            self.current_location = self.take_action(self.current_location, agent_action)
            
            # Update belief for pessimistic agent
            possible_true_states = self.agentQ.get_ball(self.purturb_state)
            belief_before_action = states_intersection(self.belief, possible_true_states)
            self.belief = set()
            for state in belief_before_action:
                self.belief.add(self.take_action(state, agent_action))
            self.agentQ.defender_policy = self.agentQ.get_policy_with_belief(
                self.agentQ.defender_policy, self.belief
            )
        
        elif self.agent_type == 'car':
            # CAR: Use simple greedy from Q-table (CAR operator)
            agent_action = self.agentQ.choose_action_car(
                ['UP', 'DOWN', 'LEFT', 'RIGHT'],
                tuple(self.purturb_state),
                0
            )
            self.current_location = self.take_action(self.current_location, agent_action)
        
        elif self.agent_type == 'offpolicy':
            # OFF-POLICY: Use greedy from Q-table at test time
            state = tuple(self.purturb_state)
            q_values = self.agentQ.q_table_withatt[state]
            maxValue = max(q_values.values())
            agent_action = np.random.choice([
                k for k, v in q_values.items() if v == maxValue
            ])
            self.current_location = self.take_action(self.current_location, agent_action)
        
        else:
            raise ValueError(f"Unknown agent_type: {self.agent_type}")
        
        terminated = list(self.current_location) in self.terminal_states
        reward = -self.get_reward(self.current_location)
        self.total_step += 1
        truncated = self.total_step >= 1000

        return np.asarray(self.current_location), reward, terminated, truncated, {}

    def take_action(self, state, action):
        last_location = state
        if action in [0,1,2,3]:
            true_action = self.actions[action]
        else:
            true_action = action

        # UP
        if true_action == 'UP':
            # If agent is at the top, stay still, collect reward
            if last_location[0] == 0:
                state = state
            else:
                state = [state[0] - 1, state[1]]

        # DOWN
        elif true_action == 'DOWN':
                # If agent is at bottom, stay still, collect reward
                if last_location[0] == self.height - 1:
                    state = state
                else:
                    state =  [state[0] + 1, state[1]]
        # LEFT
        elif true_action == 'LEFT':
                # If agent is at the left, stay still, collect reward
                if last_location[1] == 0:
                    state = state
                else:
                    state = [state[0], state[1] - 1]

        # RIGHT
        elif true_action == 'RIGHT':
                # If agent is at the right, stay still, collect reward
                if last_location[1] == self.width - 1:
                    state = state
                else:
                    state = [state[0], state[1] + 1]

        return tuple(state)


    def check_state(self):
        """Check if the agent is in a terminal state (gold or bomb), if so return 'TERMINAL'"""
        if self.current_location == self.bomb_location:
            print("BOOM")
        if self.current_location == self.gold_location:
            print("GOLD")
        if self.current_location in self.terminal_states:
            return 'TERMINAL'

    def reset(self, seed=None):
        self.current_location =  [np.random.randint(0,5), np.random.randint(0,5)]
        self.belief = set()
        for state in self.agentQ.all_state:
            self.belief.add(state)
        # setting a random value to seed
        if seed is not None:
          np.random.seed(seed)
        
        self.agentQ.defender_policy = self.agentQ.get_policy_with_belief(self.agentQ.defender_policy, self.belief)
        while self.current_location in self.terminal_states:
            self.current_location =  [np.random.randint(0,5), np.random.randint(0,5)]
        self.total_step = 0
        return np.asarray(self.current_location)

# Define the Q Agent
class Q_Agent():
    # Intialise
    def __init__(self, environment, epsilon=0.05, alpha=0.1, gamma=1):
        self.environment = environment
        self.q_table = dict() # Store all Q-values in dictionary of dictionaries
        self.q_table_withatt = dict()
        self.all_state = []
        for x in range(environment.height): # Loop through all possible grid spaces, create sub-dictionary for each
            for y in range(environment.width):
                self.all_state.append((x,y))
                self.q_table[(x,y)] = {'UP':np.random.uniform(0,1), 'DOWN':np.random.uniform(0,1), 'LEFT':np.random.uniform(0,1), 'RIGHT':np.random.uniform(0,1)} # Populate sub-dictionary with zero values for possible moves
                self.q_table_withatt[(x,y)] = {'UP':np.random.uniform(0,1), 'DOWN':np.random.uniform(0,1), 'LEFT':np.random.uniform(0,1), 'RIGHT':np.random.uniform(0,1)}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.defender_policy= None

    def choose_action(self, available_actions, eps = 0.05):
        """Returns the optimal action from Q-Value table. If multiple optimal actions, chooses random choice.
        Will make an exploratory random action dependent on epsilon."""
        if np.random.uniform(0,1) < eps:
            action = available_actions[np.random.randint(0, len(available_actions))]
        else:
            q_values_of_state = self.q_table[tuple(self.environment.current_location)]
            maxValue = max(q_values_of_state.values())
            action = np.random.choice([k for k, v in q_values_of_state.items() if v == maxValue])

        return action

    def choose_action_att(self, available_actions, state, eps = 0.05):
        """Returns the optimal action from Q-Value table. If multiple optimal actions, chooses random choice.
        Will make an exploratory random action dependent on epsilon."""
        if np.random.uniform(0,1) < eps:
            action = available_actions[np.random.randint(0, len(available_actions))]
        else:
            action = self.defender_policy[state]
        return action
    
    def choose_action_car(self, available_actions, state, eps=0.05):
        """
        Action selection for CAR operator approach.
        Takes greedy action with respect to Q-function at CURRENT state directly.
        Does NOT use belief-based policy.
        """
        if np.random.uniform(0, 1) < eps:
            # Exploration
            action = available_actions[np.random.randint(0, len(available_actions))]
        else:
            # Exploitation: greedy with respect to Q-table at current state
            q_values_of_state = self.q_table_withatt[tuple(state)]
            maxValue = max(q_values_of_state.values())
            action = np.random.choice([k for k, v in q_values_of_state.items() if v == maxValue])
        
        return action
    
    def learn(self, old_state, reward, new_state, action, alpha = 0.1):
        """Updates the Q-value table using Q-learning"""
        q_values_of_state = self.q_table[tuple(new_state)]
        max_q_value_in_new_state = max(q_values_of_state.values())
        current_q_value = self.q_table[tuple(old_state)][action]

        self.q_table[tuple(old_state)][action] = (1 - alpha) * current_q_value + alpha * (reward + self.gamma * max_q_value_in_new_state)

    def learn_with_car_operator(self, old_state, reward, new_state, action, alpha=0.1):
        """
        Updates Q-value using CAR operator from the paper.
        
        4.4:
        (T_car Q)(s,a) = r(s,a) + γ * E[min_{s'_ν ∈ B_ε(s')} Q(s', arg max_a Q(s'_ν, a))]
        
        Attacker finds worst-case perturbation, then agent acts greedily
        on the perturbed state.
        """
        # Get all possible perturbed states within epsilon-ball around new_state
        perturbed_states = self.get_ball(tuple(new_state))
        
        # For each perturbed state s'_ν, find the greedy action with respect to Q(s'_ν, ·)
        # Then evaluate Q at the TRUE next state s' with that action
        # Finally, take the minimum over all perturbations
        
        min_q_value = float('inf')
        
        for perturbed_state in perturbed_states:
            # Agent takes greedy action based on PERTURBED observation
            q_values_at_perturbed = self.q_table_withatt[perturbed_state]
            greedy_action_at_perturbed = max(q_values_at_perturbed, key=q_values_at_perturbed.get)
            
            # Evaluate Q at the TRUE next state with the action chosen from perturbed state
            q_value_at_true_state = self.q_table_withatt[tuple(new_state)][greedy_action_at_perturbed]
            
            # Attacker minimizes over perturbations
            if q_value_at_true_state < min_q_value:
                min_q_value = q_value_at_true_state
        
        # Standard Q-learning update with min-max value
        current_q_value = self.q_table_withatt[tuple(old_state)][action]
        self.q_table_withatt[tuple(old_state)][action] = (1 - alpha) * current_q_value + alpha * (reward + self.gamma * min_q_value)

    def learn_with_greedy_target(self, old_state, reward, new_state, action, alpha=0.1):
        """
        Off-policy learning: Behavior policy uses max-min with belief,
        but target policy is GREEDY (like vanilla Q-learning).
    
        Key difference from pessimistic: In the Q-update, we use
        max_a Q(s', a) instead of max-min over belief.
        """
        # Greedy target: just take max over all actions at new_state
        max_q_value = max(self.q_table_withatt[tuple(new_state)].values())
    
        # Standard Q-learning update (same as vanilla Q-learning)
        current_q_value = self.q_table_withatt[tuple(old_state)][action]
    
        self.q_table_withatt[tuple(old_state)][action] = \
            (1 - alpha) * current_q_value + alpha * (reward + self.gamma * max_q_value)
    
    def find_maxmin(self, q_table, next_state, policy, new_policy):
        """Compute max-min Bellman target (without belief tracking)."""
        states = self.get_ball(next_state)
        values = dict()
        
        # For each action, compute minimum Q-value over all perturbed states
        for action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            q_values_for_action = []
            for state in states:
                q_values_for_action.append(q_table[state][action])
            values[action] = min(q_values_for_action)
        
        # Return maximum over actions (max-min)
        return max(values.values())

    def find_maxmin_with_belief(self, q_table_true, q_table_withatt, next_state, policy, new_policy, belief):
        """Compute max-min Bellman target with belief tracking."""
        states = self.get_ball(next_state)
        possible_states = states_intersection(belief, states)
        values = dict()
        
        if len(possible_states) > 1:
            # For each action, compute minimum Q-value over all possible states in belief
            for action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                q_values_for_action = []
                for state in possible_states:
                    q_values_for_action.append(q_table_withatt[state][action])
                values[action] = min(q_values_for_action)
            # Return maximum over actions (max-min)
            return max(values.values())
        elif len(possible_states) == 1:
            # Single state in belief - return max Q-value
            return max(q_table_withatt[list(possible_states)[0]].values())
        else:
            # Empty belief intersection - fallback to next_state
            return max(q_table_withatt[next_state].values())

    def learn_with_att(self, old_state, reward, new_state, action, policy, new_policy, alpha = 0.1):
        """Updates the Q-value table using Q-learning"""
        max_q_value_in_new_state = self.find_maxmin(self.q_table_withatt, tuple(new_state), policy, new_policy)
        current_q_value = self.q_table_withatt[tuple(old_state)][action]

        self.q_table_withatt[tuple(old_state)][action] = (1 - alpha) * current_q_value + alpha * (reward + self.gamma * max_q_value_in_new_state)

    def learn_with_att_with_belief(self, old_state, reward, new_state, action, policy, new_policy, belief, alpha = 0.1):
        """Updates the Q-value table using Q-learning"""
        max_q_value_in_new_state = self.find_maxmin_with_belief(self.q_table, self.q_table_withatt, tuple(new_state), policy, new_policy, belief)
        current_q_value = self.q_table_withatt[tuple(old_state)][action]

        self.q_table_withatt[tuple(old_state)][action] = (1 - alpha) * current_q_value + alpha * (reward + self.gamma * max_q_value_in_new_state)

    def get_att_state(self, true_state, policy):
        """Find worst-case perturbed state given policy."""
        states = self.get_ball(true_state)
        values = dict()
        for i in states:
             values[i] = self.q_table_withatt[true_state][policy[i]]
        att_state = min(values, key = values.get)
        return att_state

    def pareto_update(self, policy):
        no_change = False
        old = dict()
        while not no_change:
            no_change = True
            value_min = dict()
            for state in self.all_state:
                states = self.get_ball(state)
                values = dict()
                for action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                    q_values = []
                    for i in states:
                        q_values.append(self.q_table_withatt[i][action])
                    values[action] = min(q_values)
                value_min[state] = values

            for state in self.all_state:
                old[state] = policy[state]
                policy[state] = max(value_min[state], key = value_min[state].get)

            for state in self.all_state:
                if old[state] != policy[state]:
                    no_change = False

        return policy

    def get_policy(self, init = False):
        policy = dict()
        if init:
            for state in self.all_state:
                policy[state] = np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        else:
            for state in self.all_state:
                values = dict()
                states = self.get_ball(state)
                for action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                    tmp = []
                    for true_state in states:
                        tmp.append(self.q_table_withatt[true_state][action])
                    values[action] = min(tmp)
                policy[state] = max(values, key = values.get)
        return policy

    def get_policy_naive(self):
        policy = dict()
        for state in self.all_state:
            policy[state] = max(self.q_table_withatt[state], key = self.q_table_withatt[state].get)
        return policy

    def get_policy_with_belief(self, policy, belief):
        new_policy = dict()
        value_min = dict()
        for state in self.all_state:
            states = self.get_ball(state)
            possible_states = states_intersection(belief, states)
            values = dict()
            
            if len(possible_states) > 1:
                for action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                    q_values = []
                    for i in possible_states:
                        q_values.append(self.q_table_withatt[i][action])
                    values[action] = min(q_values)
                value_min[state] = values
            elif len(possible_states) == 1:
                value_min[state] = self.q_table_withatt[list(possible_states)[0]]
            else:
                value_min[state] = {action: self.q_table_withatt[state][action] for action in ['UP', 'DOWN', 'LEFT', 'RIGHT']}

        for state in self.all_state:
            new_policy[state] = max(value_min[state], key = value_min[state].get)
        return new_policy

    def get_ball(self, state):
        x = state[0]
        y = state[1]
        states = []
        if x != 0:
            states.append((x-1,y))
        if x != 4:
            states.append((x+1,y))
        if y != 0:
            states.append((x,y-1))
        if y != 4:
            states.append((x,y+1))
        states.append((x,y))
        return states

# Define the routine - play
def play(environment, agent, trials=500, max_steps_per_episode=1000, learn=False):
    """The play function runs iterations and updates Q-values if desired."""
    reward_per_episode = [] # Initialise performance log

    for trial in range(trials): # Run trials
        cumulative_reward = 0 # Initialise values of each game
        step = 0
        game_over = False
        old_state = environment.reset()
        while step < max_steps_per_episode and game_over != True: # Run until max steps or until game is finished
            old_state = tuple(environment.current_location)
            if learn:
                eps = linear_schedule_exp(0.01, 0.2, trials, trial)
            else:
                eps = 0
            action = agent.choose_action(environment.actions, eps)
            
            _, reward, terminated, truncated, _ = environment.step(action)
            new_state = environment.current_location

            if learn == True: # Update Q-values if learning is specified
                alpha = linear_schedule_exp(0.005, 0.1, trials, trial)
                agent.learn(old_state, reward, new_state, action, alpha)

            cumulative_reward += reward
            step += 1

            if environment.check_state() == 'TERMINAL': # If game is in terminal state, game over and start next trial
                environment.__init__()
                game_over = True

        reward_per_episode.append(cumulative_reward) # Append reward for current trial to performance log
    
    return reward_per_episode # Return performance log

# Define the routine - play_att
def play_att(environment, agent, trials=500, max_steps_per_episode=1000, learn=False):
    """The play function runs iterations and updates Q-values if desired."""
    reward_per_episode = [] # Initialise performance log
    if learn:
        agent.q_table_withatt = copy.deepcopy(agent.q_table)
        agent.defender_policy = agent.get_policy_naive()
    else:
        agent.defender_policy = agent.get_policy_naive()
    
    for trial in range(trials): # Run trials
        cumulative_reward = 0 # Initialise values of each game
        step = 0
        game_over = False
        old_state = environment.reset()
        while step < max_steps_per_episode and game_over != True: # Run until max steps or until game is finished
            old_state = tuple(environment.current_location)
            att_state = agent.get_att_state(old_state, agent.defender_policy)
            if learn:
                eps = linear_schedule_exp(0.01, 0.2, trials, trial)
            else:
                eps = 0
            action = agent.choose_action_att(environment.actions, att_state, eps)
            
            if learn:
                new_policy = agent.pareto_update(agent.get_policy())
            
            _, reward, terminated, truncated, _ = environment.step(action)
            new_state = environment.current_location

            if learn == True: # Update Q-values if learning is specified
                alpha = linear_schedule_exp(0.005, 0.1, trials, trial)
                agent.learn_with_att(old_state, reward, new_state, action, agent.defender_policy, new_policy, alpha)
                agent.defender_policy = new_policy

            cumulative_reward += reward
            step += 1

            if environment.check_state() == 'TERMINAL': # If game is in terminal state, game over and start next trial
                environment.__init__()
                game_over = True

        reward_per_episode.append(cumulative_reward) # Append reward for current trial to performance log
    
    return reward_per_episode # Return performance log

# Define the routine - states_intersection
def states_intersection(current_belief, states):
    new_belief = set()
    for state in states:
        if state in current_belief:
            new_belief.add(state)
    return new_belief

# Define the routine - play_with_belief
def play_att_with_belief(environment, agent, trials=500, max_steps_per_episode=1000, learn=False, attack_model = None, measure_time=False):
    """The play function runs iterations and updates Q-values if desired."""
    inference_times = []  # Track inference time per action at test time, added Nov-29-2025
    reward_per_episode = [] # Initialise performance log
    if learn:
        agent.q_table_withatt = copy.deepcopy(agent.q_table)
        agent.defender_policy = agent.get_policy_naive()
    
    for trial in range(trials): # Run trials
        belief = set()
        for state in agent.all_state:
            belief.add(state)
        cumulative_reward = 0 # Initialise values of each game
        step = 0
        game_over = False
        old_state = environment.reset()
        agent.defender_policy = agent.get_policy_with_belief(agent.defender_policy, belief)
        while step < max_steps_per_episode and game_over != True: # Run until max steps or until game is finished
            old_state = tuple(environment.current_location)
            if attack_model != None:
                att_action = attack_model.predict(np.asarray(old_state))
                if att_action[0] in [0,1,2,3,4]:
                    true_action = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY'][att_action[0]]
                else:
                    true_action = att_action[0]
                if true_action == 'UP':
                    if old_state[0] == 0:
                        att_state = old_state
                    else:
                        att_state = [old_state[0] - 1, old_state[1]]
                elif true_action == 'DOWN':
                        if old_state[0] == environment.height - 1:
                            att_state = old_state
                        else:
                            att_state =  [old_state[0] + 1, old_state[1]]
                elif true_action == 'LEFT':
                        if old_state[1] == 0:
                            att_state = old_state
                        else:
                            att_state = [old_state[0], old_state[1] - 1]
                elif true_action == 'RIGHT':
                        if old_state[1] == environment.width - 1:
                            att_state = old_state
                        else:
                            att_state = [old_state[0], old_state[1] + 1]
                elif true_action == 'STAY':
                    att_state = old_state
                att_state = tuple(att_state)
            else:
                att_state = agent.get_att_state(old_state, agent.defender_policy)
                
            # Start timing inference (only at test time)
            if not learn and measure_time:
                action_start_time = time.perf_counter()
                
            if learn:
                eps = linear_schedule_exp(0.01, 0.2, trials, trial)
            else:
                eps = 0
            action = agent.choose_action_att(environment.actions , att_state, eps)
            # End timing inference
            if not learn and measure_time:
                action_end_time = time.perf_counter()
                inference_times.append(action_end_time - action_start_time)
            possible_true_states = agent.get_ball(att_state)
            belief_before_action = states_intersection(belief, possible_true_states)
            belief = set()
            for state in belief_before_action:
                belief.add(environment.take_action(state, action))
            
            new_policy = agent.get_policy_with_belief(agent.defender_policy, belief)
            
            _, reward, terminated, truncated, _ = environment.step(action)
            
            new_state = environment.current_location

            if learn == True: # Update Q-values if learning is specified
                alpha = linear_schedule_exp(0.01, 0.1, trials, trial)
                agent.learn_with_att_with_belief(old_state, reward, new_state, action, agent.defender_policy, new_policy, belief, alpha)
            agent.defender_policy = new_policy

            cumulative_reward += reward
            step += 1

            if environment.check_state() == 'TERMINAL': # If game is in terminal state, game over and start next trial
                environment.__init__()
                game_over = True

        reward_per_episode.append(cumulative_reward) # Append reward for current trial to performance log
    # Return performance log and optionally average inference time
    if measure_time and not learn:
        avg_inference_time = np.mean(inference_times) if inference_times else 0
        return reward_per_episode, avg_inference_time
    else:
        return reward_per_episode
    
def play_pessimistic_clean(environment, agent, trials=100, max_steps_per_episode=1000, measure_time=False):
    """
    Test Pessimistic Q-Learning in CLEAN environment (NO attacker).
    Agent acts greedily on TRUE states using trained Q-table.
    
    Args:
        environment: GridWorld environment
        agent: Q_Agent with trained q_table_withatt
        trials: Number of episodes
        max_steps_per_episode: Max steps per episode
        measure_time: If True, measure inference time
    
    Returns:
        If measure_time=True: (reward_per_episode, avg_inference_time)
        Otherwise: reward_per_episode
    """
    reward_per_episode = []
    inference_times = []
    
    for trial in range(trials):
        cumulative_reward = 0
        step = 0
        game_over = False
        environment.reset()
        
        while step < max_steps_per_episode and game_over != True:
            old_state = tuple(environment.current_location)
            
            # Start timing
            if measure_time:
                action_start_time = time.perf_counter()
            
            # NO ATTACKER - greedy action on true state
            q_values = agent.q_table_withatt[old_state]
            maxValue = max(q_values.values())
            action = np.random.choice([k for k, v in q_values.items() if v == maxValue])
            
            # End timing
            if measure_time:
                action_end_time = time.perf_counter()
                inference_times.append(action_end_time - action_start_time)
            
            # Take action in environment
            _, reward, terminated, truncated, _ = environment.step(action)
            
            cumulative_reward += reward
            step += 1
            
            if environment.check_state() == 'TERMINAL':
                environment.__init__()
                game_over = True
        
        reward_per_episode.append(cumulative_reward)
    
    if measure_time:
        avg_inference_time = np.mean(inference_times) if inference_times else 0
        return reward_per_episode, avg_inference_time
    else:
        return reward_per_episode   
    
 # New function: Vanilla Q-Learning
def play_vanilla_clean(environment, agent, trials=100, max_steps_per_episode=1000, measure_time=False):
    """
    Test vanilla Q-learning in CLEAN environment.
    Uses baseline Q-table (no adversarial training).
    """
    reward_per_episode = []
    inference_times = []
    
    for trial in range(trials):
        cumulative_reward = 0
        step = 0
        game_over = False
        environment.reset()
        
        while step < max_steps_per_episode and game_over != True:
            old_state = tuple(environment.current_location)
            
            if measure_time:
                action_start_time = time.perf_counter()
            
            # Greedy action selection from baseline Q-table
            q_values = agent.q_table[old_state]
            maxValue = max(q_values.values())
            action = np.random.choice([k for k, v in q_values.items() if v == maxValue])
            
            if measure_time:
                action_end_time = time.perf_counter()
                inference_times.append(action_end_time - action_start_time)
            
            _, reward, terminated, truncated, _ = environment.step(action)
            cumulative_reward += reward
            step += 1
            
            if environment.check_state() == 'TERMINAL':
                environment.__init__()
                game_over = True
        
        reward_per_episode.append(cumulative_reward)
    
    if measure_time:
        avg_inference_time = np.mean(inference_times) if inference_times else 0
        return reward_per_episode, avg_inference_time
    else:
        return reward_per_episode

def play_vanilla_vs_attacker(environment, agent, trials=100, max_steps_per_episode=1000, attack_model=None):
    """
    Test vanilla Q-learning against attacker.
    Same greedy testing as Off-Policy, but uses baseline Q-table (no adversarial training).
    """
    reward_per_episode = []
    
    for trial in range(trials):
        cumulative_reward = 0
        step = 0
        game_over = False
        environment.reset()
        
        while step < max_steps_per_episode and game_over != True:
            old_state = tuple(environment.current_location)
            
            if attack_model is not None:
                # Attacker perturbs observation
                attack_obs = np.array(old_state)
                attack_action, _ = attack_model.predict(attack_obs, deterministic=True)
                attack_action = int(attack_action) if attack_action.ndim == 0 else int(attack_action[0])
                
                if attack_action in [0, 1, 2, 3, 4]:
                    perturbation_direction = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY'][attack_action]
                else:
                    perturbation_direction = attack_action
                
                perturbed_state = apply_perturbation(old_state, perturbation_direction)
                q_values = agent.q_table[perturbed_state]  # Uses BASELINE q_table
            else:
                q_values = agent.q_table[old_state]
            
            # Greedy action selection
            maxValue = max(q_values.values())
            action = np.random.choice([k for k, v in q_values.items() if v == maxValue])
            
            _, reward, terminated, truncated, _ = environment.step(action)
            cumulative_reward += reward
            step += 1
            
            if environment.check_state() == 'TERMINAL':
                environment.__init__()
                game_over = True
        
        reward_per_episode.append(cumulative_reward)
    
    return reward_per_episode   

# New function: Play with CAR Operator
def play_with_car_operator(environment, agent, trials=500, max_steps_per_episode=1000, learn=False, measure_time=False, attack_model=None):
    """
    Play episodes using CAR (Consistent Adversarial Robust) operator.
    
    Training: Uses CAR operator for both behavior and target policy
    Testing: Uses greedy action selection from CAR operator
    
    Args:
        environment: GridWorld environment
        agent: Q_Agent with trained q_table_withatt
        trials: Number of episodes to run
        max_steps_per_episode: Max steps per episode
        learn: If True, train the agent; if False, test
        measure_time: If True, measure inference time during testing
        attack_model: Optional attacker model for adversarial testing
    
    Returns:
        If measure_time=True and learn=False: (reward_per_episode, avg_inference_time)
        Otherwise: reward_per_episode
    """
    reward_per_episode = []
    inference_times = []  # Track inference time per action at test time, added Nov-29-2025
    
    if learn:
        # Initialize adversarial Q-table
        agent.q_table_withatt = copy.deepcopy(agent.q_table)
    
    for trial in range(trials):
        cumulative_reward = 0
        step = 0
        game_over = False
        old_state = environment.reset()
        
        while step < max_steps_per_episode and game_over != True:
            old_state = tuple(environment.current_location)
            
            # Epsilon schedule
            if learn:
                eps = linear_schedule_exp(0.01, 0.2, trials, trial)
            else:
                eps = 0
            
            # Start timing for test-time inference
            if not learn and measure_time:
                action_start_time = time.perf_counter()
            
            # Action selection (works for both training and testing)
            # Check if we're testing against an attacker
            if attack_model is not None and not learn:
                # Adversarial testing: Attacker perturbs observation
                # Attacker sees true state and chooses perturbation direction
                attack_obs = np.array(old_state)
                attack_action, _ = attack_model.predict(attack_obs, deterministic=True)
                attack_action = int(attack_action.item() if hasattr(attack_action, "item") else attack_action)


                
                # Apply perturbation to get perturbed state
                if attack_action in [0, 1, 2, 3, 4]:
                    perturbation_direction = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY'][int(attack_action)]
                else:
                    perturbation_direction = attack_action
                
                perturbed_state = apply_perturbation(old_state, perturbation_direction)
                
                # Agent acts greedily on PERTURBED observation
                action = agent.choose_action_car(environment.actions, perturbed_state, eps=0)
            else:
                # Normal operation (training or testing without attacker)
                action = agent.choose_action_car(environment.actions, old_state, eps)
            
            # End timing
            if not learn and measure_time:
                action_end_time = time.perf_counter()
                inference_times.append(action_end_time - action_start_time)
            
            # Take action in environment
            _, reward, terminated, truncated, _ = environment.step(action)
            new_state = environment.current_location
            
            # Learning update
            if learn:
                alpha = linear_schedule_exp(0.01, 0.1, trials, trial)
                agent.learn_with_car_operator(old_state, reward, new_state, action, alpha)
            
            cumulative_reward += reward
            step += 1
            
            if environment.check_state() == 'TERMINAL':
                environment.__init__()
                game_over = True
        
        reward_per_episode.append(cumulative_reward)
    
    if measure_time and not learn:
        avg_inference_time = np.mean(inference_times) if inference_times else 0
        return reward_per_episode, avg_inference_time
    else:
        return reward_per_episode

# Helper function for applying perturbations
def apply_perturbation(state, perturbation_direction):
    """Apply a perturbation to a state based on direction."""
    x, y = state
    
    if perturbation_direction == 'UP':
        if x > 0:
            return (x - 1, y)
    elif perturbation_direction == 'DOWN':
        if x < 4:
            return (x + 1, y)
    elif perturbation_direction == 'LEFT':
        if y > 0:
            return (x, y - 1)
    elif perturbation_direction == 'RIGHT':
        if y < 4:
            return (x, y + 1)
    
    # If can't move or stay, return original state
    return state

# Nov-27-2025: New function - play_offpolicy_greedy_target
def play_offpolicy_greedy_target(environment, agent, trials=500, max_steps_per_episode=1000, learn=False, measure_time=False, attack_model=None):
    """
    Off-policy approach: Behavior policy uses pessimistic max-min with belief tracking,
    but target policy is Greedy (simple max over Q-values).
    
    This combines safety during exploration (pessimistic behavior) with 
    optimism in value estimation (greedy target).
    
    Args:
        environment: GridWorld environment
        agent: Q_Agent with q_table and q_table_withatt
        trials: Number of episodes
        max_steps_per_episode: Max steps per episode
        learn: If True, train; if False, test
        measure_time: If True, measure inference time during testing
        attack_model: Optional attacker for adversarial testing
    
    Returns:
        If measure_time=True and learn=False: (reward_per_episode, avg_inference_time)
        Otherwise: reward_per_episode
    """
    reward_per_episode = []
    inference_times = []
    
    if learn:
        # Initialize adversarial Q-table and defender policy
        agent.q_table_withatt = copy.deepcopy(agent.q_table)
        agent.defender_policy = agent.pareto_update(agent.get_policy(init=True))
    
    for trial in range(trials):
        cumulative_reward = 0
        step = 0
        game_over = False
        old_state = environment.reset()
        
        if learn:
            # TRAINING: Use belief tracking for safety
            belief = set()
            for state in agent.all_state:
                belief.add(state)
            agent.defender_policy = agent.get_policy_with_belief(agent.defender_policy, belief)
            
            while step < max_steps_per_episode and game_over != True:
                old_state = tuple(environment.current_location)
                
                # Behavior policy: pessimistic (worst-case over belief)
                att_state = agent.get_att_state(old_state, agent.defender_policy)
                
                eps = linear_schedule_exp(0.01, 0.2, trials, trial)
                action = agent.choose_action_att(environment.actions, att_state, eps)
                
                # Update belief
                possible_true_states = agent.get_ball(att_state)
                belief_before_action = states_intersection(belief, possible_true_states)
                belief = set()
                for state in belief_before_action:
                    belief.add(environment.take_action(state, action))
                
                # Update policy with new belief (for next iteration)
                new_policy = agent.get_policy_with_belief(agent.defender_policy, belief)
                
                # Take action in environment
                _, reward, terminated, truncated, _ = environment.step(action)
                new_state = environment.current_location
                
                # TARGET POLICY: Use Greedy (this is the key difference!)
                # NOT max-min like pessimistic, just simple max
                alpha = linear_schedule_exp(0.01, 0.1, trials, trial)
                agent.learn_with_greedy_target(old_state, reward, new_state, action, alpha)
                
                # Update defender policy for next step
                agent.defender_policy = new_policy
                
                cumulative_reward += reward
                step += 1
                
                if environment.check_state() == 'TERMINAL':
                    environment.__init__()
                    game_over = True
        
        else:
            # TESTING: Just use Greedy (no belief tracking)
            # This is the whole point - fast inference
            while step < max_steps_per_episode and game_over != True:
                old_state = tuple(environment.current_location)
                
                eps = 0
                
                # Start timing
                if measure_time:
                    action_start_time = time.perf_counter()
                
                # Check if we're testing against an attacker
                if attack_model is not None:
                    # ADVERSARIAL TESTING: Attacker perturbs observation
                    # Attacker sees true state and chooses perturbation
                    attack_obs = np.array(old_state)
                    attack_action, _ = attack_model.predict(attack_obs, deterministic=True)
                    attack_action = int(attack_action.item() if hasattr(attack_action, "item") else attack_action)

                    
                    # Convert integer action to string direction
                    if attack_action in [0, 1, 2, 3, 4]:
                        perturbation_direction = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY'][int(attack_action)]
                    else:
                        perturbation_direction = attack_action
                    
                    # Apply perturbation to get perturbed state
                    perturbed_state = apply_perturbation(old_state, perturbation_direction)
                    
                    # Agent acts on PERTURBED observation (doesn't know it's perturbed)
                    q_values_of_state = agent.q_table_withatt[perturbed_state]
                else:
                    # No attack: Use true state
                    q_values_of_state = agent.q_table_withatt[old_state]
                
                # Greedy action selection
                maxValue = max(q_values_of_state.values())
                action = np.random.choice([k for k, v in q_values_of_state.items() if v == maxValue])
                
                # End timing
                if measure_time:
                    action_end_time = time.perf_counter()
                    inference_times.append(action_end_time - action_start_time)
                
                # Take action in environment
                _, reward, terminated, truncated, _ = environment.step(action)
                new_state = environment.current_location
                
                cumulative_reward += reward
                step += 1
                
                if environment.check_state() == 'TERMINAL':
                    environment.__init__()
                    game_over = True
        
        reward_per_episode.append(cumulative_reward)
    
    if measure_time and not learn:
        avg_inference_time = np.mean(inference_times) if inference_times else 0
        return reward_per_episode, avg_inference_time
    else:
        return reward_per_episode


# Define the RL for training with attack vector 
def train_RL_attack_with_belief(agentQ, agent_type='pessimistic', model_name='attacker'):
    """
    Train attacker against specific agent type.
    
    This creates separate attackers for each algorithm.
    Each attacker learns to exploit the specific weaknesses of its target algorithm.
    
    Args:
        agentQ: The trained agent to attack
        agent_type: Type of agent - 'pessimistic', 'car', or 'offpolicy'
        model_name: Base name for saved model file (default: 'attacker')
    
    Returns:
        Trained attack model
    
    The attacker learns by playing against the agent in GridWorld_RL_Attack environment.
    It observes the agent's behavior and learns which perturbations are most effective.
    """
    print(f"\n{'='*70}")
    print(f"TRAINING ATTACKER AGAINST {agent_type.upper()} AGENT")
    print(f"{'='*70}")
    print(f"Agent type: {agent_type}")
    print(f"Model name: {model_name}_{agent_type}")
    print(f"Training timesteps: 200,000")
    print(f"{'='*70}\n")
    
    # Create attack environment with correct agent type
    env = GridWorld_RL_Attack(agentQ, agent_type=agent_type)
    
    # Setup checkpoint callback with agent-specific path
    checkpoint_path = f"RL_attack/{agent_type}"
    checkpoint_callback = CheckpointCallback(
        save_freq=5000, 
        save_path=checkpoint_path,
        name_prefix=f'{model_name}_{agent_type}'
    )
    
    # Train attacker
    model = DQN(
        "MlpPolicy", 
        env, 
        learning_rate=1e-3, 
        learning_starts=1000, 
        batch_size=256, 
        exploration_final_eps=0.001,
        gamma=1, 
        tau=0.2, 
        train_freq=(5, "step"), 
        policy_kwargs={"net_arch": [256, 128]}, 
        tensorboard_log=f"RL_attack/{agent_type}/", 
        verbose=1
    )
    
    # Train the attacker
    print(f"\nTraining attacker against {agent_type} agent...")
    model.learn(total_timesteps=200000, callback=checkpoint_callback)
    
    # Save final model with descriptive name
    final_model_path = f"{model_name}_{agent_type}"
    model.save(final_model_path)
    
    print(f"\n{'='*70}")
    print(f"ATTACKER TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Model saved as: {final_model_path}.zip")
    print(f"Checkpoints saved to: {checkpoint_path}/")
    print(f"{'='*70}\n")
    
    return model

# New function CSV Logging
def log_experiment_to_csv(experiment_number, results_dict, filename='gridworld_experiments.csv'):
    """
    Logs experiment results to a CSV file.
    
    Parameters:
    -----------
    experiment_number : int
        The experiment iteration number
    results_dict : dict
        Dictionary containing all experiment results
    filename : str
        Name of the CSV file to save results
    """
    import csv
    import os
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(filename)
    
    # Define all possible fields
    # Updated on Nov-25-2025 to add more fields
    fieldnames = [
    'experiment_number',
    'timestamp',
    'seed',
    'baseline_avg_reward',
    'vanilla_clean_reward',
    'vanilla_clean_std',
    'vanilla_inference_time_ms',
    'vanilla_vs_attacker_reward',
    'vanilla_vs_attacker_std',
    'pessimistic_train_avg_reward',
    'pessimistic_train_std',
    'pessimistic_clean_reward',              
    'pessimistic_clean_std',                
    'pessimistic_clean_inference_time_ms',   
    'pessimistic_vs_attacker_reward',        
    'pessimistic_vs_attacker_std',           
    'car_train_avg_reward',
    'car_train_std',
    'car_test_avg_reward',
    'car_test_std',
    'car_inference_time_ms',                 
    'car_vs_attacker_reward',
    'car_vs_attacker_std',
    'offpolicy_greedy_train_avg_reward',
    'offpolicy_greedy_train_std',
    'offpolicy_greedy_test_avg_reward',      
    'offpolicy_greedy_test_std',
    'offpolicy_greedy_inference_time_ms',
    'offpolicy_greedy_vs_attacker_reward',
    'offpolicy_greedy_vs_attacker_std',
    'performance_diff_offpolicy_vs_pessimistic',
    'performance_diff_offpolicy_vs_car',
    'performance_diff_car_vs_pessimistic',
    'speedup_offpolicy_vs_pessimistic',
    'speedup_offpolicy_vs_car',
]
    
    # Add timestamp
    results_dict['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results_dict['experiment_number'] = experiment_number
    
    # Write to CSV
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header only if file is new
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(results_dict)
    
    print(f"  ✓ Experiment {experiment_number} logged to {filename}")

# NEW FUNCTION: Save Checkpoint
def save_checkpoint(experiment_number, checkpoint_file='experiment_checkpoint.pkl'):
    """
    Save the current experiment number to a checkpoint file.
    
    Parameters:
    -----------
    experiment_number : int
        Current experiment number to save
    checkpoint_file : str
        Name of the checkpoint file
    """
    import pickle
    import os
    
    checkpoint_data = {
        'last_completed_experiment': experiment_number,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)

# NEW FUNCTION: Load Checkpoint
def load_checkpoint(checkpoint_file='experiment_checkpoint.pkl'):
    """
    Load the last completed experiment number from checkpoint file.
    
    Parameters:
    -----------
    checkpoint_file : str
        Name of the checkpoint file
        
    Returns:
    --------
    int : Last completed experiment number, or 0 if no checkpoint exists
    """
    import pickle
    import os
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            last_exp = checkpoint_data['last_completed_experiment']
            timestamp = checkpoint_data['timestamp']
            
            print(f"\n{'='*70}")
            print(f"CHECKPOINT FOUND")
            print(f"{'='*70}")
            print(f"Last completed experiment: {last_exp}")
            print(f"Checkpoint timestamp: {timestamp}")
            print(f"{'='*70}\n")
            
            return last_exp
        except Exception as e:
            print(f"Warning: Could not load checkpoint - {e}")
            print("Starting from experiment 1")
            return 0
    else:
        print(f"\nNo checkpoint found. Starting from experiment 1.")
        return 0

# NEW FUNCTION: Clear Checkpoint
def clear_checkpoint(checkpoint_file='experiment_checkpoint.pkl'):
    """
    Clear the checkpoint file (call this when all experiments are complete).
    
    Parameters:
    -----------
    checkpoint_file : str
        Name of the checkpoint file to remove
    """
    import os
    
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"✓ Checkpoint file '{checkpoint_file}' removed")

# NEW FUNCTION: Check Progress
def check_experiment_progress(csv_filename='gridworld_experiments.csv', 
                              checkpoint_file='experiment_checkpoint.pkl',
                              total_experiments=500):
    """
    Check the progress of experiments without running them.
    
    Parameters:
    -----------
    csv_filename : str
        Name of CSV file with results
    checkpoint_file : str
        Name of checkpoint file
    total_experiments : int
        Total number of experiments planned
    """
    import os
    import pandas as pd
    
    print("="*70)
    print("EXPERIMENT PROGRESS CHECK")
    print("="*70)
    
    # Check CSV
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        completed = len(df)
        print(f"\n✓ CSV file found: {csv_filename}")
        print(f"  Experiments completed: {completed}/{total_experiments} ({completed/total_experiments*100:.1f}%)")
        print(f"  Last experiment: {df['experiment_number'].max()}")
        
            
        print(f"\n  Average Rewards:")
        if 'pessimistic_test_avg_reward' in df.columns:
                print(f"    Pessimistic: {df['pessimistic_test_avg_reward'].mean():.2f}")
        if 'car_test_avg_reward' in df.columns:
                print(f"    CAR:         {df['car_test_avg_reward'].mean():.2f}")
        if 'offpolicy_greedy_test_avg_reward' in df.columns:
                print(f"    Off-Policy:  {df['offpolicy_greedy_test_avg_reward'].mean():.2f}")
    else:
        print(f"\n  No CSV file found: {csv_filename}")
        completed = 0
    
    # Check checkpoint
    print(f"\n" + "-"*70)
    if os.path.exists(checkpoint_file):
        last_completed = load_checkpoint(checkpoint_file)
        print(f"  Next experiment will be: {last_completed + 1}")
    else:
        print(f" No checkpoint file found: {checkpoint_file}")
        print(f"  Starting fresh if you run experiments")
    
    # Estimate time remaining
    if completed > 0:
        print(f"\n" + "-"*70)
        remaining = total_experiments - completed
        avg_time_per_exp = 12  # minutes (rough estimate)
        estimated_hours = (remaining * avg_time_per_exp) / 60
        print(f"Estimated time remaining: ~{estimated_hours:.1f} hours")
        print(f"  ({remaining} experiments × ~{avg_time_per_exp} min/exp)")
    
    print("="*70)
    
    return completed

# NEW FUNCTION: Analyze CSV Results updated on Nov-25-2025
def analyze_csv_results(csv_filename='gridworld_experiments.csv'):
    """
    Analyze and visualize results from the CSV file.
    Updated Nov-27-2025 to handle 3 methods (Pessimistic, CAR, Off-Policy Greedy)
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Read CSV
    df = pd.read_csv(csv_filename)
    
    print("="*70)
    print(f"ANALYSIS OF {len(df)} EXPERIMENTS")
    print("="*70)
    
    # Summary statistics
    print("\n--- Vanilla Q-Learning (Baseline) ---")
    if 'vanilla_clean_reward' in df.columns:
        print(f"Mean test reward: {df['vanilla_clean_reward'].mean():.2f} ± {df['vanilla_clean_reward'].std():.2f}")
    if 'vanilla_vs_attacker_reward' in df.columns:
        print(f"Mean vs attacker: {df['vanilla_vs_attacker_reward'].mean():.2f} ± {df['vanilla_vs_attacker_reward'].std():.2f}")

    print("\n--- Pessimistic Q-Learning ---")
    if 'pessimistic_clean_reward' in df.columns:
      print(f"Mean test reward: {df['pessimistic_clean_reward'].mean():.2f} ± {df['pessimistic_clean_reward'].std():.2f}")
    if 'pessimistic_vs_attacker_reward' in df.columns:
      print(f"Mean vs attacker: {df['pessimistic_vs_attacker_reward'].mean():.2f} ± {df['pessimistic_vs_attacker_reward'].std():.2f}")
    
    print("\n--- CAR Operator ---")
    if 'car_test_avg_reward' in df.columns:
        print(f"Mean test reward: {df['car_test_avg_reward'].mean():.2f} ± {df['car_test_avg_reward'].std():.2f}")
    if 'car_vs_attacker_reward' in df.columns:
        print(f"Mean vs attacker: {df['car_vs_attacker_reward'].mean():.2f} ± {df['car_vs_attacker_reward'].std():.2f}")
    
    # Check if off-policy columns exist
    if 'offpolicy_greedy_test_avg_reward' in df.columns:
        print("\n--- Off-Policy Greedy Target ---")
        print(f"Mean test reward: {df['offpolicy_greedy_test_avg_reward'].mean():.2f} ± {df['offpolicy_greedy_test_avg_reward'].std():.2f}")
        if 'offpolicy_greedy_vs_attacker_reward' in df.columns:
            print(f"Mean vs attacker: {df['offpolicy_greedy_vs_attacker_reward'].mean():.2f} ± {df['offpolicy_greedy_vs_attacker_reward'].std():.2f}")
        
        print("\n--- Performance Comparisons ---")
        if 'performance_diff_offpolicy_vs_pessimistic' in df.columns:
            print(f"Off-Policy vs Pessimistic: {df['performance_diff_offpolicy_vs_pessimistic'].mean():.3f}")
        if 'performance_diff_offpolicy_vs_car' in df.columns:
            print(f"Off-Policy vs CAR:         {df['performance_diff_offpolicy_vs_car'].mean():.3f}")
        if 'performance_diff_car_vs_pessimistic' in df.columns:
            print(f"CAR vs Pessimistic:        {df['performance_diff_car_vs_pessimistic'].mean():.3f}")
    
    
    # Simple summary print - no plots for now
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
    
    return df

def run_single_experiment_with_seed(experiment_num, seed, attack_models=None):
    """
    Run ONE complete experiment with fixed random seed for fair comparison.
    Tests all three algorithms (Off-Policy, Pessimistic, CAR) with same initial conditions.
    
    Args:
        experiment_num: Experiment number (for tracking)
        seed: Random seed to ensure reproducibility
        attack_models: Dict with keys 'offpolicy', 'pessimistic', 'car' 
                      containing trained attacker models (optional)
    
    Returns:
        results: Dict containing all test results
        agentQ1: Trained Off-Policy agent
        agentQ2: Trained Pessimistic agent  
        agentQ3: Trained CAR agent
    """
    results = {}
    results['experiment_number'] = experiment_num
    results['seed'] = seed
    
    # ============================================================
    # PHASE 1: BASELINE TRAINING (Same for all three algorithms)
    # ============================================================
    print(f"\n[Exp {experiment_num + 1}] Phase 1: Baseline training...")
    
    environment_baseline = GridWorld()
    environment_baseline.seed(seed)
    agent_baseline = Q_Agent(environment_baseline)
    
    # Set numpy random seed for reproducibility (offset to avoid correlation)
    np.random.seed(seed + 1000)
    
    # Train baseline with vanilla Q-learning
    reward_baseline = play(environment_baseline, agent_baseline, trials=2000, learn=True)
    results['baseline_avg_reward'] = np.mean(reward_baseline[-100:])
    
    # Store baseline Q-table to initialize all three algorithms
    baseline_q_table = copy.deepcopy(agent_baseline.q_table)
    
    # ============================================================
    # PHASE 1.5: TEST VANILLA Q-LEARNING (BASELINE COMPARISON)
    # ============================================================
    print(f"[Exp {experiment_num + 1}] Phase 1.5: Testing Vanilla Q-Learning...")

    # Test WITHOUT attacker
    np.random.seed(seed + 1000)
    reward_vanilla_clean, vanilla_time = play_vanilla_clean(
        environment_baseline, agent_baseline, trials=100, measure_time=True
    )
    results['vanilla_clean_reward'] = np.mean(reward_vanilla_clean)
    results['vanilla_clean_std'] = np.std(reward_vanilla_clean)
    results['vanilla_inference_time_ms'] = vanilla_time * 1000

    # Test WITH attacker (if available)
    #print(f"[DEBUG] attack_models is: {attack_models}")
    if attack_models is not None and 'offpolicy' in attack_models:
        #print(f"[DEBUG] Running vanilla vs attacker test...")
        np.random.seed(seed + 1000)
        reward_vanilla_attack = play_vanilla_vs_attacker(
            environment_baseline, agent_baseline, trials=100,
            attack_model=attack_models['offpolicy']
        )
        results['vanilla_vs_attacker_reward'] = np.mean(reward_vanilla_attack)
        results['vanilla_vs_attacker_std'] = np.std(reward_vanilla_attack)

    # ============================================================
    # PHASE 2: OFF-POLICY GREEDY TARGET
    # ============================================================
    print(f"[Exp {experiment_num + 1}] Phase 2: Training Off-Policy Greedy...")
    
    # Create new environment with SAME seed
    environment1 = GridWorld()
    environment1.seed(seed)
    agentQ1 = Q_Agent(environment1)
    agentQ1.q_table = copy.deepcopy(baseline_q_table)
    
    # Train Off-Policy with same random seed (offset)
    np.random.seed(seed + 1000)
    reward_train1 = play_offpolicy_greedy_target(environment1, agentQ1, trials=500, learn=True)
    results['offpolicy_greedy_train_avg_reward'] = np.mean(reward_train1)
    results['offpolicy_greedy_train_std'] = np.std(reward_train1)
    
    # Test WITHOUT attacker
    np.random.seed(seed + 1000)  # Reset for consistent testing
    reward_test1, offpolicy_time = play_offpolicy_greedy_target(
        environment1, agentQ1, trials=100, learn=False, measure_time=True
    )
    results['offpolicy_greedy_test_avg_reward'] = np.mean(reward_test1)
    results['offpolicy_greedy_test_std'] = np.std(reward_test1)
    results['offpolicy_greedy_inference_time_ms'] = offpolicy_time * 1000
    
    # Test WITH attacker (if available)
    if attack_models is not None and 'offpolicy' in attack_models:
        np.random.seed(seed + 1000)  # Reset for consistent testing
        reward_test1_attack = play_offpolicy_greedy_target(
            environment1, agentQ1, trials=100, learn=False, 
            attack_model=attack_models['offpolicy']
        )
        results['offpolicy_greedy_vs_attacker_reward'] = np.mean(reward_test1_attack)
        results['offpolicy_greedy_vs_attacker_std'] = np.std(reward_test1_attack)
    
    # ============================================================
    # PHASE 3: PESSIMISTIC Q-LEARNING
    # ============================================================
    print(f"[Exp {experiment_num + 1}] Phase 3: Training Pessimistic Q-Learning...")
    
    # Create new environment with SAME seed
    environment2 = GridWorld()
    environment2.seed(seed)
    agentQ2 = Q_Agent(environment2)
    agentQ2.q_table = copy.deepcopy(baseline_q_table)
    
    # Train Pessimistic with same random seed (offset)
    np.random.seed(seed + 1000)
    reward_train2 = play_att_with_belief(environment2, agentQ2, trials=500, learn=True)
    results['pessimistic_train_avg_reward'] = np.mean(reward_train2)
    results['pessimistic_train_std'] = np.std(reward_train2)
    
   # Test in CLEAN environment (NO attacker) - NEW!
    np.random.seed(seed + 1000)  # Reset for consistent testing
    reward_clean2, pessimistic_clean_time = play_pessimistic_clean(
    environment2, agentQ2, trials=100, measure_time=True)
    results['pessimistic_clean_reward'] = np.mean(reward_clean2)
    results['pessimistic_clean_std'] = np.std(reward_clean2)
    results['pessimistic_clean_inference_time_ms'] = pessimistic_clean_time * 1000
    
    # Test WITH attacker (if available)
    if attack_models is not None and 'pessimistic' in attack_models:
        np.random.seed(seed + 1000)  # Reset for consistent testing
        reward_test2_attack = play_att_with_belief(
            environment2, agentQ2, trials=100, learn=False,
            attack_model=attack_models['pessimistic']
        )
        results['pessimistic_vs_attacker_reward'] = np.mean(reward_test2_attack)
        results['pessimistic_vs_attacker_std'] = np.std(reward_test2_attack)
    
    # ============================================================
    # PHASE 4: CAR OPERATOR
    # ============================================================
    print(f"[Exp {experiment_num + 1}] Phase 4: Training CAR Operator...")
    
    # Create new environment with SAME seed
    environment3 = GridWorld()
    environment3.seed(seed)
    agentQ3 = Q_Agent(environment3)
    agentQ3.q_table = copy.deepcopy(baseline_q_table)
    
    # Train CAR with same random seed (offset)
    np.random.seed(seed + 1000)
    reward_train3 = play_with_car_operator(environment3, agentQ3, trials=500, learn=True)
    results['car_train_avg_reward'] = np.mean(reward_train3)
    results['car_train_std'] = np.std(reward_train3)
    
    # Test WITHOUT attacker
    np.random.seed(seed + 1000)  # Reset for consistent testing
    reward_test3, car_time = play_with_car_operator(
        environment3, agentQ3, trials=100, learn=False, measure_time=True
    )
    results['car_test_avg_reward'] = np.mean(reward_test3)
    results['car_test_std'] = np.std(reward_test3)
    results['car_inference_time_ms'] = car_time * 1000
    
    # Test WITH attacker (if available)
    if attack_models is not None and 'car' in attack_models:
        np.random.seed(seed + 1000)  # Reset for consistent testing
        reward_test3_attack = play_with_car_operator(
            environment3, agentQ3, trials=100, learn=False,
            attack_model=attack_models['car']
        )
        results['car_vs_attacker_reward'] = np.mean(reward_test3_attack)
        results['car_vs_attacker_std'] = np.std(reward_test3_attack)
    
    # ============================================================
# PHASE 5: CALCULATE PERFORMANCE METRICS
# ============================================================
    
# Performance differences 
    results['performance_diff_car_vs_pessimistic'] = results['car_test_avg_reward'] - results['pessimistic_clean_reward']
    results['performance_diff_offpolicy_vs_pessimistic'] = results['offpolicy_greedy_test_avg_reward'] - results['pessimistic_clean_reward']
    results['performance_diff_offpolicy_vs_car'] = results['offpolicy_greedy_test_avg_reward'] - results['car_test_avg_reward']
    
# Speedup calculations
    if results['pessimistic_clean_inference_time_ms'] > 0:
        results['speedup_offpolicy_vs_pessimistic'] = results['pessimistic_clean_inference_time_ms'] / results['offpolicy_greedy_inference_time_ms']
    if results['car_inference_time_ms'] > 0:
        results['speedup_offpolicy_vs_car'] = results['car_inference_time_ms'] / results['offpolicy_greedy_inference_time_ms']

    print(f"[Exp {experiment_num + 1}] Complete!")
    print(f"  Vanilla:     {results['vanilla_clean_reward']:.2f}")
    print(f"  Off-Policy:  {results['offpolicy_greedy_test_avg_reward']:.2f}")
    print(f"  Pessimistic: {results['pessimistic_clean_reward']:.2f}")  
    print(f"  CAR:         {results['car_test_avg_reward']:.2f}")
    
    return results, agentQ1, agentQ2, agentQ3
    
def visualize_experiment_results(csv_filename='gridworld_experiments.csv'):
    """
    Create comprehensive visualizations from experiment CSV results.
    
    Creates:
        - Distribution histograms (3 plots)
        - Box plot comparison
        - Cumulative distribution function
        - Scatter plot over experiments
        - Saves as 'experiment_visualizations.png'
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        print("Warning: seaborn not installed, using default matplotlib style")
    
    # Read CSV
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Error: {csv_filename} not found!")
        return None
    
    # Check if required columns exist
    # Check if required columns exist
    required_cols = [
    'vanilla_clean_reward',
    'offpolicy_greedy_test_avg_reward',
    'pessimistic_clean_reward',
    'car_test_avg_reward'
]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Error: CSV missing columns: {missing}")
        print("This CSV may be from an old experiment format.")
        return None
    
    print(f"\n{'='*70}")
    print(f"VISUALIZING {len(df)} EXPERIMENTS")
    print(f"File: {csv_filename}")
    print(f"{'='*70}")
    
    # Extract test rewards (without attacker)
    vanilla_rewards = df['vanilla_clean_reward'].values
    offpolicy_rewards = df['offpolicy_greedy_test_avg_reward'].values
    pessimistic_rewards = df['pessimistic_clean_reward'].values
    car_rewards = df['car_test_avg_reward'].values
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 12))
    
    # ========== PLOT 1: Off-Policy Distribution ==========
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(offpolicy_rewards, bins=50, color='green', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(offpolicy_rewards), color='darkgreen', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(offpolicy_rewards):.2f}')
    ax1.axvline(np.median(offpolicy_rewards), color='lime', linestyle=':', linewidth=2,
                label=f'Median: {np.median(offpolicy_rewards):.2f}')
    ax1.set_title('Off-Policy Greedy Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Test Reward')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # ========== PLOT 2: Pessimistic Distribution ==========
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(pessimistic_rewards, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(pessimistic_rewards), color='darkblue', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(pessimistic_rewards):.2f}')
    ax2.axvline(np.median(pessimistic_rewards), color='cyan', linestyle=':', linewidth=2,
                label=f'Median: {np.median(pessimistic_rewards):.2f}')
    ax2.set_title('Pessimistic Q-Learning Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Test Reward')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # ========== PLOT 3: CAR Distribution ==========
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(car_rewards, bins=50, color='red', alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(car_rewards), color='darkred', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(car_rewards):.2f}')
    ax3.axvline(np.median(car_rewards), color='orange', linestyle=':', linewidth=2,
                label=f'Median: {np.median(car_rewards):.2f}')
    ax3.set_title('CAR Operator Distribution (High Variance)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Test Reward')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # ========== PLOT 4: Box Plot Comparison ==========
    ax4 = plt.subplot(2, 3, 4)
    data = [offpolicy_rewards, pessimistic_rewards, car_rewards]
    labels = ['Off-Policy', 'Pessimistic', 'CAR']
    bp = ax4.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightblue')
    bp['boxes'][2].set_facecolor('lightcoral')
    ax4.axhline(y=5.0, color='black', linestyle=':', linewidth=2, label='Success Threshold (5.0)')
    ax4.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax4.set_ylabel('Test Reward')
    ax4.set_title('Performance Comparison (Box Plot)', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # ========== PLOT 5: Cumulative Distribution Function ==========
    ax5 = plt.subplot(2, 3, 5)
    sorted_offpolicy = np.sort(offpolicy_rewards)
    sorted_pessimistic = np.sort(pessimistic_rewards)
    sorted_car = np.sort(car_rewards)
    n = len(df)
    ax5.plot(sorted_offpolicy, np.arange(1, n+1)/n, label='Off-Policy', 
             color='green', linewidth=2)
    ax5.plot(sorted_pessimistic, np.arange(1, n+1)/n, label='Pessimistic', 
             color='blue', linewidth=2)
    ax5.plot(sorted_car, np.arange(1, n+1)/n, label='CAR', 
             color='red', linewidth=2)
    ax5.axvline(x=5.0, color='black', linestyle=':', linewidth=2, label='Success Threshold')
    ax5.set_xlabel('Test Reward')
    ax5.set_ylabel('Cumulative Probability')
    ax5.set_title('Cumulative Distribution (Shows CAR Bimodal)', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # ========== PLOT 6: Scatter Over Experiments ==========
    ax6 = plt.subplot(2, 3, 6)
    experiments = np.arange(1, n+1)
    ax6.scatter(experiments, offpolicy_rewards, alpha=0.6, s=15, color='green', label='Off-Policy')
    ax6.scatter(experiments, pessimistic_rewards, alpha=0.6, s=15, color='blue', label='Pessimistic')
    ax6.scatter(experiments, car_rewards, alpha=0.6, s=15, color='red', label='CAR')
    ax6.axhline(y=5.0, color='black', linestyle=':', linewidth=2, label='Success Threshold')
    ax6.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax6.set_xlabel('Experiment Number')
    ax6.set_ylabel('Test Reward')
    ax6.set_title('Rewards Across All Experiments', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_visualizations.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: experiment_visualizations.png")
    plt.show()
    
    # ========== Print Statistics ==========
    print("\n" + "="*70)
    print("STATISTICS SUMMARY")
    print("="*70)
    
    print(f"\nOff-Policy Greedy:")
    print(f"  Mean:    {np.mean(offpolicy_rewards):7.2f} ± {np.std(offpolicy_rewards):.2f}")
    print(f"  Median:  {np.median(offpolicy_rewards):7.2f}")
    print(f"  Min:     {np.min(offpolicy_rewards):7.2f}")
    print(f"  Max:     {np.max(offpolicy_rewards):7.2f}")
    success_rate = np.sum(offpolicy_rewards > 5) / n * 100
    print(f"  Success: {np.sum(offpolicy_rewards > 5)}/{n} ({success_rate:.1f}%)")
    
    print(f"\nPessimistic Q-Learning:")
    print(f"  Mean:    {np.mean(pessimistic_rewards):7.2f} ± {np.std(pessimistic_rewards):.2f}")
    print(f"  Median:  {np.median(pessimistic_rewards):7.2f}")
    print(f"  Min:     {np.min(pessimistic_rewards):7.2f}")
    print(f"  Max:     {np.max(pessimistic_rewards):7.2f}")
    success_rate = np.sum(pessimistic_rewards > 5) / n * 100
    print(f"  Success: {np.sum(pessimistic_rewards > 5)}/{n} ({success_rate:.1f}%)")
    
    print(f"\nCAR Operator:")
    print(f"  Mean:    {np.mean(car_rewards):7.2f} ± {np.std(car_rewards):.2f}")
    print(f"  Median:  {np.median(car_rewards):7.2f}")
    print(f"  Min:     {np.min(car_rewards):7.2f}")
    print(f"  Max:     {np.max(car_rewards):7.2f}")
    success_rate = np.sum(car_rewards > 5) / n * 100
    print(f"  Success: {np.sum(car_rewards > 5)}/{n} ({success_rate:.1f}%)")
    
    print("\n" + "="*70)
    
    return df

# ============================================================================
# EXECUTION SECTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("GRIDWORLD REINFORCEMENT LEARNING EXPERIMENTS")
    print("Comparing: Off-Policy Greedy vs Pessimistic Q-Learning vs CAR Operator")
    print("="*70)
    print("\nChoose an option:")
    print("  1. Quick test (5 experiments,)")
    print("  2. Medium test (50 experiments,)")
    print("  3. Full test (500 experiments,)")
    print()
    print("  4. Train attackers (one-time setup)")
    print("  5. Analyze existing results")
    print("  6. Visualize existing results")
    print("  7. Check progress")
    print("="*70)
    
    choice = input("\nEnter your choice (1-7): ").strip()
    
    if choice in ['1', '2', '3']:
        # ========== OPTIONS 1-3: Run Experiments ==========
        if choice == '1':
            num_experiments = 5
            csv_filename = 'gridworld_test_5runs.csv'
            checkpoint_file = 'test_checkpoint.pkl'
            est_time = "~5 minutes"
        elif choice == '2':
            num_experiments = 50
            csv_filename = 'gridworld_medium_50runs.csv'
            checkpoint_file = 'medium_checkpoint.pkl'
            est_time = "~20 minutes"
        else:  # choice == '3'
            num_experiments = 500
            csv_filename = 'gridworld_experiments_500runs.csv'
            checkpoint_file = 'experiment_checkpoint.pkl'
            est_time = "~1 hour"
        
        print(f"\n{'='*70}")
        print(f"RUNNING {num_experiments} EXPERIMENTS")
        print(f"{'='*70}")
        print(f"Estimated time: {est_time}")
        print(f"Results will be saved to: {csv_filename}")
        print(f"Press Ctrl+C at any time to pause")
        print(f"{'='*70}\n")
        
        import pandas as pd
        import os
        
        # Check if attackers exist
        attack_models = None
        if os.path.exists('attacker_offpolicy.zip') and \
           os.path.exists('attacker_pessimistic.zip') and \
           os.path.exists('attacker_car.zip'):
            use_attackers = input("Found trained attackers. Use for adversarial testing? (yes/no): ").lower()
            if use_attackers == 'yes':
                attack_models = {
                    'offpolicy': DQN.load('attacker_offpolicy'),
                    'pessimistic': DQN.load('attacker_pessimistic'),
                    'car': DQN.load('attacker_car')
                }
                print("✓ Loaded attackers for all three algorithms")
            else:
                print("Running without attackers")
        else:
            print("No attackers found. Running without adversarial testing.")
            print("(Use Option 4 to train attackers first)")
        
        # Check for existing progress
        start_exp = 0
        if os.path.exists(checkpoint_file):
            resume = input(f"\nFound checkpoint. Resume from last completed experiment? (yes/no): ").lower()
            if resume == 'yes':
                start_exp = load_checkpoint(checkpoint_file)
        
        # Run experiments
        all_results = []
        for exp_num in range(start_exp, num_experiments):
            print(f"\n{'='*70}")
            print(f"EXPERIMENT {exp_num + 1}/{num_experiments}")
            print(f"Using seed: {exp_num}")
            print(f"{'='*70}")
            
            try:
                # Run single experiment with fixed seed
                results, agent1, agent2, agent3 = run_single_experiment_with_seed(
                    experiment_num=exp_num,
                    seed=exp_num,
                    attack_models=attack_models
                )
                
                all_results.append(results)
                
                # Log to CSV
                log_experiment_to_csv(exp_num, results, filename=csv_filename)
                
                # Save checkpoint
                save_checkpoint(exp_num, checkpoint_file=checkpoint_file)
                
                print(f"\n✓ Experiment {exp_num + 1} complete and saved")
                
            except KeyboardInterrupt:
                print(f"\n\n{'='*70}")
                print("PAUSED - Progress saved")
                print(f"{'='*70}")
                print(f"Completed: {exp_num}/{num_experiments} experiments")
                print(f"Resume anytime by running this script again")
                print(f"{'='*70}")
                break
            except Exception as e:
                print(f"\n Error in experiment {exp_num}: {e}")
                print(f"Progress saved. You can resume from experiment {exp_num + 1}")
                raise e
        
        # Clear checkpoint if all complete
        if exp_num == num_experiments - 1:
            clear_checkpoint(checkpoint_file)
            
            print(f"\n{'='*70}")
            print(f"ALL {num_experiments} EXPERIMENTS COMPLETE!")
            print(f"{'='*70}")
            print(f"\nResults saved to: {csv_filename}")
            
            # Auto-prompt for analysis
            if input("\nGenerate analysis? (yes/no): ").lower() == 'yes':
                analyze_csv_results(csv_filename)
                
                if input("\nGenerate visualizations? (yes/no): ").lower() == 'yes':
                    visualize_experiment_results(csv_filename)
    
    elif choice == '4':
        # ========== OPTION 4: Train Attackers ==========
        print("\n" + "="*70)
        print("TRAIN ATTACKERS")
        print("="*70)
        print("\nThis will train 3 separate attackers:")
        print("  1. Attacker vs Pessimistic Q-Learning")
        print("  2. Attacker vs CAR Operator") 
        print("  3. Attacker vs Off-Policy Greedy")
        print("\nEach attacker will be trained for 200,000 timesteps.")
        
        
        confirm = input("Continue? (yes/no): ").lower()
        
        if confirm == 'yes':
            print("\n" + "="*70)
            print("STEP 1: Training baseline agents to attack")
            print("="*70)
            
            # Set seed for reproducible attacker training
            ATTACKER_TRAINING_SEED = 99999  # Use a different seed from testing (0-49)
            
            # Train Off-Policy agent  # ← Should align with line above
            print("\n[1/3] Training Off-Policy agent...")
            env1 = GridWorld()
            env1.seed(ATTACKER_TRAINING_SEED)
            np.random.seed(ATTACKER_TRAINING_SEED)
            agent1 = Q_Agent(env1)
            play(env1, agent1, trials=2000, learn=True)
            play_offpolicy_greedy_target(env1, agent1, trials=500, learn=True)
            print("✓ Off-Policy agent trained")
            
            # Train Pessimistic agent  # ← Same indentation
            print("\n[2/3] Training Pessimistic agent...")
            env2 = GridWorld()
            env2.seed(ATTACKER_TRAINING_SEED + 1)
            np.random.seed(ATTACKER_TRAINING_SEED + 1)
            agent2 = Q_Agent(env2)
            play(env2, agent2, trials=2000, learn=True)
            play_att_with_belief(env2, agent2, trials=500, learn=True)
            print("✓ Pessimistic agent trained")
            
            # Train CAR agent  # ← Same indentation
            print("\n[3/3] Training CAR agent...")
            env3 = GridWorld()
            env3.seed(ATTACKER_TRAINING_SEED + 2)
            np.random.seed(ATTACKER_TRAINING_SEED + 2)
            agent3 = Q_Agent(env3)
            play(env3, agent3, trials=2000, learn=True)
            play_with_car_operator(env3, agent3, trials=500, learn=True)
            print("✓ CAR agent trained")
            
            print("\n" + "="*70)  # ← Should align with lines above
            
            print("\n" + "="*70)
            print("STEP 2: Training attackers against each agent")
            print("="*70)
            
            # Train attackers
            print("\n[1/3] Training attacker against Off-Policy...")
            attack_offpolicy = train_RL_attack_with_belief(agent1, 'offpolicy', 'attacker')
            
            print("\n[2/3] Training attacker against Pessimistic...")
            attack_pessimistic = train_RL_attack_with_belief(agent2, 'pessimistic', 'attacker')
            
            print("\n[3/3] Training attacker against CAR...")
            attack_car = train_RL_attack_with_belief(agent3, 'car', 'attacker')
            
            print("\n" + "="*70)
            print("ALL ATTACKERS TRAINED SUCCESSFULLY!")
            print("="*70)
            print("\nSaved models:")
            print("  ✓ attacker_offpolicy.zip")
            print("  ✓ attacker_pessimistic.zip")
            print("  ✓ attacker_car.zip")
            print("\nThese can be used in Options 1-3 for adversarial testing")
            print("="*70)
        else:
            print("Cancelled.")
    
    elif choice == '5':
        # ========== OPTION 5: Analyze Results ==========
        print("\n" + "="*70)
        print("ANALYZE CSV RESULTS")
        print("="*70)
        print("\nWhich results do you want to analyze?")
        print("  1. Quick test (5 experiments)")
        print("  2. Medium test (50 experiments)")
        print("  3. Full experiment (500 experiments)")
        print("  4. Custom CSV file")
        
        sub_choice = input("\nEnter choice (1-4): ").strip()
        
        if sub_choice == '1':
            df = analyze_csv_results('gridworld_test_5runs.csv')
        elif sub_choice == '2':
            df = analyze_csv_results('gridworld_medium_50runs.csv')
        elif sub_choice == '3':
            df = analyze_csv_results('gridworld_experiments_500runs.csv')
        elif sub_choice == '4':
            filename = input("Enter CSV filename: ").strip()
            import os
            if os.path.exists(filename):
                df = analyze_csv_results(filename)
            else:
                print(f"Error: File '{filename}' not found.")
        else:
            print("Invalid choice.")
    
    elif choice == '6':
        # ========== OPTION 6: Visualize Results ==========
        print("\n" + "="*70)
        print("VISUALIZE EXPERIMENT RESULTS")
        print("="*70)
        print("\nWhich results do you want to visualize?")
        print("  1. Quick test (5 experiments)")
        print("  2. Medium test (50 experiments)")
        print("  3. Full experiment (500 experiments)")
        print("  4. Custom CSV file")
        
        sub_choice = input("\nEnter choice (1-4): ").strip()
        
        if sub_choice == '1':
            visualize_experiment_results('gridworld_test_5runs.csv')
        elif sub_choice == '2':
            visualize_experiment_results('gridworld_medium_50runs.csv')
        elif sub_choice == '3':
            visualize_experiment_results('gridworld_experiments_500runs.csv')
        elif sub_choice == '4':
            filename = input("Enter CSV filename: ").strip()
            import os
            if os.path.exists(filename):
                visualize_experiment_results(filename)
            else:
                print(f"Error: File '{filename}' not found.")
        else:
            print("Invalid choice.")
    
    elif choice == '7':
        # ========== OPTION 7: Check Progress ==========
        print("\n" + "="*70)
        print("CHECK EXPERIMENT PROGRESS")
        print("="*70)
        print("\nWhich experiment set do you want to check?")
        print("  1. Quick test (5 experiments)")
        print("  2. Medium test (50 experiments)")
        print("  3. Full experiment (500 experiments)")
        
        sub_choice = input("\nEnter choice (1-3): ").strip()
        
        if sub_choice == '1':
            check_experiment_progress(
                csv_filename='gridworld_test_5runs.csv',
                checkpoint_file='test_checkpoint.pkl',
                total_experiments=5
            )
        elif sub_choice == '2':
            check_experiment_progress(
                csv_filename='gridworld_medium_50runs.csv',
                checkpoint_file='medium_checkpoint.pkl',
                total_experiments=50
            )
        elif sub_choice == '3':
            check_experiment_progress(
                csv_filename='gridworld_experiments_500runs.csv',
                checkpoint_file='experiment_checkpoint.pkl',
                total_experiments=500
            )
        else:
            print("Invalid choice.")
    
    else:
        print("Invalid choice. Please run again and select 1-7.")
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)