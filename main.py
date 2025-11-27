# -*- coding: utf-8 -*-
"""gridworld_withCAR.py

Modified gridworld with CAR (Consistent Adversarial Robust) operator implementation.
Includes pause/resume functionality for long-running experiments.

Based on original gridworld.ipynb with additions for:
1. CAR operator from "Towards Optimal Adversarial Robust Q-learning with Bellman Infinity-error"
2. Comparison between Pessimistic Q-Learning and CAR operator
3. CSV logging for 500 experiments
4. Pause/Resume capability
5. Progress tracking and analysis
"""

# Import all the libraries
import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import copy
from datetime import datetime
import signal
import sys

# Adi - Changes to next two lines
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
        # Adi - Changes here to return 5th parameter
        # Separated done into terminated (natural episode end) and truncated (time limit reached)
        # Changed return from 4 values to 5 values: (observation, reward, terminated, truncated, info)
        # This matches the Gymnasium v0.26+ API that stable-baselines3 expects!
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
        """Check if the agent is in a terminal state (gold or bomb), if so return 'TERMINAL'"""
        if self.current_location in self.terminal_states:
            return 'TERMINAL'

    def reset(self, seed=None):
        self.total_step = 0
        self.current_location =  [np.random.randint(0,5), np.random.randint(0,5)]
        # Adi - Defaulting to a random value
        if seed is not None:
          np.random.seed(seed)

        while self.current_location in self.terminal_states:
            self.current_location =  [np.random.randint(0,5), np.random.randint(0,5)]
        return np.asarray(self.current_location)

# Define the attack class
class GridWorld_RL_Attack(gym.Env):
    ## Initialise starting data
    def __init__(self, agentQ):
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


        agent_action = self.agentQ.choose_action_att(['UP', 'DOWN', 'LEFT', 'RIGHT'] , tuple(self.purturb_state), 0)
        self.current_location = self.take_action(self.current_location, agent_action)
        possible_true_states = self.agentQ.get_ball(self.purturb_state)
        belief_before_action = states_intersection(self.belief, possible_true_states)
        self.belief = set()
        for state in belief_before_action:
            self.belief.add(self.take_action(state, agent_action))
        self.agentQ.defender_policy = self.agentQ.get_policy_with_belief(self.agentQ.defender_policy, self.belief)
        
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
        # Adi - setting a random value to seed
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

    def find_maxmin(self, q_table, next_state, policy, new_policy):
        states = self.get_ball(next_state)
        values = dict()
        possible_actions = []
        for i in states:
            possible_action = policy[i]
            if possible_action not in possible_actions:
                possible_actions.append(possible_action)
        for action in possible_actions:
            values[action] = q_table[next_state][action]
        return min(values.values())

    def find_maxmin_with_belief(self, q_table_true, q_table_withatt, next_state, policy, new_policy, belief):
        states = self.get_ball(next_state)
        values = dict()
        possible_actions = []
        possible_states = states_intersection(belief, states)
        if (len(possible_states) > 1):
            for i in possible_states:
                possible_action = policy[i]
                if possible_action not in possible_actions:
                    possible_actions.append(possible_action)
            for action in possible_actions:
                values[action] = q_table_withatt[next_state][action]
            return np.min(list(values.values()))
        else:
            return max(q_table_withatt[list(possible_states)[0]].values())

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
                values = []
                for i in states:
                    values.append(self.q_table_withatt[state][policy[i]])
                value_min[state] = min(values)

            for state in self.all_state:
                states = self.get_ball(state)
                for true_state in states:
                    for action in self.environment.actions:
                        condition_v = value_min[true_state]<self.q_table_withatt[true_state][action] and self.q_table_withatt[true_state][action]<self.q_table_withatt[true_state][policy[state]]
                        condition_q = True
                        condition_atleast = False
                        for i in states:
                            if i == true_state:
                                continue
                            else:
                                if self.q_table_withatt[true_state][policy[state]] > self.q_table_withatt[true_state][action]:
                                    condition_q = False
                                    break
                                if self.q_table_withatt[true_state][policy[state]] < self.q_table_withatt[true_state][action]:
                                    condition_atleast = True
                        if condition_v and condition_q and condition_atleast:
                            if state not in old.keys():
                                old[state] = policy[state]
                                policy[state] = action
                                no_change = False
                                break
                            elif old[state] != action:
                                old[state] = policy[state]
                                policy[state] = action
                                no_change = False
                                break
                            else:
                                continue
        return policy



    def get_policy(self, init = False):
        if init:
            policy = dict()
            for state in self.all_state:
                policy[state] = self.environment.actions[np.random.randint(0, len(self.environment.actions))]
        else:
            policy = dict()
            for state in self.all_state:
                values = dict()
                states = self.get_ball(state)
                for action in self.environment.actions:
                    tmp = []
                    for true_state in states:
                        tmp.append(self.q_table_withatt[true_state][action])
                    values[action] = min(tmp)
                policy[state] = max(values, key = values.get)
        return policy

    def get_policy_naive(self):
        policy = dict()
        for state in self.all_state:
            values = dict()
            for action in self.environment.actions:
                values[action] = self.q_table[state][action]
            policy[state] = max(values, key = values.get)
        return policy

    def get_policy_with_belief(self, old_policy, belief, init = False):
        if init:
            policy = dict()
            for state in self.all_state:
                policy[state] = self.environment.actions[np.random.randint(0, len(self.environment.actions))]
        else:
            policy = dict()
            if len(belief) == 1:
                values = dict()
                for action in self.environment.actions:
                    values[action] = self.q_table_withatt[list(belief)[0]][action]
                true_action = max(values, key = values.get)
                policy = copy.deepcopy(old_policy)
                for state in self.all_state:
                    policy[state] = true_action
                return policy
            for state in self.all_state:
                values = dict()
                states = self.get_ball(state)
                possible_states = states_intersection(belief, states)
                if len(possible_states)!=0:
                    for action in self.environment.actions:
                        tmp = []
                        for true_state in possible_states:
                            tmp.append(self.q_table_withatt[true_state][action])
                        values[action] = np.mean(tmp)
                    policy[state] = max(values, key = values.get)
                else:
                    policy[state] = old_policy[state]
        return policy



    def get_ball(self, state):
        result = []
        result.append([state[0]-1, state[1]])
        result.append([state[0], state[1]])
        result.append([state[0]+1, state[1]])
        result.append([state[0], state[1]-1])
        result.append([state[0], state[1]+1])
        true_result = []
        for i in result:
            if not ((i[0]<0 or i[0]>self.environment.height-1) or (i[1]<0 or i[1]>self.environment.width-1)):
                true_result.append(tuple(i))
        return true_result

# Define the play function - original
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
            eps = linear_schedule_exp(0.001, 0.5, trials, trial)
            action = agent.choose_action(environment.actions, eps)
            _, reward, terminated, truncated, _ = environment.step(action)
            new_state = environment.current_location

            if learn == True: # Update Q-values if learning is specified
                alpha = linear_schedule_exp(0.01, 0.5, trials, trial)
                agent.learn(old_state, reward, new_state, action, alpha)

            cumulative_reward += reward
            step += 1

            if environment.check_state() == 'TERMINAL': # If game is in terminal state, game over and start next trial
                environment.__init__()
                game_over = True
        reward_per_episode.append(cumulative_reward) # Append reward for current trial to performance log

    return reward_per_episode # Return performance log

# define the routine - play_att
def play_att(environment, agent, trials=500, max_steps_per_episode=1000, learn=False):
    """The play function runs iterations and updates Q-values if desired."""
    reward_per_episode = [] # Initialise performance log
    if learn:
        agent.q_table_withatt = copy.deepcopy(agent.q_table)
        agent.defender_policy = agent.pareto_update(agent.get_policy(init = True))
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
def play_att_with_belief(environment, agent, trials=500, max_steps_per_episode=1000, learn=False, attack_model = None):
    """The play function runs iterations and updates Q-values if desired."""
    reward_per_episode = [] # Initialise performance log
    if learn:
        agent.q_table_withatt = copy.deepcopy(agent.q_table)
        agent.defender_policy = agent.pareto_update(agent.get_policy(init = True))
    
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
            if learn:
                eps = linear_schedule_exp(0.01, 0.2, trials, trial)
            else:
                eps = 0
            action = agent.choose_action_att(environment.actions , att_state, eps)
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
    return reward_per_episode # Return performance log

# NEW FUNCTION: Play with CAR Operator
def play_with_car_operator(environment, agent, trials=500, max_steps_per_episode=1000, learn=False):
    """
    Play function implementing the CAR operator approach.
    
    """
    reward_per_episode = []
    
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
            
            # Action selection: greedy with respect to Q-function (CAR way)
            # This is DIFFERENT from your belief-based policy
            action = agent.choose_action_car(environment.actions, old_state, eps)
            
            # Take step in environment
            _, reward, terminated, truncated, _ = environment.step(action)
            new_state = environment.current_location
            
            # Q-learning update using CAR operator
            if learn == True:
                alpha = linear_schedule_exp(0.01, 0.1, trials, trial)
                agent.learn_with_car_operator(old_state, reward, new_state, action, alpha)
            
            cumulative_reward += reward
            step += 1
            
            if environment.check_state() == 'TERMINAL':
                environment.__init__()
                game_over = True
        
        reward_per_episode.append(cumulative_reward)
    
    return reward_per_episode

# Define the RL for training with attack vector
def train_RL_attack_with_belief(agentQ):
    env = GridWorld_RL_Attack(agentQ)
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path="RL_attack",
                                             name_prefix='rl_model')

    model = DQN("MlpPolicy", env, learning_rate = 1e-3, learning_starts = 1000, batch_size = 256, exploration_final_eps=0.001,
                gamma = 1, tau = 0.2, train_freq = (5, "step"), policy_kwargs={"net_arch" : [256,128]}, tensorboard_log="RL_attack/", verbose=1)

    model.learn(total_timesteps=20000, callback = checkpoint_callback)
    return model

# NEW FUNCTION: CSV Logging
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
    fieldnames = [
        'experiment_number',
        'timestamp',
        'baseline1_avg_reward',
        'baseline2_avg_reward',
        'pessimistic_train_avg_reward',
        'pessimistic_test_avg_reward',
        'car_train_avg_reward',
        'car_test_avg_reward',
        'pessimistic_vs_attacker_reward',
        'car_vs_attacker_reward',
        'winner',
        'performance_difference',
        'pessimistic_train_std',
        'pessimistic_test_std',
        'car_train_std',
        'car_test_std',
        'pessimistic_vs_attacker_std',
        'car_vs_attacker_std'
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
        
        # Quick stats
        if completed > 0:
            print(f"\n  Current Win Distribution:")
            print(f"    Pessimistic: {(df['winner'] == 'PESSIMISTIC').sum()} ({(df['winner'] == 'PESSIMISTIC').sum()/completed*100:.1f}%)")
            print(f"    CAR:         {(df['winner'] == 'CAR').sum()} ({(df['winner'] == 'CAR').sum()/completed*100:.1f}%)")
            print(f"    Ties:        {(df['winner'] == 'TIE').sum()} ({(df['winner'] == 'TIE').sum()/completed*100:.1f}%)")
            
            print(f"\n  Average Rewards:")
            print(f"    Pessimistic: {df['pessimistic_test_avg_reward'].mean():.2f}")
            print(f"    CAR:         {df['car_test_avg_reward'].mean():.2f}")
    else:
        print(f"\n⚠️  No CSV file found: {csv_filename}")
        completed = 0
    
    # Check checkpoint
    print(f"\n" + "-"*70)
    if os.path.exists(checkpoint_file):
        last_completed = load_checkpoint(checkpoint_file)
        print(f"  Next experiment will be: {last_completed + 1}")
    else:
        print(f"⚠️  No checkpoint file found: {checkpoint_file}")
        print(f"  Starting fresh if you run main()")
    
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

# NEW FUNCTION: Analyze CSV Results
def analyze_csv_results(csv_filename='gridworld_experiments.csv'):
    """
    Analyze and visualize results from the CSV file.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Read CSV
    df = pd.read_csv(csv_filename)
    
    print("="*70)
    print(f"ANALYSIS OF {len(df)} EXPERIMENTS")
    print("="*70)
    
    # Summary statistics
    print("\n--- Pessimistic Q-Learning ---")
    print(f"Mean test reward: {df['pessimistic_test_avg_reward'].mean():.2f} ± {df['pessimistic_test_avg_reward'].std():.2f}")
    print(f"Mean vs attacker: {df['pessimistic_vs_attacker_reward'].mean():.2f} ± {df['pessimistic_vs_attacker_reward'].std():.2f}")
    
    print("\n--- CAR Operator ---")
    print(f"Mean test reward: {df['car_test_avg_reward'].mean():.2f} ± {df['car_test_avg_reward'].std():.2f}")
    print(f"Mean vs attacker: {df['car_vs_attacker_reward'].mean():.2f} ± {df['car_vs_attacker_reward'].std():.2f}")
    
    print("\n--- Win Distribution ---")
    print(df['winner'].value_counts())
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Test rewards comparison
    axes[0, 0].hist(df['pessimistic_test_avg_reward'], alpha=0.5, label='Pessimistic', bins=30)
    axes[0, 0].hist(df['car_test_avg_reward'], alpha=0.5, label='CAR', bins=30)
    axes[0, 0].set_xlabel('Test Reward')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Test Reward Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Performance difference over experiments
    axes[0, 1].plot(df['experiment_number'], df['performance_difference'])
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Experiment Number')
    axes[0, 1].set_ylabel('Performance Difference (CAR - Pessimistic)')
    axes[0, 1].set_title('Performance Difference Over Experiments')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Rewards vs attacker
    axes[1, 0].hist(df['pessimistic_vs_attacker_reward'], alpha=0.5, label='Pessimistic', bins=30)
    axes[1, 0].hist(df['car_vs_attacker_reward'], alpha=0.5, label='CAR', bins=30)
    axes[1, 0].set_xlabel('Reward vs Attacker')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Performance Against Trained Attacker')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Scatter plot
    axes[1, 1].scatter(df['pessimistic_test_avg_reward'], df['car_test_avg_reward'], alpha=0.5)
    axes[1, 1].plot([df['pessimistic_test_avg_reward'].min(), df['pessimistic_test_avg_reward'].max()],
                     [df['pessimistic_test_avg_reward'].min(), df['pessimistic_test_avg_reward'].max()],
                     'r--', alpha=0.5, label='Equal performance')
    axes[1, 1].set_xlabel('Pessimistic Test Reward')
    axes[1, 1].set_ylabel('CAR Test Reward')
    axes[1, 1].set_title('Direct Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    #plt.tight_layout()
    #plt.savefig('experiment_analysis.png', dpi=300, bbox_inches='tight')
    #print(f"\n✓ Analysis plot saved to: experiment_analysis.png")
    #plt.show()
    
    return df

# MAIN FUNCTION with Pause/Resume
def main(num_experiments=500, save_csv=True, csv_filename='gridworld_experiments.csv', 
         checkpoint_file='experiment_checkpoint.pkl', resume=True):
    """
    Run multiple experiments comparing CAR operator vs Pessimistic Q-Learning.
    Supports pausing and resuming experiments.
    
    Parameters:
    -----------
    num_experiments : int
        Total number of experiments to run (default: 500)
    save_csv : bool
        Whether to save results to CSV (default: True)
    csv_filename : str
        Name of CSV file to save results
    checkpoint_file : str
        Name of checkpoint file for pause/resume functionality
    resume : bool
        If True, resume from last checkpoint if it exists (default: True)
    """
    import os
    
    # Track if user requested pause
    pause_requested = False
    
    def signal_handler(sig, frame):
        """Handle Ctrl+C gracefully"""
        nonlocal pause_requested
        print("\n" + "="*70)
        print("PAUSE REQUESTED - Will stop after current experiment completes")
        print("="*70)
        pause_requested = True
    
    # Register the signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    print("="*70)
    print(f"RUNNING UP TO {num_experiments} EXPERIMENTS")
    print("Comparing CAR Operator vs Pessimistic Q-Learning with Belief")
    print("="*70)
    print(f"\nResults will be saved to: {csv_filename}")
    print(f"Checkpoint file: {checkpoint_file}")
    print(f"\n⚠️  Press Ctrl+C at any time to pause (will finish current experiment)")
    print("="*70)
    
    # Determine starting experiment number
    if resume:
        start_experiment = load_checkpoint(checkpoint_file) + 1
        if start_experiment > 1:
            response = input(f"\nResume from experiment {start_experiment}? (y/n): ").lower()
            if response != 'y':
                print("Starting from experiment 1 (checkpoint will be overwritten)")
                start_experiment = 1
    else:
        start_experiment = 1
        print("Resume disabled - starting from experiment 1")
    
    # Store summary statistics across all experiments
    all_pessimistic_rewards = []
    all_car_rewards = []
    pessimistic_wins = 0
    car_wins = 0
    ties = 0
    
    # Main experiment loop
    for exp_num in range(start_experiment, num_experiments + 1):
        
        # Check if pause was requested
        if pause_requested:
            print("\n" + "="*70)
            print("PAUSING EXPERIMENTS")
            print("="*70)
            print(f"Completed up to experiment {exp_num - 1}")
            print(f"Checkpoint saved to: {checkpoint_file}")
            print(f"\nTo resume later, run:")
            print(f"  main(num_experiments={num_experiments}, resume=True)")
            print("="*70)
            return {
                'status': 'PAUSED',
                'last_completed': exp_num - 1,
                'total_experiments': num_experiments,
                'checkpoint_file': checkpoint_file,
                'csv_filename': csv_filename
            }
        
        print(f"\n{'='*70}")
        print(f"EXPERIMENT {exp_num}/{num_experiments}")
        print(f"Progress: {(exp_num-1)/num_experiments*100:.1f}% complete")
        print(f"{'='*70}")
        
        # Dictionary to store results for this experiment
        results = {}
        
        try:
            # ========== APPROACH 1: PESSIMISTIC Q-LEARNING WITH BELIEF ==========
            print(f"\n[Exp {exp_num}] Training YOUR Pessimistic Q-Learning...")
            
            environment1 = GridWorld()
            agentQ1 = Q_Agent(environment1)
            
            # Train baseline
            reward_baseline1 = play(environment1, agentQ1, trials=2000, learn=True)
            results['baseline1_avg_reward'] = np.mean(reward_baseline1[-100:])
            
            # Train with pessimistic approach
            reward_train1 = play_att_with_belief(environment1, agentQ1, trials=500, learn=True)
            results['pessimistic_train_avg_reward'] = np.mean(reward_train1)
            results['pessimistic_train_std'] = np.std(reward_train1)
            
            # Test without attacks
            reward_test1 = play_att_with_belief(environment1, agentQ1, trials=100, learn=False, attack_model=None)
            results['pessimistic_test_avg_reward'] = np.mean(reward_test1)
            results['pessimistic_test_std'] = np.std(reward_test1)
            
            print(f"  ✓ Pessimistic - Test reward: {results['pessimistic_test_avg_reward']:.2f}")
            
            # ========== APPROACH 2: CAR OPERATOR ==========
            print(f"\n[Exp {exp_num}] Training CAR Operator...")
            
            environment2 = GridWorld()
            agentQ2 = Q_Agent(environment2)
            
            # Train baseline
            reward_baseline2 = play(environment2, agentQ2, trials=2000, learn=True)
            results['baseline2_avg_reward'] = np.mean(reward_baseline2[-100:])
            
            # Train with CAR operator
            reward_train2 = play_with_car_operator(environment2, agentQ2, trials=500, learn=True)
            results['car_train_avg_reward'] = np.mean(reward_train2)
            results['car_train_std'] = np.std(reward_train2)
            
            # Test without attacks
            reward_test2 = play_with_car_operator(environment2, agentQ2, trials=100, learn=False)
            results['car_test_avg_reward'] = np.mean(reward_test2)
            results['car_test_std'] = np.std(reward_test2)
            
            print(f"  ✓ CAR operator - Test reward: {results['car_test_avg_reward']:.2f}")
            
            # ========== TEST WITH TRAINED ATTACKER ==========
            print(f"\n[Exp {exp_num}] Testing against trained attacker...")
            
            # Train attacker on pessimistic agent
            attack_model = train_RL_attack_with_belief(agentQ1)
            
            # Test pessimistic vs attacker
            reward_pessimistic_vs_attacker = play_att_with_belief(
                environment1, agentQ1, trials=100, learn=False, attack_model=attack_model
            )
            results['pessimistic_vs_attacker_reward'] = np.mean(reward_pessimistic_vs_attacker)
            results['pessimistic_vs_attacker_std'] = np.std(reward_pessimistic_vs_attacker)
            
            # Test CAR vs attacker
            reward_car_vs_attacker = play_with_car_operator(
                environment2, agentQ2, trials=100, learn=False
            )
            results['car_vs_attacker_reward'] = np.mean(reward_car_vs_attacker)
            results['car_vs_attacker_std'] = np.std(reward_car_vs_attacker)
            
            # ========== DETERMINE WINNER ==========
            pessimistic_score = results['pessimistic_test_avg_reward']
            car_score = results['car_test_avg_reward']
            performance_diff = car_score - pessimistic_score
            
            results['performance_difference'] = performance_diff
            
            if abs(performance_diff) < 0.01:  # Essentially tied
                results['winner'] = 'TIE'
                ties += 1
                print(f"\n  → Result: TIE (diff: {performance_diff:.3f})")
            elif performance_diff > 0:
                results['winner'] = 'CAR'
                car_wins += 1
                print(f"\n  → Result: CAR WINS by {performance_diff:.2f}")
            else:
                results['winner'] = 'PESSIMISTIC'
                pessimistic_wins += 1
                print(f"\n  → Result: PESSIMISTIC WINS by {abs(performance_diff):.2f}")
            
            # Store for overall statistics
            all_pessimistic_rewards.append(pessimistic_score)
            all_car_rewards.append(car_score)
            
            # ========== LOG TO CSV ==========
            if save_csv:
                log_experiment_to_csv(exp_num, results, filename=csv_filename)
            
            # ========== SAVE CHECKPOINT ==========
            save_checkpoint(exp_num, checkpoint_file=checkpoint_file)
            
        except Exception as e:
            print(f"\n⚠️  ERROR in experiment {exp_num}: {e}")
            print(f"Checkpoint saved at experiment {exp_num - 1}")
            print(f"You can resume from this point later.")
            raise e
        
        # ========== PROGRESS UPDATE ==========
        if exp_num % 10 == 0 or exp_num == num_experiments:
            elapsed_exp = exp_num - start_experiment + 1
            print(f"\n{'='*70}")
            print(f"PROGRESS UPDATE - Completed {exp_num}/{num_experiments} experiments")
            print(f"{'='*70}")
            print(f"Pessimistic wins: {pessimistic_wins} ({pessimistic_wins/elapsed_exp*100:.1f}%)")
            print(f"CAR wins:         {car_wins} ({car_wins/elapsed_exp*100:.1f}%)")
            print(f"Ties:             {ties} ({ties/elapsed_exp*100:.1f}%)")
            print(f"{'='*70}")
    
    # ========== ALL EXPERIMENTS COMPLETE ==========
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*70)
    
    # Clear checkpoint since we're done
    clear_checkpoint(checkpoint_file)
    
    # ========== FINAL SUMMARY ==========
    print(f"\nTotal experiments: {num_experiments}")
    print(f"\nWin Distribution:")
    print(f"  Pessimistic Q-Learning wins: {pessimistic_wins} ({pessimistic_wins/num_experiments*100:.1f}%)")
    print(f"  CAR Operator wins:           {car_wins} ({car_wins/num_experiments*100:.1f}%)")
    print(f"  Ties:                        {ties} ({ties/num_experiments*100:.1f}%)")
    
    print(f"\nOverall Performance:")
    print(f"  Pessimistic Q-Learning:")
    print(f"    Mean reward: {np.mean(all_pessimistic_rewards):.2f}")
    print(f"    Std dev:     {np.std(all_pessimistic_rewards):.2f}")
    print(f"    Min reward:  {np.min(all_pessimistic_rewards):.2f}")
    print(f"    Max reward:  {np.max(all_pessimistic_rewards):.2f}")
    
    print(f"\n  CAR Operator:")
    print(f"    Mean reward: {np.mean(all_car_rewards):.2f}")
    print(f"    Std dev:     {np.std(all_car_rewards):.2f}")
    print(f"    Min reward:  {np.min(all_car_rewards):.2f}")
    print(f"    Max reward:  {np.max(all_car_rewards):.2f}")
    
    avg_difference = np.mean(all_car_rewards) - np.mean(all_pessimistic_rewards)
    print(f"\nAverage performance difference: {avg_difference:.3f}")
    
    if abs(avg_difference) < 0.01:
        print("→ Overall: Methods perform EQUALLY")
    elif avg_difference > 0:
        print(f"→ Overall: CAR Operator performs BETTER (by {avg_difference:.2f})")
    else:
        print(f"→ Overall: Pessimistic Q-Learning performs BETTER (by {abs(avg_difference):.2f})")
    
    if save_csv:
        print(f"\n✓ All results saved to: {csv_filename}")
    
    print("="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    
    return {
        'status': 'COMPLETE',
        'pessimistic_rewards': all_pessimistic_rewards,
        'car_rewards': all_car_rewards,
        'pessimistic_wins': pessimistic_wins,
        'car_wins': car_wins,
        'ties': ties,
        'csv_filename': csv_filename
    }


# ============================================================================
# EXECUTION SECTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("GRIDWORLD WITH CAR OPERATOR")
    print("Comparing Pessimistic Q-Learning vs CAR Operator")
    print("="*70)
    print("\nChoose an option:")
    print("  1. Quick test (5 experiments)")
    print("  2. Medium test (50 experiments)")
    print("  3. Full experiment (500 experiments)")
    print("  4. Check progress of existing experiments")
    print("="*70)
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        # ========== OPTION 1: Quick Test (5 experiments) ==========
        print("\n" + "="*70)
        print("QUICK TEST - 5 EXPERIMENTS")
        print("="*70)
        print("This will take approximately 1 hour")
        print("Press Ctrl+C at any time to pause\n")
        
        results = main(
            num_experiments=5,
            save_csv=True,
            csv_filename='gridworld_test_5runs.csv',
            checkpoint_file='test_checkpoint.pkl',
            resume=True
        )
        
        if results['status'] == 'COMPLETE':
            print("\n" + "="*70)
            print("Test complete! Generating analysis...")
            print("="*70)
            df = analyze_csv_results('gridworld_test_5runs.csv')
    
    elif choice == '2':
        # ========== OPTION 2: Medium Test (50 experiments) ==========
        print("\n" + "="*70)
        print("MEDIUM TEST - 50 EXPERIMENTS")
        print("="*70)
        print("This will take approximately 10 hours")
        print("Press Ctrl+C at any time to pause\n")
        
        results = main(
            num_experiments=50,
            save_csv=True,
            csv_filename='gridworld_medium_50runs.csv',
            checkpoint_file='medium_checkpoint.pkl',
            resume=True
        )
        
        if results['status'] == 'COMPLETE':
            print("\n" + "="*70)
            print("Experiments complete! Generating analysis...")
            print("="*70)
            df = analyze_csv_results('gridworld_medium_50runs.csv')
    
    elif choice == '3':
        # ========== OPTION 3: Full Experiment (500 experiments) ==========
        print("\n" + "="*70)
        print("FULL EXPERIMENT - 500 EXPERIMENTS")
        print("="*70)
        print("⚠️  WARNING: This will take approximately 100 hours (4+ days)")
        print("Press Ctrl+C at any time to pause")
        print("Run this script again to resume from where you left off\n")
        
        confirm = input("Are you sure you want to start? (yes/no): ").lower()
        
        if confirm == 'yes':
            results = main(
                num_experiments=500,
                save_csv=True,
                csv_filename='gridworld_experiments_500runs.csv',
                checkpoint_file='experiment_checkpoint.pkl',
                resume=True
            )
            
            if results['status'] == 'COMPLETE':
                print("\n" + "="*70)
                print("All experiments complete! Generating analysis...")
                print("="*70)
                df = analyze_csv_results('gridworld_experiments_500runs.csv')
        else:
            print("Cancelled.")
    
    elif choice == '4':
        # ========== OPTION 4: Check Progress ==========
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
    
    else:
        print("Invalid choice. Please run again and select 1-5.")
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)