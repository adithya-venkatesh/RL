import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import copy
# from models import *
# import torch.optim as optim
# import torch.nn as nn
# from attacks import *
# import gym
# from stable_baselines3 import SAC, PPO, TD3, DQN
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback


def linear_schedule_exp(min_exp, ini_exp, total, current):
    return max(min_exp, ((total-current)*ini_exp)/total)

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
        if seed is not None:
            np.random.seed(seed)

    def model_dynamics(self, state, action):
        """
        Deterministic single-step model:
        returns (reward, next_state, done) from an arbitrary 'state' and 'action'
        without touching env.current_location.
        """
        next_state = self.take_action(tuple(state), action)
        reward = self.get_reward(next_state)
        done = list(next_state) in self.terminal_states
        return reward, next_state, done

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
        done = self.current_location in self.terminal_states
        if self.total_step >= 1000:
            done = True

        return np.asarray(self.current_location), reward, done, {}

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
        # if self.current_location == self.bomb_location:
        #     print("BOOM")
        # if self.current_location == self.gold_location:
        #     print("GOLD")
        if self.current_location in self.terminal_states:
            return 'TERMINAL'

    def reset(self):
        self.total_step = 0
        self.current_location =  [np.random.randint(0,5), np.random.randint(0,5)]
        while self.current_location in self.terminal_states:
            self.current_location =  [np.random.randint(0,5), np.random.randint(0,5)]
        return np.asarray(self.current_location)

class GridWorld_RL_Attack(gym.Env):
    def __init__(self, agentQ):
        self.height = 5
        self.width = 5
        self.grid = np.zeros(( self.height, self.width)) - 1
        self.agentQ = agentQ

        self.current_location = [4, np.random.randint(0,5)]

        self.bomb_location = [4,3]
        self.gold_location = [0,3]
        self.terminal_states = [ self.bomb_location, self.gold_location]

        self.grid[ self.bomb_location[0], self.bomb_location[1]] = -10
        self.grid[ self.gold_location[0], self.gold_location[1]] = 10

        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.MultiDiscrete([5,5])
        self.purturb_state = None
        self.belief = None


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
        if seed is not None:
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
        #print(self.total_step, self.current_location, self.purturb_state, agent_action, self.belief)
        self.current_location = self.take_action(self.current_location, agent_action)
        possible_true_states = self.agentQ.get_ball(self.purturb_state)
        belief_before_action = states_intersection(self.belief, possible_true_states)
        self.belief = set()
        for state in belief_before_action:
            self.belief.add(self.take_action(state, agent_action))
        self.agentQ.defender_policy = self.agentQ.get_policy_with_belief(self.agentQ.defender_policy, self.belief)
        done = list(self.current_location) in self.terminal_states
        reward = -self.get_reward(self.current_location)
        self.total_step+=1
        if self.total_step>=1000:
            done = True
        return np.asarray(self.current_location), reward, done, {}

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

    def reset(self):
        self.current_location =  [np.random.randint(0,5), np.random.randint(0,5)]
        self.belief = set()
        for state in self.agentQ.all_state:
            self.belief.add(state)
        #self.current_location =  [4, np.random.randint(0,5)]
        self.agentQ.defender_policy = self.agentQ.get_policy_with_belief(self.agentQ.defender_policy, self.belief)
        while self.current_location in self.terminal_states:
            self.current_location =  [np.random.randint(0,5), np.random.randint(0,5)]
        self.total_step = 0
        return np.asarray(self.current_location)

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

        self._mode = None
        self._td_accum = []
        self.td_logs = {'vanilla': [], 'att': [], 'att_belief': []}

    def begin_episode(self, mode):
        self._mode = mode
        self._td_accum = []

    def record_td(self, delta):
        if self._mode is not None:
            self._td_accum.append(abs(float(delta)))

    def end_episode(self):
        if self._mode is not None:
            mean_abs = float(np.mean(self._td_accum)) if self._td_accum else 0.0
            self.td_logs[self._mode].append(mean_abs)
            self._mode = None
            self._td_accum = []

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

    def learn(self, old_state, reward, new_state, action, alpha = 0.1):
        """Updates the Q-value table using Q-learning (vanilla) + logs TD error"""
        q_values_of_state = self.q_table[tuple(new_state)]
        max_q_value_in_new_state = max(q_values_of_state.values())
        current_q_value = self.q_table[tuple(old_state)][action]
        done = list(new_state) in self.environment.terminal_states
        target = reward if done else reward + self.gamma * max_q_value_in_new_state
        delta = target - current_q_value
        # update
        self.q_table[tuple(old_state)][action] = (1 - alpha) * current_q_value + alpha * (target)
        # log
        self.record_td(delta)
        return delta

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
        """Updates the Q-value table using robust (pessimistic) target + logs robust TD error"""
        max_q_value_in_new_state = self.find_maxmin(self.q_table_withatt, tuple(new_state), policy, new_policy)
        current_q_value = self.q_table_withatt[tuple(old_state)][action]
        done = list(new_state) in self.environment.terminal_states
        target = reward if done else reward + self.gamma * max_q_value_in_new_state
        delta = target - current_q_value
        self.q_table_withatt[tuple(old_state)][action] = (1 - alpha) * current_q_value + alpha * (target)
        self.record_td(delta)
        return delta

    def learn_with_att_with_belief(self, old_state, reward, new_state, action, policy, new_policy, belief, alpha = 0.1):
        """Updates the Q-value table using robust target with belief + logs robust TD error"""
        max_q_value_in_new_state = self.find_maxmin_with_belief(self.q_table, self.q_table_withatt, tuple(new_state), policy, new_policy, belief)
        current_q_value = self.q_table_withatt[tuple(old_state)][action]
        done = list(new_state) in self.environment.terminal_states
        target = reward if done else reward + self.gamma * max_q_value_in_new_state
        delta = target - current_q_value
        self.q_table_withatt[tuple(old_state)][action] = (1 - alpha) * current_q_value + alpha * (target)
        self.record_td(delta)
        return delta

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
                                print("Change policy")
                                break
                            elif old[state] != action:
                                old[state] = policy[state]
                                policy[state] = action
                                no_change = False
                                break
                                print("Change policy")
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

    def mean_bellman_residual_vanilla(self, env):
        vals = []
        for s in self.all_state:
            for a in env.actions:
                r, s_next, done = env.model_dynamics(s, a)
                max_q = max(self.q_table[s_next].values())
                current = self.q_table[s][a]
                target = r if done else r + self.gamma * max_q
                vals.append(abs(target - current))
        return float(np.mean(vals)) if vals else 0.0

    def mean_bellman_residual_att(self, env, policy):
        """Uses current robust table and a provided policy mapping for pessimistic target."""
        if policy is None:
            return float('nan')
        vals = []
        dummy = {}
        for s in self.all_state:
            for a in env.actions:
                r, s_next, done = env.model_dynamics(s, a)
                maxmin = self.find_maxmin(self.q_table_withatt, s_next, policy, dummy)
                current = self.q_table_withatt[s][a]
                target = r if done else r + self.gamma * maxmin
                vals.append(abs(target - current))
        return float(np.mean(vals)) if vals else 0.0


def play(environment, agent, trials=500, max_steps_per_episode=1000, learn=False):
    """The play function runs iterations and updates Q-values if desired."""
    reward_per_episode = [] # Initialise performance log

    for trial in range(trials): # Run trials
        cumulative_reward = 0 # Initialise values of each game
        step = 0
        game_over = False
        old_state = environment.reset()
        if learn:
            agent.begin_episode('vanilla')  
        while step < max_steps_per_episode and game_over != True: # Run until max steps or until game is finished
            old_state = tuple(environment.current_location)
            eps = linear_schedule_exp(0.001, 0.5, trials, trial)
            action = agent.choose_action(environment.actions, eps)
            _,reward,_,_ = environment.step(action)
            new_state = environment.current_location

            if learn == True: # Update Q-values if learning is specified
                alpha = linear_schedule_exp(0.01, 0.5, trials, trial)
                agent.learn(old_state, reward, new_state, action, alpha)  # returns delta (logged internally)

            cumulative_reward += reward
            step += 1

            if environment.check_state() == 'TERMINAL': # If game is in terminal state, game over and start next trial
                environment.__init__()
                game_over = True
        if learn:
            agent.end_episode()  
        reward_per_episode.append(cumulative_reward) # Append reward for current trial to performance log

    return reward_per_episode # Return performance log

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
        if learn:
            agent.begin_episode('att')  
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
            _,reward,_,_ = environment.step(action)
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

        if learn:
            agent.end_episode()  
        reward_per_episode.append(cumulative_reward) # Append reward for current trial to performance log
    return reward_per_episode # Return performance log

def states_intersection(current_belief, states):
    new_belief = set()
    for state in states:
        if state in current_belief:
            new_belief.add(state)
    return new_belief

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
        if learn:
            agent.begin_episode('att_belief')  
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
            _,reward,_,_ = environment.step(action)
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

        if learn:
            agent.end_episode()  
        reward_per_episode.append(cumulative_reward) # Append reward for current trial to performance log
    return reward_per_episode # Return performance log

def play_att_with_belief_pgd(environment, agent, model, trials=500, max_steps_per_episode=1000, learn=False):
    """The play function runs iterations and updates Q-values if desired."""
    reward_per_episode = [] # Initialise performance log
    if learn:
        agent.q_table_withatt = copy.deepcopy(agent.q_table)
        agent.defender_policy = agent.pareto_update(agent.get_policy(init = True))
    for trial in range(trials): # Run trials
        print("Round ", trial)
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
            tensor_old_state = torch.from_numpy(np.ascontiguousarray(old_state)).unsqueeze(0).cuda().to(torch.float32)
            #att_state = agent.get_att_state(old_state, agent.defender_policy)
            y = model.act(tensor_old_state)
            att_state = tuple(pgd(model, tensor_old_state, y, env_id = "Grid").data[0].detach().cpu().numpy().tolist())
            if (att_state[0]!=old_state[0]) and (att_state[1]!= old_state[1]):
                print("detect")
                att_state = list(att_state)
                if np.random.uniform()>=0.5:
                    att_state[0] = old_state[0]
                else:
                    att_state[1] = old_state[1]
                att_state = tuple(att_state)
            if learn:
                eps = linear_schedule_exp(0.01, 0.1, trials, trial)
            else:
                eps = 0

            att_state = agent.get_ball(old_state)[np.random.choice(range(len(agent.get_ball(old_state))))]
            action = agent.choose_action_att(environment.actions , att_state, eps)
            possible_true_states = agent.get_ball(att_state)
            belief_before_action = states_intersection(belief, possible_true_states)
            belief = set()
            for state in belief_before_action:
                belief.add(environment.take_action(state, action))
            if not learn:
                print(step)
                print("true state is", old_state)
                print("noise state is ", att_state)
                print("belief is",belief)

            new_policy = agent.get_policy_with_belief(agent.defender_policy, belief)
            _,reward,_,_ = environment.step(action)
            if not learn:
                print("next state is ", environment.current_location)
            new_state = environment.current_location

            if learn == True: # Update Q-values if learning is specified
                alpha = linear_schedule_exp(0.005, 0.1, trials, trial)
                agent.learn_with_att_with_belief(old_state, reward, new_state, action, agent.defender_policy, new_policy, belief, alpha)
            agent.defender_policy = new_policy

            cumulative_reward += reward
            step += 1

            if environment.check_state() == 'TERMINAL': # If game is in terminal state, game over and start next trial
                environment.__init__()
                game_over = True
        print(cumulative_reward)

        reward_per_episode.append(cumulative_reward) # Append reward for current trial to performance log
    return reward_per_episode # Return performance log

def train_RL_attack_with_belief(agentQ):
    env = GridWorld_RL_Attack(agentQ)
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path="RL_attack",
                                             name_prefix='rl_model')

    model = DQN("MlpPolicy", env, learning_rate = 1e-3, learning_starts = 1000, batch_size = 256, exploration_final_eps=0.001,
                gamma = 1, tau = 0.2, train_freq = (5, "step"), policy_kwargs={"net_arch" : [256,128]}, tensorboard_log="RL_attack/", verbose=1)

    model.learn(total_timesteps=20000, callback = checkpoint_callback)
    return model


def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))

def main():
    environment = GridWorld()
    agentQ = Q_Agent(environment)

    # Note the learn=True argument!
    reward_per_episode = play(environment, agentQ, trials=2000, learn=True)
    reward_no_defense = play_att(environment, agentQ, trials = 100, learn=False)
    print("Begin Attack Learning")
    reward_per_episode = play_att_with_belief(environment, agentQ, trials=500, learn=True)

    #reward_per_episode_test = play_att_with_belief_pgd(environment, agentQ, model, trials=100, learn=False)
    #attack_model = train_RL_attack_with_belief(agentQ)
    #reward_per_episode_test = play_att_with_belief(environment, agentQ, trials=100, learn=False, attack_model = attack_model)
    reward_per_episode_test = play_att_with_belief(environment, agentQ, trials=100, learn=False, attack_model = None)

    print("No_attack_reward", np.mean(reward_per_episode))
    print("Reward_with_no_defense", np.mean(reward_no_defense))
    print("Myopic_attack", np.mean(reward_per_episode_test))

    
if __name__ == "__main__":
    main()

