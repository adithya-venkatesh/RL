# -*- coding: utf-8 -*-
"""main_vanilla_pessimistic_offpolicy.py

Gridworld RL comparison: Vanilla vs Pessimistic vs Off-Policy Greedy.

Reference: Sun & Zheng (ICLR 2024) — Belief-Enriched Pessimistic Q-Learning
Against Adversarial State Perturbations.

Compares three algorithms:
1. Vanilla Q-Learning (baseline) — Eq. 4.1
2. Pessimistic Q-Learning — Algorithm 1 + Algorithm 2 (belief)
3. Off-Policy Greedy Target (extension) — Eq. 5.1–5.2

Includes inference timing, CSV logging, pause/resume, and progress tracking.
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

import stable_baselines3
from stable_baselines3 import SAC, PPO, TD3, DQN
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback

def linear_schedule_exp(min_exp, ini_exp, total, current):
    """**Eq. 6.3:** ε_t = max(ε_min, (T − t) · ε_init / T)

    Annealing schedule for exploration rate ε′_t and learning rate α_t.
    """
    return max(min_exp, ((total-current)*ini_exp)/total)

# Eq. 1.1: MDP M = ⟨S, A, P, R, γ⟩
#   S = 5×5 grid states, A = {UP, DOWN, LEFT, RIGHT}
#   R(bomb) = -10, R(gold) = +10, R(default) = -1
class GridWorld(gym.Env):
    def __init__(self):
        """**Eq. 1.1:** M = ⟨S, A, P, R, γ⟩

        S = 5×5 grid; A = {UP, DOWN, LEFT, RIGHT};
        R(bomb)=−10, R(gold)=+10, R(default)=−1; γ used by Q_Agent.
        """
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

    def get_available_actions(self):
        """**Eq. 1.1:** a ∈ A = {UP, DOWN, LEFT, RIGHT}

        Returns the discrete action space A of the MDP.
        """
        return self.actions

    def agent_on_map(self):
        """Visualizes agent position on the grid for debugging."""
        grid = np.zeros(( self.height, self.width))
        grid[ self.current_location[0], self.current_location[1]] = 1
        return grid

    def get_reward(self, new_location):
        """**Eq. 1.2:** R(s, a) = E[R_t | s_{t−1}=s, a_{t−1}=a]

        In this grid, reward depends on state s only (bomb/gold/default cells).
        """
        return self.grid[ new_location[0], new_location[1]]

    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        if seed is not None:
            np.random.seed(seed)

    def step(self, action):
        """**Eq. 6.1:** s′ = T(s, a);  **Eq. 6.2:** terminated = 𝟙[s′ ∈ {s_bomb, s_gold}]

        Environment transition: observe R(s,a), move agent, return (s′, R, done).
        """
        # Store previous location
        last_location = self.current_location
        if action in [0,1,2,3]:
            true_action = self.actions[action]
        else:
            true_action = action

        # UP
        if true_action == 'UP':
            if last_location[0] == 0:
                reward = self.get_reward(last_location)
            else:
                self.current_location = [self.current_location[0] - 1, self.current_location[1]]
                reward = self.get_reward(self.current_location)

        # DOWN
        elif true_action == 'DOWN':
                if last_location[0] == self.height - 1:
                    reward = self.get_reward(last_location)
                else:
                    self.current_location =  [self.current_location[0] + 1, self.current_location[1]]
                    reward = self.get_reward(self.current_location)

        # LEFT
        elif true_action == 'LEFT':
                if last_location[1] == 0:
                    reward = self.get_reward(last_location)
                else:
                    self.current_location = [self.current_location[0], self.current_location[1] - 1]
                    reward = self.get_reward(self.current_location)

        # RIGHT
        elif true_action == 'RIGHT':
                if last_location[1] == self.width - 1:
                    reward = self.get_reward(last_location)
                else:
                    self.current_location = [self.current_location[0], self.current_location[1] + 1]
                    reward = self.get_reward(self.current_location)
        self.total_step += 1
      
        terminated = self.current_location in self.terminal_states
        truncated = self.total_step >= 1000

        return np.asarray(self.current_location), reward, terminated, truncated, {}

    def take_action(self, state, action):
        """**Eq. 6.1:** s′ = T(s, a)  (deterministic transition, no reward)

        Used in belief propagation (Eq. 3.2): M′_t = {s′ : ∃s∈M_t, P(s′|s,a)>0}.
        """
        last_location = state
        if action in [0,1,2,3]:
            true_action = self.actions[action]
        else:
            true_action = action

        # UP
        if true_action == 'UP':
            if last_location[0] == 0:
                state = state
            else:
                state = [state[0] - 1, state[1]]

        # DOWN
        elif true_action == 'DOWN':
                if last_location[0] == self.height - 1:
                    state = state
                else:
                    state =  [state[0] + 1, state[1]]
        # LEFT
        elif true_action == 'LEFT':
                if last_location[1] == 0:
                    state = state
                else:
                    state = [state[0], state[1] - 1]

        # RIGHT
        elif true_action == 'RIGHT':
                if last_location[1] == self.width - 1:
                    state = state
                else:
                    state = [state[0], state[1] + 1]

        return tuple(state)

    def check_state(self):
        """**Eq. 6.2:** terminated = 𝟙[s ∈ {s_bomb, s_gold}]"""
        if self.current_location in self.terminal_states:
            return 'TERMINAL'
    
    def reset(self, seed=None):
        """**Algorithm 1, Line 3:** Initialize true state s₀ ∈ S \\ {terminals}

        Sample random non-terminal starting position for a new episode.
        """
        self.total_step = 0
        if seed is not None:  
            np.random.seed(seed)
        
        self.current_location = [np.random.randint(0,5), np.random.randint(0,5)]
        
        while self.current_location in self.terminal_states:
            self.current_location = [np.random.randint(0,5), np.random.randint(0,5)]
        return np.asarray(self.current_location)

# Eq. 1.5–1.6: Attacker MDP ⟨S, S, P̃, R̃, γ⟩ for training DQN attacker (Section 2.2)
#   R̃(s, s̃) = − Σ_a π(a|s̃) R(s,a)
#   P̃(s'|s, s̃) = Σ_a π(a|s̃) P(s'|s,a)
class GridWorld_RL_Attack(gym.Env):
    def __init__(self, agentQ, agent_type='pessimistic'):
        """**Eq. 1.5–1.6:** Attacker MDP ⟨S, S, P̃, R̃, γ⟩

        R̃(s, s̃) = −Σ_a π(a|s̃) R(s,a);  P̃(s′|s,s̃) = Σ_a π(a|s̃) P(s′|s,a).
        Wraps a trained defender (pessimistic or off-policy) for DQN attacker training.
        """
        self.agent_type = agent_type
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

    def get_available_actions(self):
        """**Eq. 1.4:** Attacker action space A_att = {UP, DOWN, LEFT, RIGHT, STAY}

        Each action selects a perturbation s̃ ∈ B_ε(s).
        """
        return self.actions

    def agent_on_map(self):
        """Visualizes agent position on the grid for debugging."""
        grid = np.zeros(( self.height, self.width))
        grid[ self.current_location[0], self.current_location[1]] = 1
        return grid

    def get_reward(self, new_location):
        """**Eq. 1.2:** R(s, a) — defender reward at state s (negated for attacker in step)."""
        return self.grid[ new_location[0], new_location[1]]

    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        if seed is not None:
            np.random.seed(seed)

    def step(self, action):
        """**Eq. 1.4–1.6:** s̃ ∈ B_ε(s);  a = π(s̃);  R̃(s,s̃) = −R(s,a)

        Attacker MDP step: perturb observation, defender acts, attacker gets R̃.
        """
        # Store previous location
        last_location = self.current_location
        if action in [0,1,2,3,4]:
            true_action = self.actions[action]
        else:
            true_action = action
        # Eq. 1.4: Compute perturbed state s̃ from attacker action (s̃ ∈ B_ε(s))
        # UP
        if true_action == 'UP':
            if last_location[0] == 0:
                self.purturb_state = self.current_location
            else:
                self.purturb_state = [self.current_location[0] - 1, self.current_location[1]]
        # DOWN
        elif true_action == 'DOWN':
                if last_location[0] == self.height - 1:
                    self.purturb_state = self.current_location
                else:
                    self.purturb_state =  [self.current_location[0] + 1, self.current_location[1]]
        # LEFT
        elif true_action == 'LEFT':
                if last_location[1] == 0:
                    self.purturb_state = self.current_location
                else:
                    self.purturb_state = [self.current_location[0], self.current_location[1] - 1]
        # RIGHT
        elif true_action == 'RIGHT':
                if last_location[1] == self.width - 1:
                    self.purturb_state = self.current_location
                else:
                    self.purturb_state = [self.current_location[0], self.current_location[1] + 1]
        elif true_action == 'STAY':
            self.purturb_state = self.current_location

        # Agent action selection depends on agent type
        if self.agent_type == 'pessimistic':
            # Eq. 2.1 + Eq. 3.4: a = π(s̃) via belief-enriched defender policy
            agent_action = self.agentQ.choose_action_att(
                ['UP', 'DOWN', 'LEFT', 'RIGHT'], 
                tuple(self.purturb_state), 
                0
            )
            self.current_location = self.take_action(self.current_location, agent_action)
            
            # Eq. 3.2–3.3: Belief update M_{t+1} = M'_t ∩ B_ε(s̃_{t+1})
            possible_true_states = self.agentQ.get_ball(self.purturb_state)
            belief_before_action = states_intersection(self.belief, possible_true_states)
            self.belief = set()
            for state in belief_before_action:
                self.belief.add(self.take_action(state, agent_action))
            self.agentQ.defender_policy = self.agentQ.get_policy_with_belief(
                self.agentQ.defender_policy, self.belief
            )
        
        elif self.agent_type == 'offpolicy':
            # Eq. 4.2: Greedy test policy a = argmax_a Q(s̃, a) (no belief at test)
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
        # Eq. 1.5: Attacker reward R̃(s, s̃) = −R(s, a)
        reward = -self.get_reward(self.current_location)
        self.total_step += 1
        truncated = self.total_step >= 1000

        return np.asarray(self.current_location), reward, terminated, truncated, {}

    def take_action(self, state, action):
        """**Eq. 6.1:** s′ = T(s, a)  (deterministic transition for defender in attacker MDP)"""
        last_location = state
        if action in [0,1,2,3]:
            true_action = self.actions[action]
        else:
            true_action = action

        # UP
        if true_action == 'UP':
            if last_location[0] == 0:
                state = state
            else:
                state = [state[0] - 1, state[1]]

        # DOWN
        elif true_action == 'DOWN':
                if last_location[0] == self.height - 1:
                    state = state
                else:
                    state =  [state[0] + 1, state[1]]
        # LEFT
        elif true_action == 'LEFT':
                if last_location[1] == 0:
                    state = state
                else:
                    state = [state[0], state[1] - 1]

        # RIGHT
        elif true_action == 'RIGHT':
                if last_location[1] == self.width - 1:
                    state = state
                else:
                    state = [state[0], state[1] + 1]

        return tuple(state)

    def check_state(self):
        """**Eq. 6.2:** terminated = 𝟙[s ∈ {s_bomb, s_gold}]"""
        if self.current_location == self.bomb_location:
            print("BOOM")
        if self.current_location == self.gold_location:
            print("GOLD")
        if self.current_location in self.terminal_states:
            return 'TERMINAL'

    def reset(self, seed=None):
        """**Eq. 3.1:** M_0 = B_ε(s̃_0)  (initialized to all states in this tabular grid)

        Reset attacker environment and defender belief/policy.
        """
        self.current_location =  [np.random.randint(0,5), np.random.randint(0,5)]
        # Eq. 3.1: Initialize belief M_0 = all states (relaxed from B_ε(s̃_0) for tabular grid)
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

# Algorithm 1: Pessimistic Q-Learning (tabular implementation)
# Maintains Q(s,a) in q_table (vanilla) and q_table_withatt (robust)
class Q_Agent():
    def __init__(self, environment, epsilon=0.05, alpha=0.1, gamma=1):
        """**Algorithm 1, Line 1:** Initialize Q(s,a) ∀ s∈S, a∈A

        Tabular Q-tables: q_table (vanilla) and q_table_withatt (robust).
        """
        self.environment = environment
        self.q_table = dict()
        self.q_table_withatt = dict()
        self.all_state = []
        for x in range(environment.height):
            for y in range(environment.width):
                self.all_state.append((x,y))
                self.q_table[(x,y)] = {'UP':np.random.uniform(0,1), 'DOWN':np.random.uniform(0,1), 'LEFT':np.random.uniform(0,1), 'RIGHT':np.random.uniform(0,1)}
                self.q_table_withatt[(x,y)] = {'UP':np.random.uniform(0,1), 'DOWN':np.random.uniform(0,1), 'LEFT':np.random.uniform(0,1), 'RIGHT':np.random.uniform(0,1)}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.defender_policy = None

    def choose_action(self, available_actions, eps = 0.05):
        """**Eq. 2.3 / Eq. 4.2:** a = argmax_a Q(s,a) w.p. 1−ε′; else a ∈ A uniform

        Vanilla ε-greedy policy (Algorithm 1, Lines 8–9, no perturbation).
        """
        if np.random.uniform(0,1) < eps:
            action = available_actions[np.random.randint(0, len(available_actions))]
        else:
            q_values_of_state = self.q_table[tuple(self.environment.current_location)]
            maxValue = max(q_values_of_state.values())
            action = np.random.choice([k for k, v in q_values_of_state.items() if v == maxValue])

        return action

    def choose_action_att(self, available_actions, state, eps = 0.05):
        """**Eq. 2.3 / Eq. 3.4:** a = π(s̃) w.p. 1−ε′; else a ∈ A uniform

        π(s̃) from get_policy_with_belief (Algorithm 5, Line 7).
        """
        if np.random.uniform(0,1) < eps:
            action = available_actions[np.random.randint(0, len(available_actions))]
        else:
            action = self.defender_policy[state]
        return action
    
    def learn(self, old_state, reward, new_state, action, alpha = 0.1):
        """**Eq. 4.1 / Eq. 1.3:** Q(s,a) ← (1−α)Q(s,a) + α[R + γ max_{a′} Q(s′,a′)]

        Standard vanilla Q-learning temporal-difference update.
        """
        q_values_of_state = self.q_table[tuple(new_state)]
        max_q_value_in_new_state = max(q_values_of_state.values())
        current_q_value = self.q_table[tuple(old_state)][action]

        self.q_table[tuple(old_state)][action] = (1 - alpha) * current_q_value + alpha * (reward + self.gamma * max_q_value_in_new_state)

    def learn_with_greedy_target(self, old_state, reward, new_state, action, alpha=0.1):
        """**Eq. 5.2:** Q(s,a) ← (1−α)Q(s,a) + α[R + γ max_{a′} Q(s′,a′)]

        Off-policy extension: greedy bootstrap (Eq. 1.3), pessimistic behavior (Eq. 2.1).
        """
        # Eq. 1.3: Greedy bootstrap max_{a'} Q(s', a')
        max_q_value = max(self.q_table_withatt[tuple(new_state)].values())
    
        current_q_value = self.q_table_withatt[tuple(old_state)][action]
    
        self.q_table_withatt[tuple(old_state)][action] = \
            (1 - alpha) * current_q_value + alpha * (reward + self.gamma * max_q_value)
    
    def find_maxmin_with_belief(self, q_table_true, q_table_withatt, next_state, policy, new_policy, belief):
        """**Eq. 2.6:** (T_{car}Q)(s,a) target = γ · max_{a′} min_{m ∈ M∩B_ε(s′)} Q(m,a′)

        Pessimistic Bellman target (Algorithm 4, Line 13; BP-DQN loss, tabular form).
        Intersects the propagated belief with B_ε(s′) for the *true* next state s′,
        then takes max_a' min_m Q(m, a') over that set.
        """
        states = self.get_ball(next_state)
        possible_states = states_intersection(belief, states)
        values = dict()
        
        if len(possible_states) > 1:
            for action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                q_values_for_action = []
                for state in possible_states:
                    q_values_for_action.append(q_table_withatt[state][action])
                # min_{m ∈ M∩B} Q(m, a) for each action a
                values[action] = min(q_values_for_action)
            # max_{a'} over actions
            return max(values.values())
        elif len(possible_states) == 1:
            return max(q_table_withatt[list(possible_states)[0]].values())
        else:
            return max(q_table_withatt[next_state].values())

    def learn_with_att_with_belief(self, old_state, reward, new_state, action, policy, new_policy, belief, alpha = 0.1):
        """**Eq. 2.4 / Algorithm 1, Line 11:** Q(s,a) ← Q(s,a) + α[R + γ·target − Q(s,a)]

        target from find_maxmin_with_belief (Eq. 2.6).
        """
        max_q_value_in_new_state = self.find_maxmin_with_belief(self.q_table, self.q_table_withatt, tuple(new_state), policy, new_policy, belief)
        current_q_value = self.q_table_withatt[tuple(old_state)][action]

        self.q_table_withatt[tuple(old_state)][action] = (1 - alpha) * current_q_value + alpha * (reward + self.gamma * max_q_value_in_new_state)

    def get_att_state(self, true_state, policy):
        """**Eq. 2.2 / Algorithm 1, Line 6:** ω_π(s) = argmin_{s̃∈B_ε(s)} Q(s, π(s̃))

        Worst-case perturbed state for pessimistic training.
        Scores each candidate observation by the true-state value of the
        action the current policy would take under that observation.
        """
        states = self.get_ball(true_state)
        values = dict()
        for i in states:
            values[i] = self.q_table_withatt[true_state][policy[i]]
        att_state = min(values, key = values.get)
        return att_state

    def pareto_update(self, policy):
        """**Eq. 2.1 (iterative):** π(s) = argmax_a min_{s̄∈B_ε(s)} Q(s̄,a) until convergence

        Fixed-point policy initialization before pessimistic training.
        """
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
        """**Eq. 2.1 / Algorithm 1, Line 5:** π(s̃) = argmax_a min_{s̄∈B_ε(s̃)} Q(s̄,a)

        Tabular maximin policy over the ε-ball.
        """
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
        """**Eq. 4.2:** π(s) = argmax_a Q(s,a)"""
        policy = dict()
        for state in self.all_state:
            policy[state] = max(self.q_table_withatt[state], key = self.q_table_withatt[state].get)
        return policy

    def get_policy_with_belief(self, policy, belief):
        """**Eq. 3.4 / Algorithm 5, Line 7:** π(s̃) = argmax_a min_{m∈M_t∩B_ε(s̃)} Q(m,a)
        """
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
        """**Eq. 1.4:** B_ε(s) = {s} ∪ {one-step neighbors on 5×5 grid}

        Discrete ε-ball: L1 neighborhood for state perturbation constraint.
        """
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

def play(environment, agent, trials=500, max_steps_per_episode=1000, learn=False):
    """**Eq. 4.1:** Q(s,a) ← (1−α)Q(s,a) + α[R + γ max_{a′} Q(s′,a′)]

    Vanilla Q-learning episode loop with ε-greedy exploration (Eq. 2.3).
    """
    reward_per_episode = []

    for trial in range(trials):
        cumulative_reward = 0
        step = 0
        game_over = False
        old_state = environment.reset()
        while step < max_steps_per_episode and game_over != True:
            old_state = tuple(environment.current_location)
            if learn:
                eps = linear_schedule_exp(0.01, 0.2, trials, trial)
            else:
                eps = 0
            action = agent.choose_action(environment.actions, eps)
            
            _, reward, terminated, truncated, _ = environment.step(action)
            new_state = environment.current_location

            if learn == True:
                alpha = linear_schedule_exp(0.005, 0.1, trials, trial)
                agent.learn(old_state, reward, new_state, action, alpha)

            cumulative_reward += reward
            step += 1

            if environment.check_state() == 'TERMINAL':
                environment.__init__()
                game_over = True

        reward_per_episode.append(cumulative_reward)
    
    return reward_per_episode

def play_vanilla_test(environment, agent, trials=100, max_steps_per_episode=1000, measure_time=False, attack_model=None):
    """**Eq. 4.2:** a = argmax_a Q(s̃, a)

    Test vanilla agent; optional attacker provides perturbed observation s̃.
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
            
            # Get attacker perturbation if attack model provided
            if attack_model is not None:
                attack_obs = np.array(old_state)
                attack_action, _ = attack_model.predict(attack_obs, deterministic=True)
                attack_action = int(attack_action.item() if hasattr(attack_action, "item") else attack_action)
                
                if attack_action in [0, 1, 2, 3, 4]:
                    perturbation_direction = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY'][int(attack_action)]
                else:
                    perturbation_direction = attack_action
                
                perturbed_state = apply_perturbation(old_state, perturbation_direction)
            else:
                perturbed_state = old_state
            
            # Start timing
            if measure_time:
                action_start_time = time.perf_counter()
            
            # Greedy action selection from baseline Q-table
            q_values = agent.q_table[perturbed_state]
            maxValue = max(q_values.values())
            action = np.random.choice([k for k, v in q_values.items() if v == maxValue])
            
            # End timing
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

def states_intersection(current_belief, states):
    """**Eq. 3.3 / Algorithm 2, Line 9:** M_{t+1} = M′_t ∩ B_ε(s̃)

    Returns belief states that also lie in the ε-ball B_ε(s̃).
    """
    new_belief = set()
    for state in states:
        if state in current_belief:
            new_belief.add(state)
    return new_belief

def play_att_with_belief(environment, agent, trials=500, max_steps_per_episode=1000, learn=False, attack_model = None, measure_time=False):
    """**Algorithm 1 + Algorithm 2:** Pessimistic Q-learning with belief update

    Eq. 2.2 (ω_π), Eq. 3.4 (π with M_t), Eq. 2.6 (Q-update), Eq. 3.2–3.3 (belief).
    """
    inference_times = []
    reward_per_episode = []
    if learn:
        agent.q_table_withatt = copy.deepcopy(agent.q_table)
        agent.defender_policy = agent.pareto_update(agent.get_policy(init = True))
    
    for trial in range(trials):
        belief = set()
        for state in agent.all_state:
            belief.add(state)
        cumulative_reward = 0
        step = 0
        game_over = False
        old_state = environment.reset()
        agent.defender_policy = agent.get_policy_with_belief(agent.defender_policy, belief)
        while step < max_steps_per_episode and game_over != True:
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
                # Eq. 2.2: Training-time attacker ω_π(s) = argmin_{s̃∈B_ε(s)} Q(s, π(s̃))
                att_state = agent.get_att_state(old_state, agent.defender_policy)
                
            if learn:
                eps = linear_schedule_exp(0.01, 0.2, trials, trial)
            else:
                eps = 0
            
            # START TIMING NOW (after attacker, before agent inference)
            if not learn and measure_time:
                action_start_time = time.perf_counter()
            
            action = agent.choose_action_att(environment.actions , att_state, eps)
            
            # Eq. 3.2–3.3: Belief propagation and intersection with B_ε(s̃)
            possible_true_states = agent.get_ball(att_state)
            belief_before_action = states_intersection(belief, possible_true_states)
            belief = set()
            for state in belief_before_action:
                belief.add(environment.take_action(state, action))
            
            # Eq. 3.4: Update defender policy π from belief M_t
            new_policy = agent.get_policy_with_belief(agent.defender_policy, belief)
            
            # End timing AFTER full inference (belief + policy update)
            if not learn and measure_time:
                action_end_time = time.perf_counter()
                inference_times.append(action_end_time - action_start_time)
            
            _, reward, terminated, truncated, _ = environment.step(action)
            
            new_state = environment.current_location

            if learn == True:
                alpha = linear_schedule_exp(0.01, 0.1, trials, trial)
                agent.learn_with_att_with_belief(old_state, reward, new_state, action, agent.defender_policy, new_policy, belief, alpha)
            agent.defender_policy = new_policy

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

def play_offpolicy_greedy_target(environment, agent, trials=500, max_steps_per_episode=1000, learn=False, measure_time=False, attack_model=None):
    """**Eq. 5.1:** Behavior = pessimistic (Eq. 2.1–3.4); Target = greedy (Eq. 5.2)

    Train with belief+maximin; test with Eq. 4.2 greedy on s̃ (no belief).
    """
    reward_per_episode = []
    inference_times = []
    
    if learn:
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
                
                # Eq. 2.2: Worst-case perturbation during training
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
                
                # Eq. 5.2: Off-policy greedy target update (not Eq. 2.6)
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
            # TESTING: Eq. 4.2 — greedy on perturbed state, no belief (Eq. 3.4 skipped)
            while step < max_steps_per_episode and game_over != True:
                old_state = tuple(environment.current_location)
                
                eps = 0
                
                # Get attacker perturbation FIRST (NOT timed)
                if attack_model is not None:
                    attack_obs = np.array(old_state)
                    attack_action, _ = attack_model.predict(attack_obs, deterministic=True)
                    attack_action = int(attack_action.item() if hasattr(attack_action, "item") else attack_action)
                    
                    if attack_action in [0, 1, 2, 3, 4]:
                        perturbation_direction = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY'][int(attack_action)]
                    else:
                        perturbation_direction = attack_action
                    
                    perturbed_state = apply_perturbation(old_state, perturbation_direction)
                else:
                    perturbed_state = old_state
                
                # START TIMING NOW (after attacker, before agent inference)
                if measure_time:
                    action_start_time = time.perf_counter()
                
                # Eq. 4.2: a = argmax_a Q(s̃, a)
                q_values_of_state = agent.q_table_withatt[perturbed_state]
                
                # Greedy action selection
                maxValue = max(q_values_of_state.values())
                action = np.random.choice([k for k, v in q_values_of_state.items() if v == maxValue])
                
                # END TIMING (after agent inference only)
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

def apply_perturbation(state, perturbation_direction):
    """**Eq. 1.4:** s̃ ∈ B_ε(s) — map attack direction to perturbed grid cell"""
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

def train_RL_attack_with_belief(agentQ, agent_type='pessimistic', model_name='attacker'):
    """**Eq. 6.4 / Eq. 1.5:** min_ω Σ_t R̃(s_t, ω(s_t));  R̃(s,s̃) = −R(s, π(s̃))

    Train DQN attacker ω on attacker MDP (PA-AD / Section 2.2).
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

def log_experiment_to_csv(experiment_number, results_dict, filename='gridworld_experiments.csv'):
    """**Eq. 7.1:** R̄ = (1/N) Σ_i R_i  — log empirical mean rewards and metrics to CSV

    Infrastructure function; records experiment outputs (no learning update).
    """
    import csv
    import os
    
    file_exists = os.path.isfile(filename)
    
    fieldnames = [
        'experiment_number',
        'timestamp',
        'seed',
        'baseline_avg_reward',
        'vanilla_test_avg_reward',
        'vanilla_test_std',
        'vanilla_inference_time_ms',
        'vanilla_vs_attacker_reward',
        'vanilla_vs_attacker_std',
        'vanilla_attack_inference_time_ms',
        'pessimistic_train_avg_reward',
        'pessimistic_train_std',
        'pessimistic_test_avg_reward',
        'pessimistic_test_std',
        'pessimistic_inference_time_ms',
        'pessimistic_vs_attacker_reward',
        'pessimistic_vs_attacker_std',
        'pessimistic_attack_inference_time_ms',
        'offpolicy_greedy_train_avg_reward',
        'offpolicy_greedy_train_std',
        'offpolicy_greedy_test_avg_reward',
        'offpolicy_greedy_test_std',
        'offpolicy_greedy_inference_time_ms',
        'offpolicy_greedy_vs_attacker_reward',
        'offpolicy_greedy_vs_attacker_std',
        'offpolicy_attack_inference_time_ms',
        'performance_diff_offpolicy_vs_pessimistic',
        'performance_diff_offpolicy_vs_vanilla',
        'performance_diff_pessimistic_vs_vanilla',
        'speedup_offpolicy_vs_pessimistic',
        'speedup_offpolicy_vs_vanilla',
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

def save_checkpoint(experiment_number, checkpoint_file='experiment_checkpoint.pkl'):
    """Save the current experiment number to a checkpoint file."""
    import pickle
    import os
    
    checkpoint_data = {
        'last_completed_experiment': experiment_number,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)

def load_checkpoint(checkpoint_file='experiment_checkpoint.pkl'):
    """Load the last completed experiment number from checkpoint file."""
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

def clear_checkpoint(checkpoint_file='experiment_checkpoint.pkl'):
    """Clear the checkpoint file when all experiments are complete."""
    import os
    
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"✓ Checkpoint file '{checkpoint_file}' removed")

def check_experiment_progress(csv_filename='gridworld_experiments.csv', 
                              checkpoint_file='experiment_checkpoint.pkl',
                              total_experiments=500):
    """**Eq. 7.1:** R̄ = (1/N) Σ_i R_i  — report running mean rewards from CSV

    Progress monitor; no learning update.
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
        if 'vanilla_test_avg_reward' in df.columns:
            print(f"    Vanilla:     {df['vanilla_test_avg_reward'].mean():.2f}")
        if 'pessimistic_test_avg_reward' in df.columns:
            print(f"    Pessimistic: {df['pessimistic_test_avg_reward'].mean():.2f}")
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
        avg_time_per_exp = 10  # minutes (rough estimate)
        estimated_hours = (remaining * avg_time_per_exp) / 60
        print(f"Estimated time remaining: ~{estimated_hours:.1f} hours")
        print(f"  ({remaining} experiments × ~{avg_time_per_exp} min/exp)")
    
    print("="*70)
    
    return completed

def analyze_csv_results(csv_filename='gridworld_experiments.csv'):
    """**Eq. 7.1:** R̄ ± σ = mean ± std of per-experiment rewards

    Summarize vanilla, pessimistic, and off-policy results from CSV.
    """
    import pandas as pd
    
    # Read CSV
    df = pd.read_csv(csv_filename)
    
    print("="*70)
    print(f"ANALYSIS OF {len(df)} EXPERIMENTS")
    print("="*70)
    
    # Summary statistics
    print("\n--- Vanilla Q-Learning (Baseline) ---")
    if 'vanilla_test_avg_reward' in df.columns:
        vanilla_clean = df['vanilla_test_avg_reward'].dropna()
        if len(vanilla_clean) > 0:
            print(f"Mean test reward: {vanilla_clean.mean():.2f} ± {vanilla_clean.std():.2f}")
    
    if 'vanilla_vs_attacker_reward' in df.columns:
        vanilla_attack = df['vanilla_vs_attacker_reward'].dropna()
        if len(vanilla_attack) > 0:
            print(f"Mean vs attacker: {vanilla_attack.mean():.2f} ± {vanilla_attack.std():.2f}")

    print("\n--- Pessimistic Q-Learning")
    if 'pessimistic_test_avg_reward' in df.columns:
        pessimistic_clean = df['pessimistic_test_avg_reward'].dropna()
        if len(pessimistic_clean) > 0:
            print(f"Mean test reward: {pessimistic_clean.mean():.2f} ± {pessimistic_clean.std():.2f}")
    
    if 'pessimistic_vs_attacker_reward' in df.columns:
        pessimistic_attack = df['pessimistic_vs_attacker_reward'].dropna()
        if len(pessimistic_attack) > 0:
            print(f"Mean vs attacker: {pessimistic_attack.mean():.2f} ± {pessimistic_attack.std():.2f}")
    
    print("\n--- Off-Policy Greedy Target ---")
    if 'offpolicy_greedy_test_avg_reward' in df.columns:
        offpolicy_clean = df['offpolicy_greedy_test_avg_reward'].dropna()
        if len(offpolicy_clean) > 0:
            print(f"Mean test reward: {offpolicy_clean.mean():.2f} ± {offpolicy_clean.std():.2f}")
    
    if 'offpolicy_greedy_vs_attacker_reward' in df.columns:
        offpolicy_attack = df['offpolicy_greedy_vs_attacker_reward'].dropna()
        if len(offpolicy_attack) > 0:
            print(f"Mean vs attacker: {offpolicy_attack.mean():.2f} ± {offpolicy_attack.std():.2f}")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
    
    return df

def play_pessimistic_clean(environment, agent, trials=100, max_steps_per_episode=1000, measure_time=False):
    """**Eq. 4.2:** a = argmax_a Q(s, a)  on true state s (no attacker, no belief)

    Clean-environment test of trained q_table_withatt.
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

def run_single_experiment_with_seed(experiment_num, seed, attack_models=None):
    """**Eq. 7.1–7.2:** Compare R̄ and speedup across vanilla / pessimistic / off-policy

    Phase 1: Eq. 4.1 train | 1.5: Eq. 4.2 test | 2: Algorithm 1+2 | 3: Eq. 5.1–5.2
    """
    results = {}
    results['experiment_number'] = experiment_num
    results['seed'] = seed
    
    # ============================================================
    # PHASE 1: BASELINE TRAINING (Same for all)
    # ============================================================
    print(f"\n[Exp {experiment_num + 1}] Phase 1: Baseline training...")
    
    environment_baseline = GridWorld()
    environment_baseline.seed(seed)
    agent_baseline = Q_Agent(environment_baseline)
    
    np.random.seed(seed + 1000)
    
    reward_baseline = play(environment_baseline, agent_baseline, trials=2000, learn=True)
    results['baseline_avg_reward'] = np.mean(reward_baseline[-100:])
    
    baseline_q_table = copy.deepcopy(agent_baseline.q_table)
    
    # ============================================================
    # PHASE 1.5: TEST VANILLA Q-LEARNING
    # ============================================================
    print(f"[Exp {experiment_num + 1}] Phase 1.5: Testing Vanilla Q-Learning...")

    np.random.seed(seed + 1000)
    reward_vanilla_test, vanilla_time = play_vanilla_test(
        environment_baseline, agent_baseline, trials=100, measure_time=True
    )
    results['vanilla_test_avg_reward'] = np.mean(reward_vanilla_test)
    results['vanilla_test_std'] = np.std(reward_vanilla_test)
    results['vanilla_inference_time_ms'] = vanilla_time * 1000

    if attack_models is not None and 'pessimistic' in attack_models:
        np.random.seed(seed + 1000)
        reward_vanilla_attack, vanilla_attack_time = play_vanilla_test(
            environment_baseline, agent_baseline, trials=100,
            attack_model=attack_models['pessimistic'], measure_time=True
        )
        results['vanilla_attack_inference_time_ms'] = vanilla_attack_time * 1000
        results['vanilla_vs_attacker_reward'] = np.mean(reward_vanilla_attack)
        results['vanilla_vs_attacker_std'] = np.std(reward_vanilla_attack)
    else:
        results['vanilla_vs_attacker_reward'] = np.nan
        results['vanilla_vs_attacker_std'] = np.nan

    # ============================================================
    # PHASE 2: PESSIMISTIC Q-LEARNING
    # ============================================================
    print(f"[Exp {experiment_num + 1}] Phase 2: Training Pessimistic Q-Learning...")
    
    environment2 = GridWorld()
    environment2.seed(seed)
    agentQ2 = Q_Agent(environment2)
    agentQ2.q_table = copy.deepcopy(baseline_q_table)
    
    np.random.seed(seed + 1000)
    reward_train2 = play_att_with_belief(environment2, agentQ2, trials=500, learn=True)
    results['pessimistic_train_avg_reward'] = np.mean(reward_train2)
    results['pessimistic_train_std'] = np.std(reward_train2)
    
    # Test in CLEAN environment (NO attacker) - use play_pessimistic_clean for greedy on true state
    np.random.seed(seed + 1000)
    reward_test2, pessimistic_time = play_pessimistic_clean(
        environment2, agentQ2, trials=100, measure_time=True)
    results['pessimistic_test_avg_reward'] = np.mean(reward_test2)
    results['pessimistic_test_std'] = np.std(reward_test2)
    results['pessimistic_inference_time_ms'] = pessimistic_time * 1000
    
    # Test WITH attacker (use play_att_with_belief for belief-based defense)
    if attack_models is not None and 'pessimistic' in attack_models:
        np.random.seed(seed + 1000)
        reward_test2_attack, pessimistic_attack_time = play_att_with_belief(
            environment2, agentQ2, trials=100, learn=False, measure_time=True,
            attack_model=attack_models['pessimistic']
        )
        results['pessimistic_attack_inference_time_ms'] = pessimistic_attack_time * 1000
        results['pessimistic_vs_attacker_reward'] = np.mean(reward_test2_attack)
        results['pessimistic_vs_attacker_std'] = np.std(reward_test2_attack)
    
    # ============================================================
    # PHASE 3: OFF-POLICY GREEDY TARGET
    # ============================================================
    print(f"[Exp {experiment_num + 1}] Phase 3: Training Off-Policy Greedy...")
    
    environment1 = GridWorld()
    environment1.seed(seed)
    agentQ1 = Q_Agent(environment1)
    agentQ1.q_table = copy.deepcopy(baseline_q_table)
    
    np.random.seed(seed + 1000)
    reward_train1 = play_offpolicy_greedy_target(environment1, agentQ1, trials=500, learn=True)
    results['offpolicy_greedy_train_avg_reward'] = np.mean(reward_train1)
    results['offpolicy_greedy_train_std'] = np.std(reward_train1)
    
    # Test WITHOUT attacker
    np.random.seed(seed + 1000)
    reward_test1, offpolicy_time = play_offpolicy_greedy_target(
        environment1, agentQ1, trials=100, learn=False, measure_time=True
    )
    results['offpolicy_greedy_test_avg_reward'] = np.mean(reward_test1)
    results['offpolicy_greedy_test_std'] = np.std(reward_test1)
    results['offpolicy_greedy_inference_time_ms'] = offpolicy_time * 1000
    
    # Test WITH attacker
    if attack_models is not None and 'offpolicy' in attack_models:
        np.random.seed(seed + 1000)
        reward_test1_attack, offpolicy_attack_time = play_offpolicy_greedy_target(
            environment1, agentQ1, trials=100, learn=False, measure_time=True,
            attack_model=attack_models['offpolicy']
        )
        results['offpolicy_attack_inference_time_ms'] = offpolicy_attack_time * 1000
        results['offpolicy_greedy_vs_attacker_reward'] = np.mean(reward_test1_attack)
        results['offpolicy_greedy_vs_attacker_std'] = np.std(reward_test1_attack)
    
    # ============================================================
    # PHASE 4: CALCULATE PERFORMANCE METRICS
    # ============================================================
    
    # Ensure all attack fields are populated
    for field in ['pessimistic_vs_attacker_reward', 'pessimistic_vs_attacker_std', 
                  'offpolicy_greedy_vs_attacker_reward', 'offpolicy_greedy_vs_attacker_std']:
        if field not in results:
            results[field] = np.nan
    
    # Performance differences 
    results['performance_diff_offpolicy_vs_pessimistic'] = results['offpolicy_greedy_test_avg_reward'] - results['pessimistic_test_avg_reward']
    results['performance_diff_offpolicy_vs_vanilla'] = results['offpolicy_greedy_test_avg_reward'] - results['vanilla_test_avg_reward']
    results['performance_diff_pessimistic_vs_vanilla'] = results['pessimistic_test_avg_reward'] - results['vanilla_test_avg_reward']
    
    # Speedup calculations
    if results['pessimistic_inference_time_ms'] > 0:
        results['speedup_offpolicy_vs_pessimistic'] = results['pessimistic_inference_time_ms'] / results['offpolicy_greedy_inference_time_ms']
    if results['vanilla_inference_time_ms'] > 0:
        results['speedup_offpolicy_vs_vanilla'] = results['vanilla_inference_time_ms'] / results['offpolicy_greedy_inference_time_ms']

    print(f"[Exp {experiment_num + 1}] Complete!")
    print(f"  Vanilla:     {results['vanilla_test_avg_reward']:.2f}")
    print(f"  Pessimistic: {results['pessimistic_test_avg_reward']:.2f}")  
    print(f"  Off-Policy:  {results['offpolicy_greedy_test_avg_reward']:.2f}")
    print(f"\n  Inference Times (ms):")
    print(f"    Vanilla:     {results['vanilla_inference_time_ms']:.4f}")
    print(f"    Pessimistic: {results['pessimistic_inference_time_ms']:.4f}")
    print(f"    Off-Policy:  {results['offpolicy_greedy_inference_time_ms']:.4f}")
    print(f"\n  Attack Inference Times (ms):")
    print(f"    Vanilla:     {results.get('vanilla_attack_inference_time_ms', 'N/A')}")
    print(f"    Pessimistic: {results.get('pessimistic_attack_inference_time_ms', 'N/A')}")
    print(f"    Off-Policy:  {results.get('offpolicy_attack_inference_time_ms', 'N/A')}")
    
    return results, agentQ1, agentQ2, agent_baseline

# ============================================================================
# EXECUTION SECTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("GRIDWORLD REINFORCEMENT LEARNING EXPERIMENTS")
    print("Comparing: Vanilla vs Pessimistic (old) vs Off-Policy Greedy")
    print("="*70)
    print("\nChoose an option:")
    print("  1. Quick test (5 experiments)")
    print("  2. Medium test (50 experiments)")
    print("  3. Full test (500 experiments)")
    print()
    print("  4. Train attackers (one-time setup)")
    print("  5. Analyze existing results")
    print("  6. Check progress")
    print("="*70)
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
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
           os.path.exists('attacker_pessimistic.zip'):
            use_attackers = input("Found trained attackers. Use for adversarial testing? (yes/no): ").lower()
            if use_attackers == 'yes':
                attack_models = {
                    'offpolicy': DQN.load('attacker_offpolicy'),
                    'pessimistic': DQN.load('attacker_pessimistic'),
                }
                print("✓ Loaded attackers for both algorithms")
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
                results, agent1, agent2, agent3 = run_single_experiment_with_seed(
                    experiment_num=exp_num,
                    seed=exp_num,
                    attack_models=attack_models
                )
                
                all_results.append(results)
                
                log_experiment_to_csv(exp_num, results, filename=csv_filename)
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
            
            if input("\nGenerate analysis? (yes/no): ").lower() == 'yes':
                analyze_csv_results(csv_filename)
    
    elif choice == '4':
        # ========== OPTION 4: Train Attackers ==========
        print("\n" + "="*70)
        print("TRAIN ATTACKERS")
        print("="*70)
        print("\nThis will train 2 separate attackers:")
        print("  1. Attacker vs Pessimistic Q-Learning")
        print("  2. Attacker vs Off-Policy Greedy")
        
        confirm = input("\nContinue? (yes/no): ").lower()
        
        if confirm == 'yes':
            print("\n" + "="*70)
            print("STEP 1: Training baseline agents to attack")
            print("="*70)
            
            ATTACKER_TRAINING_SEED = 99999
            
            print("\n[1/2] Training Pessimistic agent...")
            env2 = GridWorld()
            env2.seed(ATTACKER_TRAINING_SEED + 1)
            np.random.seed(ATTACKER_TRAINING_SEED + 1)
            agent2 = Q_Agent(env2)
            play(env2, agent2, trials=2000, learn=True)
            play_att_with_belief(env2, agent2, trials=500, learn=True)
            print("✓ Pessimistic agent trained")
            
            print("\n[2/2] Training Off-Policy agent...")
            env1 = GridWorld()
            env1.seed(ATTACKER_TRAINING_SEED)
            np.random.seed(ATTACKER_TRAINING_SEED)
            agent1 = Q_Agent(env1)
            play(env1, agent1, trials=2000, learn=True)
            play_offpolicy_greedy_target(env1, agent1, trials=500, learn=True)
            print("✓ Off-Policy agent trained")
            
            print("\n" + "="*70)
            print("STEP 2: Training attackers against each agent")
            print("="*70)
            
            print("\n[1/2] Training attacker against Pessimistic...")
            attack_pessimistic = train_RL_attack_with_belief(agent2, 'pessimistic', 'attacker')
            
            print("\n[2/2] Training attacker against Off-Policy...")
            attack_offpolicy = train_RL_attack_with_belief(agent1, 'offpolicy', 'attacker')
            
            print("\n" + "="*70)
            print("ALL ATTACKERS TRAINED SUCCESSFULLY!")
            print("="*70)
            print("\nSaved models:")
            print("  ✓ attacker_pessimistic.zip")
            print("  ✓ attacker_offpolicy.zip")
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
        # ========== OPTION 6: Check Progress ==========
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
        print("Invalid choice. Please run again and select 1-6.")
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)