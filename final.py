import gym
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

def evaluate(environment, values, policy, discount, max_iters, convergence_tol):
    conv_track = []
    for it in range(max_iters):
        conv_track.append(np.linalg.norm(values, 2))
        next_values = np.zeros(environment.observation_space.n)
        for state in environment.P:
            outer_sum = 0
            for action in environment.P[state]:
                inner_sum = 0
                for prob, next_state, reward, terminal in environment.P[state][action]:
                    inner_sum += prob * (reward + discount * values[next_state])
                outer_sum += policy[state, action] * inner_sum
            next_values[state] = outer_sum
        if np.max(np.abs(next_values - values)) < convergence_tol:
            values = next_values
            break
        values = next_values
    return values

def improve(environment, values, num_actions, num_states, discount):
    import numpy as np
    q_matrix = np.zeros((num_states, num_actions))
    improved_policy = np.zeros((num_states, num_actions))
    
    for state_index in range(num_states):
        for action_index in range(num_actions):
            for prob, next_state, reward, terminal in environment.P[state_index][action_index]:
                q_matrix[state_index, action_index] += prob * (reward + discount * values[next_state])
        
        best_actions = np.where(q_matrix[state_index, :] == np.max(q_matrix[state_index, :]))

        improved_policy[state_index, best_actions] = 1 / np.size(best_actions)
    return improved_policy, q_matrix

def find_max(q_table, state):
    i = state // 4
    j = state % 4

    if j > 0:
        left = q_table[i][j-1]
    else:
        left = q_table[i][j]
        
    if i > 0:
        up = q_table[i-1][j]
    else:
        up = q_table[i][j]
        
    if j < 3:
        right = q_table[i][j+1]
    else:
        right = q_table[i][j]
    
    if i < 3:
        down = q_table[i+1][j]
    else:
        down = q_table[i][j]
    

    max_val = max([left, down, right, up])
    
    
    if q_table[i][j] == 1:
        if q_table[i][j-1] == 0:
            return 0
        elif q_table[i-1][j] == 0:
            return 3
        elif q_table[i][j+1] == 0:
            return 2
        elif q_table[i+1][j] == 0:
            return 1
    elif max_val == left:
        return 0
    elif max_val == down:
        return 1
    elif max_val == right:
        return 2
    elif max_val == up:
        return 3

environment = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human")

initial_values = np.zeros(environment.observation_space.n)
iterative_policy_eval_max_iters = 1000
iterative_policy_eval_conv_tol = 10**(-6)
discount_factor = 0.9
num_states = environment.observation_space.n
num_actions = environment.action_space.n
policy_iteration_max_iters = 1500

initial_policy = (1 / num_actions) * np.ones((num_states, num_actions))

for it in range(policy_iteration_max_iters):
    if it == 0:
        curr_policy = initial_policy
    computed_values = evaluate(environment, initial_values, curr_policy, discount_factor,
                                            iterative_policy_eval_max_iters,
                                            iterative_policy_eval_conv_tol)
    improved_policy, q_matrix = improve(environment, computed_values, num_actions, num_states,
                                                      discount_factor)
    if np.allclose(curr_policy, improved_policy):
        curr_policy = improved_policy
        break
    curr_policy = improved_policy

obs, info = environment.reset()
state = obs
step = 0
done = False

for step in range(100):
    action = find_max(computed_values.reshape(4, 4), state)

    new_state, reward, done, truncated, info = environment.step(action)

    if done:
        print("Steps taken:", step)
        break
    state = new_state


environment.close()
