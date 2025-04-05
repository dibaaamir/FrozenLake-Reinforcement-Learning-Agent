# 🧊 FrozenLake Solver using Policy Iteration (OpenAI Gym)

This project implements a custom **policy iteration algorithm** to solve the classic **FrozenLake-v1** environment from OpenAI Gym. The goal is to teach an agent to reach the goal tile safely without falling into holes using iterative **policy evaluation** and **policy improvement**.

---

## 🌍 Environment Overview

**FrozenLake-v1** is a grid-based environment where:
- The agent starts on a frozen tile (`S`) and must reach the goal (`G`)
- Some tiles are holes (`H`) where the agent falls and loses
- The rest are frozen (`F`) and safe to walk on
- The agent can take 4 actions: `Left`, `Down`, `Right`, `Up`
- The environment is deterministic (`is_slippery=False`)

---

## 🔁 What This Project Does

- Implements **Iterative Policy Evaluation** to estimate the value of each state under a policy
- Implements **Policy Improvement** using a greedy strategy based on estimated values
- Updates policy until convergence (no changes across iterations)
- Simulates a run using the learned policy to reach the goal
- Uses **OpenAI Gym** for the environment and **NumPy** for math

---

## 🛠 Implementation Details

### ✅ Main Components

- `evaluate()`  
  Iteratively computes the value of each state under the current policy using Bellman expectations.

- `improve()`  
  Calculates Q-values for all actions in each state and updates the policy by selecting optimal actions.

- `find_max()`  
  Used during the simulation run to determine which action to take based on value estimates.

- Policy iteration stops when the policy becomes stable (no further improvement).

---

## ⚙️ Hyperparameters

| Parameter                  | Value        |
|---------------------------|--------------|
| Discount Factor (γ)       | 0.9          |
| Max Policy Iteration Steps| 1500         |
| Policy Evaluation Tolerance | 1e-6        |
| Evaluation Max Iters      | 1000         |

---

## 🚀 How to Run

1. **Install dependencies**:
   ```bash
   pip install gym matplotlib numpy
2. **Run the script or notebook** containing the policy iteration code.
3. **Watch the agent** take actions in the environment after training:
   - The environment is rendered in `human` mode.
   - The agent follows the learned policy step by step.

---

## 🧪 Sample Output

After training, you'll see output like:
     Steps taken: 6 

This means the agent successfully reached the goal in 6 steps using the learned optimal policy.

---

## 📦 Dependencies

- `gym` – For the FrozenLake environment
- `numpy` – For matrix operations and value calculations
- `matplotlib` – *(Optional)* For visualizations or debugging
- `warnings` – To suppress deprecation notices

You can install the dependencies with:

```bash
        pip install gym numpy matplotlib
## 📚 Key Concepts Covered

- **Reinforcement Learning (RL)**
- **Policy Iteration Algorithm**
- **Markov Decision Processes (MDPs)**
- **Dynamic Programming in RL**
- **OpenAI Gym Integration**
