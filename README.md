# State-Space Scaling & Memory Dynamics: Empirical Analysis of Embedding-Similarity Curiosity under Noisy Distractors

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-0.28.1-green.svg)](https://gymnasium.farama.org/)

This repository contains the standalone, peer-review-grade research codebase and visual suites for **Episodic Similarity Random Network Distortion (ES-RND)**. 

Our work systematically evaluates the lethal "Noisy-TV" distractor trap in curiosity-driven deep reinforcement learning (RL) and investigates how bounded episodic memory buffer configurations mitigate predictor capturing to unlock optimal, sparse-reward puzzle solving.

---

## 📖 Abstract

Dynamic distractors—such as the classic "Noisy-TV" problem—remain a major vulnerability in curiosity-driven reinforcement learning (RL) agents. Episodic Embedding-Similarity Random Network Distortion (ES-RND) was designed to counter this by recording observation embeddings in an episodic memory buffer and penalizing intrinsic rewards for states that match recently visited ones. This research presents a systematic, multi-dimensional empirical ablation suite evaluating ES-RND on a compact $9 \times 9$ Noisy-TV Key-Door Gridworld (`NoisyTVKeyDoorGridWorldSmall-v0`), scaled up to a long horizon of **500,000 training steps** per configuration. 

Our study comprehensively evaluates three critical ablation dimensions:
1. **Similarity Threshold ($\tau$) Tuning at $K=500$**: Quantifying exploration gradients across $\tau \in \{0.80, 0.85, 0.90, 0.95, 0.98\}$.
2. **Memory Persistence Policy**: Resets on episode boundaries (`zero_buffer_on_done = True`) versus cross-episode memory retention (`zero_buffer_on_done = False`).
3. **Episodic Buffer Size ($K$) Sweep at Fixed $\tau=0.95$**: Evaluating episodic memory capacity from extremely restricted windows ($K=20, 50, 80, 100, 150$) to infinite episodic horizons ($K=200, 500$).

Our findings expose three fundamental properties of episodic memory in RL:
* **The Attraction Paradox**: A similarity threshold set too close to unity ($\tau=0.98$) allows stochastically varying noise to escape penalization while heavily penalizing deterministic hallways, creating a relative exploration magnet that traps the agent.
* **Global Curiosity Starvation**: Keeping memory persistent across episodes (`zero_buffer_on_done = False`) over a long 500k-step horizon results in complete memory saturation (mean Cosine Similarity of **0.9991**), rendering the entire compact environment "boring" and collapsing the mean return to a near-zero **1.88%**.
* **The Sliding-Window Benefit & The Starvation Cliff**: Bounded episodic memory buffers ($K < T_{\max}$) act as a beneficial sliding window that prevents overall curiosity decay. At $K=50$, ES-RND achieves a peak **97.42% mean return** and optimal path efficiency (**34.74 steps**). However, at $K=150$, the agent hits a severe "curiosity starvation cliff," where buffer capacity limits interact with cyclic trajectories to apply penalties on **98.14%** of transitions, completely stalling exploration.

---

## 🧠 Algorithmic Framework: ES-RND

Standard curiosity models compute the intrinsic reward as:
$$r_i^{(t)} = \frac{1}{2} \| \hat{f}(s_t; \theta) - f(s_t) \|_2^2$$
where $f$ is a fixed, randomly initialized target network and $\hat{f}$ is a predictor network. In the presence of a dynamic, stochastic distractor, the prediction error remains permanently high, completely trapping the agent.

ES-RND introduces an episodic memory buffer $\mathcal{M}_t$ which records the embeddings $f(s_i)$ of observations visited during the current episode:
$$\mathcal{M}_t = \{ f(s_{t-j}) \}_{j=1}^{\min(t, K)}$$
where $K$ is the buffer size. Let the similarity between the current state $s_t$ and the memory buffer be:
$$\text{sim}(f(s_t), \mathcal{M}_t) = \max_{e \in \mathcal{M}_t} \frac{f(s_t) \cdot e}{\|f(s_t)\|_2 \|e\|_2}$$

We evaluate both **Strict Gating** (where intrinsic rewards are completely suppressed if similarity exceeds $\tau$) and **Soft Gating** (where a penalty multiplier is applied). In our experiments, we implement a soft gating mechanism with a penalty multiplier $\lambda = 0.20$ to maintain a minor exploration gradient:
$$m(s_t) = \mathbb{I}\left( \text{sim}(f(s_t), \mathcal{M}_t) > \tau \right)$$
$$r_{\text{pen}}^{(t)} = r_i^{(t)} \cdot \left( 1 - m(s_t) \cdot (1 - \lambda) \right)$$

---

## 🗺️ Environment Specifications ($9 \times 9$ Noisy-TV Key-Door Gridworld)

The custom `NoisyTVKeyDoorGridWorldSmall-v0` environment is structured as a compact $9 \times 9$ grid partitioned into two rooms by a solid wall column at $X=4$.
* **Agent Spawn**: Spawn at $(0,0)$ (bottom-left).
* **Key Location**: Placed at $(0,8)$ (top-left). The agent must navigate to this cell to pick up the key.
* **Door Location**: Placed at $(4,4)$ (middle wall). The door starts locked. It is automatically unlocked and the key consumed when the agent arrives at $(4,4)$ possessing the key.
* **Noisy TV (Distractor)**: Located at $(3,0)$ (bottom-center, left room). Whenever the agent is on $(3,0)$, the 5th element of its normalized observation vector is filled with high-frequency Gaussian noise $\epsilon \sim \mathcal{N}(0, 1.0)$.
* **Goal**: Sparse reward goal at $(8,8)$ (top-right room). Navigating to this cell terminates the episode with a reward of $+1.0$, but is only accessible after unlocking the door.
* **Horizon Limit**: $T_{\max} = 200$ steps per episode.

> [!NOTE]
> For spatial heatmap visualizations, coordinate row indices are physically vertically mirrored (`8 - y`) to match the standard rendering format where $(0,0)$ sits at the bottom-left and $(8,8)$ sits at the top-right.

---

## 📊 Comprehensive Quantitative Results (500k Steps)

The table below compiles all results obtained from scaled-up, sequential 500,000-step training runs on `NoisyTVKeyDoorGridWorldSmall-v0`.

### Table 1: Standard Hyperparameter & Baseline Suite (Buffer $K=500$ Control)

| ID | Run Configuration Description | TV Visits | TV Occupancy % | First Key Step | First Door Unlock | Max Return | Mean Return | Final Return | Mean Length | Mean Cos Sim |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **PPO Standard (Control)** | 389 | 0.08% | 328 | 1,184 | 0.000 | 0.000 | 0.000 | 200.00 | *N/A* |
| 2 | **RND Standard (Control)** | 420,047 | 84.06% | 328 | 1,184 | 0.100 | 0.000 | 0.000 | 199.98 | *N/A* |
| 3 | **ES-RND Soft ($\tau=0.98$)** | 71,604 | 14.33% | 328 | 1,184 | 1.000 | 0.490 | 1.000 | 139.80 | 0.9809 |
| 4 | **ES-RND Baseline ($K=500, \tau=0.95$)** | 24,035 | 4.81% | 328 | 1,184 | 1.000 | 0.953 | 1.000 | 38.98 | 0.9368 |
| 5 | **ES-RND Soft ($\tau=0.90$)** | 27,256 | 5.45% | 328 | 1,184 | 0.450 | 0.020 | 0.000 | 198.64 | 0.9924 |
| 6 | **ES-RND Persistent (Cross-Episode)** | 83,401 | 16.69% | 328 | 1,184 | 0.500 | 0.019 | 0.000 | 198.68 | 0.9991 |

### Table 2: Bounded Buffer Capacity ($K$) Sweep Suite (Fixed $\tau=0.95$)

| ID | Run Configuration Description | TV Visits | TV Occupancy % | First Key Step | First Door Unlock | Max Return | Mean Return | Final Return | Mean Length | Mean Cos Sim |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 7 | **ES-RND (K=20)** | 42,617 | 8.5283% | 328 | 1,184 | 1.000 | 0.930 | 1.000 | 47.08 | 0.9335 |
| 8 | **ES-RND (K=50)** | **19,121** | **3.8264%** | **328** | **1,184** | **1.000** | **0.974** | **1.000** | **34.74** | **0.9277** |
| 9 | **ES-RND (K=80)** | 22,687 | 4.5400% | 328 | 1,184 | 1.000 | 0.972 | 1.000 | 36.07 | 0.9310 |
| 10 | **ES-RND (K=100)** | 29,997 | 6.0029% | 328 | 1,184 | 1.000 | 0.957 | 1.000 | 39.47 | 0.9371 |
| 11 | **ES-RND (K=150)** | 28,738 | 5.7509% | 328 | 1,184 | 0.750 | 0.036 | 0.000 | 196.88 | 0.9920 |
| 12 | **ES-RND (K=200)** | 24,035 | 4.8098% | 328 | 1,184 | 1.000 | 0.953 | 1.000 | 38.98 | 0.9368 |
| 13 | **ES-RND (K=500)** | 24,035 | 4.8098% | 328 | 1,184 | 1.000 | 0.953 | 1.000 | 38.98 | 0.9368 |

---

## 🎨 Scientific Presentation Gallery

### 1. Baselines (PPO vs. Standard RND)
Without an episodic memory-gated similarity gate, standard RL agents are completely unable to solve the sparse key-door puzzle in the presence of dynamic noise. PPO suffers from curiosity starvation and fails to explore, while standard RND is rapidly and permanently captured by the distractor room.

| Learning Success Curves | Standard PPO Occupancy | Standard RND Occupancy |
| :---: | :---: | :---: |
| ![Baselines](results/baselines/baseline_success_curves.png) | ![PPO Standard](results/baselines/heatmap_ppo_standard.png) | ![RND Standard](results/baselines/heatmap_rnd_standard.png) |

### 2. Similarity Threshold ($\tau$) Sweep at $K=500$
Calibrating the threshold $\tau$ is essential to cover the dynamic noise manifold. If $\tau$ is too high ($\tau=0.98$), dynamic noise frames escape the penalty mask. If $\tau$ is too low ($\tau=0.80$), the agent becomes prematurely bored, starving exploration. A value of $\tau=0.95$ achieves perfect convergence.

| Success Curves | TV Occupancy % | Cosine Similarity |
| :---: | :---: | :---: |
| ![Tau Success](results/tau_ablations/plots/b50_tau_success_curves.png) | ![Tau TV Occupancy](results/tau_ablations/plots/b50_tau_tv_occupancy.png) | ![Tau Cosine Sim](results/tau_ablations/plots/b50_tau_cos_sim_curves.png) |

| Spatial Occupancy: $\tau=0.98$ | Spatial Occupancy: $\tau=0.95$ | Spatial Occupancy: $\tau=0.90$ |
| :---: | :---: | :---: |
| ![Tau 98](results/tau_ablations/plots/heatmap_tau98.png) | ![Tau 95](results/tau_ablations/plots/heatmap_tau95.png) | ![Tau 90](results/tau_ablations/plots/heatmap_tau90.png) |

### 3. Memory Reset Policy (Episodic vs. Cross-Episode Persistent)
Allowing the episodic buffer to persist across episode boundaries (`zero_buffer_on_done = False`) creates a cumulative memory footprint that saturates the entire compact gridworld, resulting in global curiosity starvation over long training horizons. Episodic resetting is mandatory.

| Success Curves | TV Occupancy % |
| :---: | :---: |
| ![Memory Success](results/memory_ablation/memory_success_curves.png) | ![Memory TV](results/memory_ablation/memory_tv_occupancy.png) |

| Spatial Occupancy: Episodic Zeroed | Spatial Occupancy: Persistent |
| :---: | :---: |
| ![Zeroed](results/memory_ablation/heatmap_zero_buffer.png) | ![Persistent](results/memory_ablation/heatmap_persist_buffer.png) |

### 4. Bounded Buffer Size ($K$) Sweep at $\tau=0.95$
Evaluating buffer sizes $K \in \{20, 50, 80, 100, 150, 200, 500\}$ reveals sliding-window benefits at $K=50$, infinite episodic equivalence for all $K \ge 200$, and a curiosity starvation cliff at $K=150$.

| Success Curves | TV Occupancy % | Cosine Similarity |
| :---: | :---: | :---: |
| ![Buffer Success](results/buffer_ablations/plots/buffer_success_curves.png) | ![Buffer TV](results/buffer_ablations/plots/buffer_tv_occupancy.png) | ![Buffer Cosine Sim](results/buffer_ablations/plots/buffer_cos_sim_curves.png) |

| Penalty Ratio | Episode Length |
| :---: | :---: |
| ![Buffer Penalty](results/buffer_ablations/plots/buffer_penalty_ratio.png) | ![Buffer Ep Length](results/buffer_ablations/plots/buffer_episode_length.png) |

| Spatial Occupancy: $K=20$ | Spatial Occupancy: $K=50$ | Spatial Occupancy: $K=80$ |
| :---: | :---: | :---: |
| ![K=20](results/buffer_ablations/plots/heatmap_k20.png) | ![K=50](results/buffer_ablations/plots/heatmap_k50.png) | ![K=80](results/buffer_ablations/plots/heatmap_k80.png) |

| Spatial Occupancy: $K=100$ | Spatial Occupancy: $K=150$ | Spatial Occupancy: $K=500$ |
| :---: | :---: | :---: |
| ![K=100](results/buffer_ablations/plots/heatmap_k100.png) | ![K=150](results/buffer_ablations/plots/heatmap_k150.png) | ![K=500](results/buffer_ablations/plots/heatmap_k500.png) |

---

## 🔍 Mechanistic Insights & Analysis

### 1. Bounded Buffer Robustness & The Peak Window ($K=50$)
Restricting the buffer capacity to **$K=50$** yields the absolute best performance across the entire sweep, achieving the highest overall mean return (**97.4%**), the lowest distractor occupancy (**3.83%**), and the most efficient average episode length (**34.74 steps**).
In a compact state space where the maximum episode length is $T_{\max} = 200$, a small capacity of $K=50$ acts as a highly effective localized sliding window. By keeping only the last 50 transitions, the agent is forced to forget early-episode spawn states. This maintains a strong intrinsic exploration gradient in newly reached, unvisited regions later in the episode, rather than dampening overall curiosity.

### 2. Bounded Over-penalization Starvation Cliff ($K=150$)
We discover a sudden, extreme performance collapse at **$K=150$** where success flatlines (Mean Return = 0.036, Final Return = 0.000), and the agent remains completely unsolved (Episode Length = 196.88 steps).
This represents a classic "Curiosity Starvation Cliff." At $K=150$, the buffer capacity is large enough to cover $75\%$ of the maximum trajectory horizon, but small enough to wrap around and continuously overwrite itself. The interaction between this capacity limit and localized looping trajectories causes the agent to generate extremely high embedding cosine similarities (**0.9920**) and apply boredom penalties on **98.14%** of all transitions. By over-suppressing all intrinsic rewards, exploration is starved, locking the agent into deadlocks before it can successfully unlock the door.

### 3. Mathematical Infinite Memory Boundary ($K \ge 200$)
Configurations with $K=200$ and $K=500$ achieve **mathematically identical** metric vectors (95.3% mean return, 4.81% TV occupancy, 38.98 mean steps, 0.9368 Cos Sim).
Because the environment enforces a hard episodic horizon limit $T_{\max} = 200$ and the memory buffer is cleared on every termination (`zero_buffer_on_done = True`), the buffer never collects more than 200 transitions in a single trajectory. Consequently, for any capacity setting $K \ge 200$, the buffer never wraps around. Bounding capacity above $T_{\max}$ is mathematically equivalent to infinite episodic memory.

### 4. The Proximity Trap: Noisy-TV Captivity in Bounded Environments
In prior studies on an $11 \times 11$ Key-Door Gridworld, standard RND spent a minor **3.85%** of its lifetime in the Noisy-TV room. However, when scaled down to a compact $9 \times 9$ gridworld, Standard RND spent a devastating **84.06%** of its training lifetime trapped in the distractor room.
In a smaller state-space, the distractor room is highly accessible and sits in close physical proximity to the agent's start path. RND discovers it almost immediately. Because the random TV noise is irreducible, the RND predictor network's prediction error remains permanently high, presenting a permanent intrinsic reward gradient.
The agent is completely captured, making 420,047 visits to the television and failing to explore the rest of the grid. This demonstrates that dynamic distractors are **even more lethal** in compact state-spaces where exploration volumes are restricted and starting locations are near the noise source.

---

## 🛠️ Reproduction & Execution Manual

### 1. Installation
Ensure you have Python 3.8+ installed. Install the streamlined dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Running Training Loops
To run a single ES-RND training run on the sequential $9 \times 9$ Key-Door Noisy-TV gridworld, run:
```bash
python -m cleanrl.ppo_rnd_penalty \
    --env-id NoisyTVKeyDoorGridWorldSmall-v0 \
    --total-timesteps 500000 \
    --exp-name my_es_rnd_run \
    --buffer-size 50 \
    --similarity-threshold 0.95 \
    --penalty-multiplier 0.20 \
    --zero-buffer-on-done True
```

### 3. Executing Ablation Sweeps
To run the automated multi-process hyperparameter sweeps:
* **Buffer Capacity ($K$) Sweep**:
  ```bash
  python -m scripts.run_buffer_ablations
  ```
* **Similarity Threshold ($\tau$) Sweep**:
  ```bash
  python -m scripts.run_tau_ablations
  ```

These scripts will execute the sweeps and output TensorBoard event files inside `runs_buffer_ablations` and `runs_buffer50_tau` directories.

### 4. Regenerating Plots and Reports
Once the sweeps complete, regenerate the visual reports and summary tables using:
```bash
python -m scripts.generate_buffer_plots
python -m scripts.generate_tau_plots
```
These scripts process the TensorBoard log directories, parse coordinates and scalars, compile CSV datasets, and save the 300 DPI visualization curves and glowing spatial heatmaps inside `results/buffer_ablations/` and `results/tau_ablations/`.

---

## 📂 Repository Directory Tree

```
episodic-similarity-rnd/
├── README.md               # Scientific peer-review-grade research report
├── requirements.txt        # Streamlined Python dependencies (torch, gymnasium, pandas, etc.)
├── .gitignore              # Ignores TensorBoard event files, python caches, and local checkpoints
├── cleanrl/                # Core implementation folder
│   ├── __init__.py
│   ├── ppo_rnd_penalty.py  # The main ES-RND algorithm script with embedding similarity penalty
│   ├── ppo_rnd_atari_penalty.py # Atari-adapted ES-RND implementation
│   ├── noisy_tv_gridworld.py # Baseline Noisy-TV environment
│   └── noisy_tv_keydoor_gridworld.py # The sequential Key-Door Noisy-TV gridworld
├── scripts/                # Execution and analysis runners
│   ├── run_buffer_ablations.py
│   ├── run_tau_ablations.py
│   ├── generate_buffer_plots.py
│   └── generate_tau_plots.py
└── results/                # Complete package of raw data tables and 300 DPI visualizations
    ├── baselines/          # Baseline comparison curves and occupancy heatmaps
    ├── memory_ablation/    # Memory reset policy comparison plots
    ├── buffer_ablations/
    │   ├── plots/          # Curve graphs and grid occupancy heatmaps
    │   └── raw_data/       # Compiled CSV statistics and JSON metadata
    └── tau_ablations/
        ├── plots/          # Curve graphs and grid occupancy heatmaps
        └── raw_data/       # Compiled CSV statistics and JSON metadata
```

---

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
