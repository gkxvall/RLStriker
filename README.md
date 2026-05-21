# RLStriker

RLStriker is a 2D 1v1 soccer reinforcement learning environment built with Python, Pygame, PyTorch, pandas, and matplotlib.

The project is being built version by version. The current version is **V8**, which adds graph generation from saved training runs on top of the existing environment, rewards, logging, random agents, and DQN training pipeline.

![RLStriker Pygame preview](assets/preview.png)

## Current Features

- Pygame soccer pitch with two players, one ball, and two goals
- Manual keyboard control for both players
- Ball physics: friction, wall bounce, player collision, and kick action
- Episode rules: goal detection, score tracking, max-step termination
- Gym-style environment API: `reset()`, `step()`, `render()`, `close()`
- Headless simulation for faster training
- Random baseline agents
- Episode-level CSV logging and optional step-level CSV logging
- Reward shaping V1
- DQN agent with:
  - PyTorch Q-network
  - Replay buffer
  - Target network
  - Epsilon-greedy exploration
  - Checkpoint saving
- Training analysis scripts that generate PNG graphs from `episodes.csv`

## Project Status

| Version | Status | Summary |
| --- | --- | --- |
| V1 | Done | Basic Pygame soccer field |
| V2 | Done | Physics, kicking, goals, score, episode timer |
| V3 | Done | Gym-style environment |
| V4 | Done | Random agents |
| V5 | Done | Episode logging |
| V6 | Done | Simple reward shaping |
| V7 | Done | Basic DQN training pipeline |
| V8 | Done | Training dashboard graphs |

Next planned version: **V9 - Curriculum learning**.

## Installation

```bash
git clone https://github.com/gkxvall/RLStriker.git
cd RLStriker
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

## Manual Play

```bash
python main.py
```

Controls:

| Player | Move | Kick |
| --- | --- | --- |
| Blue / agent 1 | `WASD` or arrow keys | `Space` |
| Red / agent 2 | `I J K L` | `Enter` |

Reset the episode with `R` or the reset button.

## Random Self-Play

Run two random agents and save episode logs:

```bash
python run_random.py --episodes 100
```

Useful options:

```bash
python run_random.py --episodes 50 --run-name random_v6_baseline
python run_random.py --episodes 10 --render
python run_random.py --episodes 10 --log-steps
```

## DQN Training

Train a DQN agent against a random opponent:

```bash
python train.py --episodes 500
```

Train agent 2 instead of agent 1:

```bash
python train.py --episodes 500 --train-agent 2
```

Render training visually:

```bash
python train.py --episodes 20 --render
```

Use a custom run name:

```bash
python train.py --episodes 1000 --run-name dqn_agent1_v7
```

Useful training options:

| Option | Default | Description |
| --- | ---: | --- |
| `--episodes` | `500` | Number of episodes |
| `--train-agent` | `1` | Which side learns, `1` or `2` |
| `--checkpoint-every` | `100` | Save a model every N episodes |
| `--learning-rate` | `0.001` | Adam learning rate |
| `--gamma` | `0.99` | Discount factor |
| `--epsilon-start` | `1.0` | Initial exploration rate |
| `--epsilon-min` | `0.05` | Minimum exploration rate |
| `--epsilon-decay` | `0.995` | Episode-level epsilon decay |
| `--batch-size` | `64` | Replay batch size |
| `--buffer-size` | `50000` | Replay memory capacity |
| `--log-steps` | off | Save optional `steps.csv` |
| `--render` | off | Open the Pygame window |

## Training Graphs

Generate all V8 graphs for a saved run:

```bash
python -m analysis.plot_all --run-dir data/training_runs/dqn_agent1_v7
```

By default, plots are saved to:

```text
data/training_runs/<run_name>/plots/
```

Generated files:

```text
plots/
├── rewards.png
├── winrate.png
├── goals.png
├── touches.png
├── distance_to_ball.png
├── episode_length.png
└── epsilon.png
```

You can run individual plot scripts too:

```bash
python -m analysis.plot_rewards --run-dir data/training_runs/dqn_agent1_v7
python -m analysis.plot_goals --run-dir data/training_runs/dqn_agent1_v7
python -m analysis.plot_winrate --run-dir data/training_runs/dqn_agent1_v7
python -m analysis.plot_touches --run-dir data/training_runs/dqn_agent1_v7
```

Useful graph options:

| Option | Default | Description |
| --- | ---: | --- |
| `--run-dir` | required | Folder containing `episodes.csv` |
| `--output-dir` | `<run-dir>/plots` | Where PNG files are saved |
| `--window` | `100` | Rolling average window |
| `--show` | off | Open an interactive matplotlib window |

## First Training Run Visualization

These charts come from the first DQN training run saved as `agent1`. They are included in `assets/plots/agent1/` so they render directly on GitHub.

### Rewards

![Agent 1 reward training plot](assets/plots/agent1/rewards.png)

### Win Rate

![Agent 1 win rate training plot](assets/plots/agent1/winrate.png)

### Goals

![Agent 1 goals training plot](assets/plots/agent1/goals.png)

### Ball Touches

![Agent 1 ball touches training plot](assets/plots/agent1/touches.png)

### Distance to Ball

![Agent 1 average distance to ball training plot](assets/plots/agent1/distance_to_ball.png)

### Episode Length

![Agent 1 episode length training plot](assets/plots/agent1/episode_length.png)

### Epsilon Decay

![Agent 1 epsilon decay training plot](assets/plots/agent1/epsilon.png)

## Output Data

Every logged run creates a folder under:

```text
data/training_runs/<run_name>/
```

Example V8 training output:

```text
data/training_runs/run_YYYYMMDD_HHMMSS/
├── config.json
├── episodes.csv
├── steps.csv              # optional, only with --log-steps
├── checkpoints/
│   ├── episode_000100.pt
│   ├── episode_000200.pt
│   └── final.pt
└── plots/
    ├── rewards.png
    ├── winrate.png
    ├── goals.png
    ├── touches.png
    ├── distance_to_ball.png
    ├── episode_length.png
    └── epsilon.png
```

`episodes.csv` includes episode length, winner, scores, goals, touches, kicks, steals, average distance to ball, total rewards, epsilon, and timestamp.

## Reward Shaping V1

The current reward function is intentionally simple:

| Event | Reward |
| --- | ---: |
| Score a goal | `+100` |
| Concede a goal | `-100` |
| Touch the ball | `+2` |
| Ball moves toward opponent goal | `+0.2` |
| Every step | `-0.01` |

Team 1 attacks the right goal. Team 2 attacks the left goal.

## Environment API

```python
from env.soccer_env import ACTION_SPACE_SIZE, SoccerEnv
from agents.random_agent import RandomAgent

env = SoccerEnv(render_mode=None)
agent_1 = RandomAgent()
agent_2 = RandomAgent()

state = env.reset()
done = False

while not done:
    action_1 = agent_1.act()
    action_2 = agent_2.act()
    state, reward_1, reward_2, done, info = env.step(action_1, action_2)

env.close()
```

Actions:

| ID | Action |
| ---: | --- |
| `0` | Stay |
| `1` | Up |
| `2` | Down |
| `3` | Left |
| `4` | Right |
| `5` | Kick |

## Repository Structure

```text
RLStriker/
├── agents/
│   ├── dqn_agent.py
│   ├── model.py
│   ├── random_agent.py
│   └── replay_buffer.py
├── env/
│   ├── constants.py
│   ├── entities.py
│   ├── field.py
│   ├── physics.py
│   ├── rewards.py
│   ├── rules.py
│   ├── soccer_env.py
│   └── state.py
├── logging_utils/
│   ├── episode_logger.py
│   └── metrics.py
├── analysis/
│   ├── plot_all.py
│   ├── plot_goals.py
│   ├── plot_rewards.py
│   ├── plot_touches.py
│   └── plot_winrate.py
├── data/
│   └── training_runs/
├── main.py
├── run_random.py
├── train.py
├── requirements.txt
└── README.md
```

## Development Roadmap

- V9: Curriculum learning
- V10: Better state representation
- V11: Self-play against older checkpoints
- V12: More detailed soccer rewards and reward components
- V13: Demo mode for trained models
- V14: Human vs AI mode
- V15: Multi-agent 2v2 expansion
- V16: Final portfolio polish

## License

MIT. See [LICENSE](LICENSE).
