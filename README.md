# RLStriker

RLStriker is a 2D 1v1 soccer reinforcement learning environment built with Python and Pygame. It supports real-time visualization, fast headless simulation, shaped rewards for training, and structured run logs under `data/training_runs/`.

**Current release: V6**

## Features

- Playable pitch with two players, one ball, and goals
- Physics: ball friction, player–ball collision, kick
- Rules: goal detection, cumulative score, episode step limit
- Gym-style API: `SoccerEnv` (`reset`, `step`, `render`, `close`)
- **Reward shaping V1** (see below)
- Random baseline agents and automated self-play with CSV logging

## Reward shaping (V6)
Each environment step applies these rewards (see `env/rewards.py`):
| Signal | Agent 1 | Agent 2 |
|--------|---------|---------|
| Score a goal | +100 | — |
| Concede a goal | −100 | — |
| Touch the ball | +2 | +2 |
| Ball moves toward **opponent** goal | +0.2 | +0.2 |
| Time (every step) | −0.01 | −0.01 |
Team 1 attacks the **right** goal; team 2 attacks the **left** goal. Episode totals are summed into `total_reward_agent_1` and `total_reward_agent_2` in `episodes.csv`.

## Requirements

- Python 3.10+
- Pygame 2.5+ (`requirements.txt`)

## Installation

```bash
git clone https://github.com/gkxvall/RLStriker.git
cd RLStriker
pip install -r requirements.txt
```

## Usage

### Manual play

```bash
python main.py
```

| Team | Move | Kick |
|------|------|------|
| Blue (agent 1) | `WASD` / arrow keys | `Space` |
| Red (agent 2) | `I` `J` `K` `L` | `Enter` |

Reset episode: `R` or **Reset** button.

### Random self-play + logging

```bash
python run_random.py --episodes 100
```

| Flag | Description |
|------|-------------|
| `--run-name NAME` | Output folder under `data/training_runs/` |
| `--log-steps` | Write optional `steps.csv` (per-step rewards included) |
| `--epsilon FLOAT` | Logged in `episodes.csv` (exploration rate for future DQN runs) |
| `--render` | Open Pygame window |
| `--no-log` | Disable file output |

Example:

```bash
python run_random.py --episodes 50 --run-name random_v6_baseline
```

### Training run output

```
data/training_runs/<run_name>/
├── config.json      # includes reward_version: "v1"
├── episodes.csv     # total_reward_agent_1 / total_reward_agent_2 per episode
└── steps.csv        # optional (--log-steps)
```

## Environment API

```python
import random
from env.soccer_env import SoccerEnv, ACTION_SPACE_SIZE

env = SoccerEnv(render_mode=None)
env.reset()

total_r1 = total_r2 = 0.0
for _ in range(5000):
    a1 = random.randrange(ACTION_SPACE_SIZE)
    a2 = random.randrange(ACTION_SPACE_SIZE)
    _, r1, r2, done, info = env.step(a1, a2)
    total_r1 += r1
    total_r2 += r2
    if done:
        env.reset()

env.close()
```

Actions: `0` stay, `1` up, `2` down, `3` left, `4` right, `5` kick.

## Project structure

```text
RLStriker/
├── agents/
│   └── random_agent.py
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
├── data/
│   └── training_runs/     # created at runtime (gitignored)
├── main.py
├── run_random.py
├── requirements.txt
└── README.md
```

## Roadmap

Next: DQN agent (V7), training plots (V8), curriculum and self-play, demo and human-vs-AI modes.

## License

MIT — see [LICENSE](LICENSE).
