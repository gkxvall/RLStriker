# RLStriker

RLStriker is a 2D 1v1 soccer reinforcement learning environment built with Python and Pygame. It supports real-time visualization, fast headless simulation, and structured training run logs for later analysis.

**Current release: V5**

## Features

- Playable pitch with two players, one ball, and goals
- Physics: ball friction, player–ball collision, kick
- Rules: goal detection, cumulative score, episode limit (`MAX_STEPS` in `env/constants.py`)
- Gym-style API: `SoccerEnv` with `reset`, `step`, `render`, and `close`
- Random baseline agents and an automated runner
- **Episode logging**: each training run writes `config.json`, `episodes.csv`, and optional `steps.csv` under `data/training_runs/`

## Requirements

- Python 3.10+
- Pygame 2.5+ (see `requirements.txt`)

## Installation

```bash
git clone https://github.com/gkxvall/RLStriker.git
cd RLStriker
pip install -r requirements.txt
```

## Usage

### Manual play (window)

```bash
python main.py
```

| Team | Move | Kick |
|------|------|------|
| Blue (agent 1) | `WASD` or arrow keys | `Space` |
| Red (agent 2) | `I` `J` `K` `L` | `Enter` |

Use `R` or the **Reset** button to start a new episode.

### Random self-play (with logging)

Each run creates a folder:

`data/training_runs/run_YYYYMMDD_HHMMSS/`

(or a custom name via `--run-name`).

```bash
python run_random.py --episodes 100
```

| Flag | Description |
|------|-------------|
| `--run-name NAME` | Folder name under `data/training_runs/` |
| `--log-steps` | Also write `steps.csv` (large files over long runs) |
| `--epsilon FLOAT` | Value stored in `episodes.csv` (default `1.0` for random agents) |
| `--render` | Show the match in a window |
| `--no-log` | Skip writing log files |

Example with optional step log:

```bash
python run_random.py --episodes 50 --run-name my_baseline --log-steps
```

### Logged files

| File | Description |
|------|-------------|
| `config.json` | Run settings (episodes, `max_steps`, `epsilon`, flags, etc.) |
| `episodes.csv` | One row per episode with metrics (winner, rewards, touches, kicks, distances, …) |
| `steps.csv` | Optional per-step state and actions (only with `--log-steps`) |

## Environment API

`SoccerEnv` lives in `env/soccer_env.py`.

- `reset() -> state`
- `step(action_1, action_2) -> state, reward_1, reward_2, done, info`
- `render()` / `close()`

Discrete actions: `0` stay, `1` up, `2` down, `3` left, `4` right, `5` kick.

Headless loop example:

```python
import random
from env.soccer_env import SoccerEnv, ACTION_SPACE_SIZE

env = SoccerEnv(render_mode=None)
env.reset()
for _ in range(5000):
    a1 = random.randrange(ACTION_SPACE_SIZE)
    a2 = random.randrange(ACTION_SPACE_SIZE)
    _, _, _, done, _ = env.step(a1, a2)
    if done:
        env.reset()
env.close()
```

## Repository layout

```text
RLStriker/
├── agents/
│   ├── __init__.py
│   └── random_agent.py
├── env/
│   ├── constants.py
│   ├── entities.py
│   ├── field.py
│   ├── physics.py
│   ├── rules.py
│   ├── soccer_env.py
│   └── state.py
├── logging_utils/
│   ├── __init__.py
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

Planned next steps: richer reward shaping, DQN training, plotting scripts, curriculum and self-play, demo and human-vs-AI modes.

## License

MIT — see [LICENSE](LICENSE).
