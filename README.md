# RLStriker

RLStriker is a 2D 1v1 soccer reinforcement learning environment built with Python and Pygame.
It supports real-time visualization and fast headless simulation for training loops.

Current implementation level: **V4**.

## Features (V1-V4)

- Playable soccer field with two players, one ball, and visible goals
- Basic physics: ball friction, player-ball collision, kick action
- Match rules: goal detection, score tracking, max-step episode ending
- Gym-style environment API via `SoccerEnv`
- Random-vs-random baseline runner for automated episodes

## Installation

```bash
git clone https://github.com/gkxvall/RLStriker.git
cd RLStriker
pip install -r requirements.txt
```

## Quick Start

### 1) Manual match (rendered)

```bash
python main.py
```

Controls:

- Team 1 (blue): `WASD` or arrow keys, kick with `Space`
- Team 2 (red): `I J K L`, kick with `Enter`
- Reset current episode: `R` or click `Reset`

### 2) Automated random self-play (headless by default)

```bash
python run_random.py --episodes 100
```

Optional rendered random simulation:

```bash
python run_random.py --episodes 20 --render
```

## Environment API

`env/soccer_env.py` provides a Gym-style interface:

- `reset() -> state`
- `step(action_1, action_2) -> state, reward_1, reward_2, done, info`
- `render()`
- `close()`

Action space:

- `0`: stay
- `1`: up
- `2`: down
- `3`: left
- `4`: right
- `5`: kick

Headless example:

```python
import random
from env.soccer_env import SoccerEnv, ACTION_SPACE_SIZE

env = SoccerEnv(render_mode=None)
state = env.reset()

for _ in range(2000):
    a1 = random.randrange(ACTION_SPACE_SIZE)
    a2 = random.randrange(ACTION_SPACE_SIZE)
    state, r1, r2, done, info = env.step(a1, a2)
    if done:
        state = env.reset()

env.close()
```

## Repository Structure

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
├── main.py
├── run_random.py
├── requirements.txt
└── README.md
```

## Verification Checklist

- `python main.py` opens a playable match window
- Ball movement, collisions, kick, and goal detection work
- Episodes end after a goal or max steps
- `python run_random.py --episodes 100` completes automatically and prints episode summaries

## Roadmap

Upcoming milestones include episode logging, DQN training, analytics scripts, curriculum learning, self-play improvements, and demo/manual portfolio modes.

## License

MIT License - see [LICENSE](LICENSE).
