# RLStriker

A Pygame 1v1 soccer environment where two DQN agents can eventually play, receive rewards, train over many episodes, and improve from random movement to ball-chasing and basic scoring.

**Current version: V3** — Gym-style environment with headless stepping.

## What's implemented

| Version | Status | Features |
|---------|--------|----------|
| V1 | Done | Pygame pitch, two players, ball, goals, manual control, reset |
| V2 | Done | Ball physics, friction, collisions, kick, goal detection, score, episode timer |
| V3 | Done | `SoccerEnv` with `reset`, `step`, `render`, state/reward/done/info returns |
| V4+ | Planned | Random agents, logging, DQN training, analysis, demo/manual modes |

## Requirements

- Python 3.10+
- [Pygame](https://www.pygame.org/) 2.5+

## Install

```bash
git clone https://github.com/gkxvall/RLStriker.git
cd RLStriker
pip install -r requirements.txt
```

## Run (V3 manual play)

```bash
python main.py
```

### Controls

| Player | Move | Kick |
|--------|------|------|
| Team 1 (blue) | WASD or arrow keys | Space |
| Team 2 (red) | I J K L | Enter |

- **Reset** button or **R** — start a new episode (positions and ball reset; match score is kept across episodes until you restart the app).
- Team 1 attacks the **right** goal; team 2 attacks the **left** goal.

### Episode rules

- An episode ends when a **goal** is scored or after **1000 steps** (`MAX_STEPS` in `env/constants.py`).
- When the episode ends, an overlay appears; press **R** or **Reset** to play again.

## Gym-style environment (V3)

The environment API is:

- `reset() -> state`
- `step(action_1, action_2) -> state, reward_1, reward_2, done, info`
- `render()`

Discrete actions:

- `0`: stay
- `1`: up
- `2`: down
- `3`: left
- `4`: right
- `5`: kick

Minimal headless example (no rendering):

```python
from env.soccer_env import SoccerEnv, ACTION_SPACE_SIZE
import random

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

## Project layout

```
RLStriker/
├── main.py              # Manual game loop using SoccerEnv
├── requirements.txt
├── env/
│   ├── constants.py     # Sizes, speeds, MAX_STEPS
│   ├── entities.py      # Player, Ball
│   ├── field.py         # Pitch and goals drawing
│   ├── physics.py       # Movement, friction, collision, kick
│   ├── rules.py         # Goals, scoring, episode state
│   ├── state.py         # State builder for RL loops
│   └── soccer_env.py    # Gym-style environment API
└── README.md
```

## Verify V3 works

1. Run `python main.py` — window opens with score and step counter.
2. Play manually: movement, collisions, goals, and episode ending should match V2 behavior.
3. Run the headless example above; it should execute thousands of steps without opening a window.
4. Confirm each `step()` call returns `(state, reward_1, reward_2, done, info)`.

## Roadmap

Next up: **V4 — Random Agents** and an auto-run loop script (`run_random.py`).

## License

MIT — see [LICENSE](LICENSE).
