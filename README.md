# RLStriker

A Pygame 1v1 soccer environment where two DQN agents can eventually play, receive rewards, train over many episodes, and improve from random movement to ball-chasing and basic scoring.

**Current version: V2** — physics, goals, and episodes.

## What's implemented

| Version | Status | Features |
|---------|--------|----------|
| V1 | Done | Pygame pitch, two players, ball, goals, manual control, reset |
| V2 | Done | Ball physics, friction, collisions, kick, goal detection, score, episode timer |
| V3+ | Planned | Gym-style env, agents, DQN training, logging, demos |

## Requirements

- Python 3.10+
- [Pygame](https://www.pygame.org/) 2.5+

## Install

```bash
git clone https://github.com/gkxvall/RLStriker.git
cd RLStriker
pip install -r requirements.txt
```

## Run (V2)

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

## Project layout

```
RLStriker/
├── main.py              # Playable game loop
├── requirements.txt
├── env/
│   ├── constants.py     # Sizes, speeds, MAX_STEPS
│   ├── entities.py      # Player, Ball
│   ├── field.py         # Pitch and goals drawing
│   ├── physics.py       # Movement, friction, collision, kick
│   └── rules.py         # Goals, scoring, episode state
└── README.md
```

## Verify V2 works

1. Run `python main.py` — window opens with score and step counter.
2. Move into the ball — it should roll with friction and bounce off pitch edges.
3. Press kick (Space / Enter) near the ball — it should shoot in your movement direction.
4. Score in the opponent's goal — overlay shows "GOAL!", episode stops advancing.
5. Let the step counter reach 1000 without a goal — episode ends with max-steps message.
6. Press **R** — new kickoff positions, play continues.

## Roadmap

Next up: **V3 — Gym-style `SoccerEnv`** (`reset`, `step`, headless simulation without rendering).

## License

MIT — see [LICENSE](LICENSE).
