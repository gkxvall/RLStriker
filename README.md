# RLStriker

RLStriker is a 2D soccer reinforcement learning project built with Python, Pygame, PyTorch, pandas, and matplotlib.

The project is complete through **V16**, the final portfolio polish pass. It includes a playable Pygame soccer environment, Gym-style headless simulation, DQN training, checkpoint self-play, curriculum learning, reward-component logging, graph generation, demo mode, human-vs-AI mode, and a first 2v2 team environment.

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
- Reward shaping V2 with component-level debugging
- DQN agent with:
  - PyTorch Q-network
  - Replay buffer
  - Target network
  - Epsilon-greedy exploration
  - Checkpoint saving
- Training analysis scripts that generate PNG graphs from `episodes.csv`
- Curriculum learning stages for easier DQN skill progression
- V10 state vector with distance, angle, goal-distance, ball-direction, and last-touch features
- V11 self-play against older checkpoint opponents and random baselines
- V12 reward components for goals, touches, progress, steals, useful kicks, positioning, energy, own-goal pushes, and unnecessary kicks
- V13 demo mode with score, episode, reward, event, model, epsilon, and FPS overlay
- V14 human-vs-AI mode with random, weak, strong, and latest AI difficulties
- V15 2v2 team soccer environment with role baselines, passing reward, spacing reward, and team rewards
- V16 portfolio-ready README, architecture notes, run instructions, demo guidance, and project explanation

## Project Status

| Version | Status | Summary                                       |
| ------- | ------ | --------------------------------------------- |
| V1      | Done   | Basic Pygame soccer field                     |
| V2      | Done   | Physics, kicking, goals, score, episode timer |
| V3      | Done   | Gym-style environment                         |
| V4      | Done   | Random agents                                 |
| V5      | Done   | Episode logging                               |
| V6      | Done   | Simple reward shaping                         |
| V7      | Done   | Basic DQN training pipeline                   |
| V8      | Done   | Training dashboard graphs                     |
| V9      | Done   | Curriculum learning                           |
| V10     | Done   | Better state representation                   |
| V11     | Done   | Checkpoint self-play                          |
| V12     | Done   | Better rewards and reward components          |
| V13     | Done   | Demo mode for trained models                  |
| V14     | Done   | Human vs AI mode                              |
| V15     | Done   | Multi-agent 2v2 expansion                     |
| V16     | Done   | Final portfolio polish                        |

RLStriker is now in a portfolio-ready state. Future work can focus on stronger 2v2 learning, richer evaluation, and longer training runs.

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

## Quick Start

| Goal | Command |
| ---- | ------- |
| Play the 1v1 sandbox | `python main.py` |
| Run random 1v1 agents | `python run_random.py --episodes 100` |
| Train a DQN learner | `python train.py --episodes 500` |
| Train with curriculum | `python train.py --episodes 1000 --curriculum` |
| Train with checkpoint self-play | `python self_play.py --episodes 1000` |
| Watch a trained checkpoint | `python demo.py --checkpoint data/training_runs/newAgent/checkpoints/final.pt` |
| Play against an AI | `python manual.py` |
| Run 2v2 baseline agents | `python run_team_random.py --episodes 10` |
| Generate training graphs | `python -m analysis.plot_all --run-dir data/training_runs/<run_name>` |

## Manual Play

```bash
python main.py
```

Controls:

| Player         | Move                 | Kick    |
| -------------- | -------------------- | ------- |
| Blue / agent 1 | `WASD` or arrow keys | `Space` |
| Red / agent 2  | `I J K L`            | `Enter` |

Reset the episode with `R` or the reset button.

## Human vs AI

Play as one player against an AI opponent:

```bash
python manual.py
```

The game opens a difficulty menu with:

| Difficulty | Behavior |
| ---------- | -------- |
| Random AI | Baseline random-action opponent |
| Weak trained model | Uses the `agent2` checkpoint when available |
| Strong trained model | Uses the `alpha` curriculum checkpoint when available |
| Latest model | Uses the newest compatible non-smoke checkpoint |

Skip the menu and start directly:

```bash
python manual.py --difficulty strong
```

Play as red / agent 2:

```bash
python manual.py --human-agent 2 --difficulty latest
```

Override the checkpoint used by trained difficulties:

```bash
python manual.py --difficulty strong --checkpoint data/training_runs/alpha/checkpoints/final.pt
```

Controls: move with `WASD` or arrow keys, kick with `Space`, restart with `R`, reopen the difficulty menu with `M`, and quit with `Esc`.

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

## 2v2 Team Mode

Run the 2v2 team environment with baseline team agents:

```bash
python run_team_random.py --episodes 10
```

Render the 2v2 match:

```bash
python run_team_random.py --episodes 3 --render
```

Try the simple role baseline against random team actions:

```bash
python run_team_random.py --team-1 role --team-2 random
```

V15 introduces `TeamSoccerEnv`, where each team has two players. Rewards are team-level and include:

| Signal | Purpose |
| ------ | ------- |
| Team goal / concede | Shared team scoring reward |
| Ball progress | Reward moving the ball toward the opponent goal |
| Touch reward | Small reward when a team controls the ball |
| Passing reward | Reward when teammates exchange possession while the ball is moving |
| Spacing reward | Reward teammates for not collapsing onto the same position |

This is the first 2v2 foundation. It does not replace the 1v1 DQN/self-play stack; it gives the next versions a clean team environment to build on.

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

Train with curriculum learning:

```bash
python train.py --episodes 1000 --curriculum --run-name curriculum_agent1_v9
```

Use custom episode counts for the five curriculum stages:

```bash
python train.py --episodes 1000 --curriculum --curriculum-stage-episodes 100,150,200,250,300
```

Train with checkpoint self-play:

```bash
python self_play.py --episodes 1000 --run-name self_play_v11
```

Seed self-play with an existing compatible checkpoint:

```bash
python self_play.py --episodes 1000 --initial-opponent data/training_runs/agent2/checkpoints/final.pt
```

Useful training options:

| Option                        |      Default | Description                                   |
| ----------------------------- | -----------: | --------------------------------------------- |
| `--episodes`                  |        `500` | Number of episodes                            |
| `--train-agent`               |          `1` | Which side learns, `1` or `2`                 |
| `--checkpoint-every`          |        `100` | Save a model every N episodes                 |
| `--learning-rate`             |      `0.001` | Adam learning rate                            |
| `--gamma`                     |       `0.99` | Discount factor                               |
| `--epsilon-start`             |        `1.0` | Initial exploration rate                      |
| `--epsilon-min`               |       `0.05` | Minimum exploration rate                      |
| `--epsilon-decay`             |      `0.995` | Episode-level epsilon decay                   |
| `--batch-size`                |         `64` | Replay batch size                             |
| `--buffer-size`               |      `50000` | Replay memory capacity                        |
| `--curriculum`                |          off | Enable curriculum learning                    |
| `--curriculum-stage-episodes` | split evenly | Episode counts for the five curriculum stages |
| `--log-steps`                 |          off | Save optional `steps.csv`                     |
| `--render`                    |          off | Open the Pygame window                        |

## Demo Mode

Watch a trained checkpoint play in the Pygame window:

```bash
python demo.py --checkpoint data/training_runs/newAgent/checkpoints/final.pt
```

By default, the trained model controls agent 1 against a random opponent. You can put the model on the red side:

```bash
python demo.py --checkpoint data/training_runs/newAgent/checkpoints/final.pt --model-agent 2
```

Watch a checkpoint-vs-checkpoint match:

```bash
python demo.py \
  --checkpoint data/training_runs/newAgent/checkpoints/final.pt \
  --opponent checkpoint \
  --opponent-checkpoint data/training_runs/agent2/checkpoints/final.pt
```

The demo overlay shows score, episode, step, cumulative rewards, last event, model name, saved epsilon, and FPS. Press `R` to reset the current episode or `Esc` to quit.

Useful demo options:

| Option                  |        Default | Description                             |
| ----------------------- | -------------: | --------------------------------------- |
| `--checkpoint`          | newAgent final | Model checkpoint to watch               |
| `--model-agent`         |            `1` | Side controlled by the model            |
| `--opponent`            |         random | Use `random` or `checkpoint` opponent   |
| `--opponent-checkpoint` |           none | Checkpoint path for checkpoint opponent |
| `--episodes`            |          `100` | Number of demo episodes before exiting  |
| `--fps`                 |           `60` | Playback speed                          |

## Self-Play

V11 adds checkpoint-based self-play. The current learner periodically saves model snapshots, stores them in an opponent pool, and trains against a mix of:

- Random opponents
- Older snapshots from the current run
- Optional checkpoint opponents passed with `--initial-opponent`

This helps avoid training only against weak random behavior and starts pushing the agent toward policies that can beat earlier versions of itself.

Useful self-play options:

| Option                     | Default | Description                                                   |
| -------------------------- | ------: | ------------------------------------------------------------- |
| `--checkpoint-every`       |   `100` | Save regular learner checkpoints                              |
| `--opponent-refresh-every` |   `250` | Add the current learner to the opponent pool every N episodes |
| `--opponent-pool-size`     |     `8` | Maximum number of older checkpoint opponents to keep          |
| `--random-opponent-prob`   |  `0.25` | Chance of using a random opponent instead of a checkpoint     |
| `--initial-opponent`       |    none | Seed the pool with one or more compatible checkpoint files    |

Self-play checkpoints require the same state size as the current environment. V10+ runs use an 18-value state vector, so older 9-value checkpoints are skipped automatically.

## Curriculum Learning

Curriculum learning adds an optional staged training path. The goal is to teach the agent small soccer skills before asking it to survive a full match.

| Stage | Name                 | Opponent      | Training focus                                      |
| ----: | -------------------- | ------------- | --------------------------------------------------- |
|     1 | Reach the Ball       | None          | Move toward the ball and touch it                   |
|     2 | Push Toward Goal     | None          | Touch the ball and move it toward the opponent goal |
|     3 | Score Goals          | None          | Use normal goal rewards with scoring enabled        |
|     4 | Weak Random Opponent | Weak random   | Add light pressure from an opponent                 |
|     5 | Self-Play Mirror     | Current model | Play against the learner's current policy           |

Curriculum runs still write the same `episodes.csv`, checkpoints, and optional `steps.csv` files. The run `config.json` includes the full curriculum schedule.

## Alpha Curriculum Learning Plots

The `alpha` run is a full curriculum training run using the V12 reward system and the V10 18-value state representation. It trained agent 1 for 5,000 episodes: 1,000 episodes in each curriculum stage.

![Alpha curriculum dashboard](assets/plots/alpha/curriculum_dashboard.png)

### Alpha Final Performance

In the final 100 episodes, alpha reached a much cleaner scoring rhythm:

| Last 100 Episodes         |    Alpha |
| ------------------------- | -------: |
| Agent 1 average reward    | `155.10` |
| Agent 1 goals / episode   |   `0.99` |
| Agent 2 goals / episode   |   `0.00` |
| Agent 1 touches / episode |   `1.68` |
| Average episode length    |  `138.5` |
| Final epsilon             |   `0.05` |

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

| Option         |           Default | Description                           |
| -------------- | ----------------: | ------------------------------------- |
| `--run-dir`    |          required | Folder containing `episodes.csv`      |
| `--output-dir` | `<run-dir>/plots` | Where PNG files are saved             |
| `--window`     |             `100` | Rolling average window                |
| `--show`       |               off | Open an interactive matplotlib window |

## Training Run Visualizations

These charts compare the original `agent1` training run with the newer `newAgent` run.
Both runs used 1k episodes of 1k steps, which makes up to `1 MILLION` possible environment steps per run.

Note: this is not a perfectly controlled A/B test. `agent1` used the older V7 setup with the 9-value state vector and reward v1 against a random opponent. `newAgent` used the V12 setup with the 18-value state vector, reward v2, and checkpoint self-play seeded from `agent2`.

### Comparison Dashboard

![agent1 vs newAgent comparison dashboard](assets/plots/comparison/agent1_vs_newAgent/comparison_dashboard.png)

### Training Comparison

`newAgent` gets much higher shaped reward under V12, but `agent1` still scored and won more often in the final 100 episodes. This suggests V12 changed the reward incentives substantially, but goal conversion still needs work.

| Last 100 Episodes         | agent1 run | newAgent run |
| ------------------------- | ---------: | -----------: |
| Average reward            |    `73.67` |     `158.11` |
| Win rate                  |      `54%` |         `7%` |
| Goals / episode           |     `0.54` |       `0.07` |
| Touches / episode         |     `1.65` |       `0.75` |
| Connected kicks / episode |     `0.06` |       `0.13` |
| Steals / episode          |     `0.01` |       `0.00` |
| Average distance to ball  |   `171.94` |     `202.49` |
| Average episode length    |   `554.33` |     `925.31` |
| Final epsilon             |     `0.05` |       `0.05` |

## Output Data

Every logged run creates a folder under:

```text
data/training_runs/<run_name>/
```

Example training output:

```text
data/training_runs/run_YYYYMMDD_HHMMSS/
├── config.json
├── episodes.csv
├── steps.csv              # optional, only with --log-steps
├── checkpoints/
│   ├── episode_000100.pt
│   ├── episode_000200.pt
│   └── final.pt
├── opponents/             # self_play.py only
│   ├── episode_000250.pt
│   └── episode_000500.pt
└── plots/
    ├── rewards.png
    ├── winrate.png
    ├── goals.png
    ├── touches.png
    ├── distance_to_ball.png
    ├── episode_length.png
    └── epsilon.png
```

`episodes.csv` includes episode length, winner, scores, goals, touches, kicks, steals, average distance to ball, total rewards, reward component totals, epsilon, and timestamp.

## Architecture

The detailed architecture note is in [ARCHITECTURE.md](ARCHITECTURE.md).

```mermaid
flowchart LR
    A["Pygame modes<br/>main.py, demo.py, manual.py"] --> E["SoccerEnv"]
    B["Training modes<br/>train.py, self_play.py"] --> E
    C["2v2 baseline<br/>run_team_random.py"] --> T["TeamSoccerEnv"]
    E --> P["Physics, rules, rewards, state"]
    T --> P
    B --> D["DQNAgent + replay buffer"]
    E --> L["EpisodeLogger + Metrics"]
    T --> L
    L --> CSV["episodes.csv"]
    CSV --> G["analysis/plot_*.py"]
```

The 1v1 environment is the main reinforcement learning target. It exposes a compact Gym-style API for training and a Pygame renderer for visual inspection. The 2v2 environment is currently a clean foundation with team-level rewards and baseline team agents.

## Reward Shaping V2

V12 keeps the original goal/touch/progress signals and adds soccer-specific reward shaping with component-level logging. This makes reward behavior easier to debug in `episodes.csv`.

| Event or behavior                     | Reward / penalty |
| ------------------------------------- | ---------------: |
| Score a goal                          |           `+100` |
| Concede a goal                        |           `-100` |
| Touch the ball                        |             `+2` |
| Ball moves toward opponent goal       |           `+0.2` |
| Steal possession                      |             `+8` |
| Useful kick toward opponent goal      |             `+1` |
| Move into a better attacking position |           `+0.5` |
| Own goal                              |             `-1` |
| Unnecessary kick                      |          `-0.05` |
| Energy usage                          | `-0.001 * speed` |
| Push ball toward own goal             |           `-0.2` |
| Every step                            |          `-0.01` |

Team 1 attacks the right goal. Team 2 attacks the left goal.

Reward components are logged separately for both agents:

```text
goal_reward
touch_reward
progress_reward
steal_reward
useful_kick_reward
attacking_position_reward
own_goal_penalty
unnecessary_kick_penalty
energy_penalty
own_goal_push_penalty
time_penalty
```

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

|  ID | Action |
| --: | ------ |
| `0` | Stay   |
| `1` | Up     |
| `2` | Down   |
| `3` | Left   |
| `4` | Right  |
| `5` | Kick   |

## State Representation

V10 expands the environment state from the original compact state to an 18-value vector:

```text
[
  agent_1_x,
  agent_1_y,
  agent_2_x,
  agent_2_y,
  ball_x,
  ball_y,
  ball_vx,
  ball_vy,
  agent_1_distance_to_ball,
  agent_1_angle_to_ball,
  agent_2_distance_to_ball,
  agent_2_angle_to_ball,
  ball_distance_to_agent_1_goal,
  ball_distance_to_agent_2_goal,
  ball_moving_toward_agent_1_goal,
  ball_moving_toward_agent_2_goal,
  last_touch_owner,
  steps,
]
```

`last_touch_owner` is encoded as `0`, `1`, or `2`. Because the DQN input size changed in V10, checkpoints from older state sizes should be retrained before use with the current model.

## Learning Algorithm

RLStriker uses Deep Q-Learning for the trainable 1v1 agent:

| Piece | Implementation |
| ----- | -------------- |
| Q-network | `agents/model.py` |
| Agent logic | `agents/dqn_agent.py` |
| Replay memory | `agents/replay_buffer.py` |
| Random baseline | `agents/random_agent.py` |
| Checkpoint opponent | `agents/checkpoint_opponent.py` |

At each step, the learner chooses a discrete action with epsilon-greedy exploration, stores the transition in replay memory, samples mini-batches, updates the Q-network with temporal-difference targets, and periodically syncs a target network. Checkpoints are saved as `.pt` files with metadata such as run name, episode, trained side, reward version, and state size.

Self-play adds older checkpoints to the opponent pool so the learner faces stronger policies than pure random movement. Curriculum learning stages make the task easier at first by focusing on reaching the ball, pushing toward goal, scoring, weak opponents, and finally self-play-like pressure.

## Repository Structure

```text
RLStriker/
├── ARCHITECTURE.md
├── agents/
│   ├── checkpoint_opponent.py
│   ├── dqn_agent.py
│   ├── model.py
│   ├── random_agent.py
│   ├── replay_buffer.py
│   └── team_agent.py
├── env/
│   ├── constants.py
│   ├── entities.py
│   ├── field.py
│   ├── physics.py
│   ├── rewards.py
│   ├── rules.py
│   ├── soccer_env.py
│   ├── state.py
│   └── team_soccer_env.py
├── logging_utils/
│   ├── episode_logger.py
│   └── metrics.py
├── analysis/
│   ├── plot_all.py
│   ├── plot_goals.py
│   ├── plot_rewards.py
│   ├── plot_touches.py
│   └── plot_winrate.py
├── curriculum/
│   ├── curriculum_manager.py
│   └── stages.py
├── visual/
│   ├── debug_overlay.py
│   └── menu.py
├── data/
│   └── training_runs/
├── demo.py
├── main.py
├── manual.py
├── run_random.py
├── run_team_random.py
├── self_play.py
├── train.py
├── requirements.txt
└── README.md
```

## Development Roadmap

The roadmap is complete through V16:

- V1-V3: Pygame field, physics, rules, and Gym-style environment
- V4-V8: Random agents, logging, rewards, DQN training, and graph generation
- V9-V12: Curriculum learning, better state representation, self-play, and richer rewards
- V13-V16: Demo mode, human-vs-AI mode, 2v2 team mode, and portfolio polish

Good next research directions are stronger 2v2 learning, model evaluation tournaments, reward ablations, and longer curriculum/self-play runs.

## License

MIT. See [LICENSE](LICENSE).
