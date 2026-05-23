# RLStriker

RLStriker is a 2D 1v1 soccer reinforcement learning environment built with Python, Pygame, PyTorch, pandas, and matplotlib.

The project is being built version by version. The current version is **V13**, which adds demo mode for watching trained checkpoints play with a live debug overlay, on top of the existing environment, DQN training pipeline, graph generation tools, curriculum learning, richer V10 state representation, V11 checkpoint self-play, and V12 reward components.

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

Next planned version: **V14 - Human vs AI mode**.

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

| Player         | Move                 | Kick    |
| -------------- | -------------------- | ------- |
| Blue / agent 1 | `WASD` or arrow keys | `Space` |
| Red / agent 2  | `I J K L`            | `Enter` |

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

| Option                  | Default | Description                                  |
| ----------------------- | ------: | -------------------------------------------- |
| `--checkpoint`          | newAgent final | Model checkpoint to watch              |
| `--model-agent`         |     `1` | Side controlled by the model                 |
| `--opponent`            | random  | Use `random` or `checkpoint` opponent        |
| `--opponent-checkpoint` |    none | Checkpoint path for checkpoint opponent      |
| `--episodes`            |   `100` | Number of demo episodes before exiting       |
| `--fps`                 |    `60` | Playback speed                               |

## Self-Play

V11 adds checkpoint-based self-play. The current learner periodically saves model snapshots, stores them in an opponent pool, and trains against a mix of:

- Random opponents
- Older snapshots from the current run
- Optional checkpoint opponents passed with `--initial-opponent`

This helps avoid training only against weak random behavior and starts pushing the agent toward policies that can beat earlier versions of itself.

Useful self-play options:

| Option                    | Default | Description                                                |
| ------------------------- | ------: | ---------------------------------------------------------- |
| `--checkpoint-every`      |   `100` | Save regular learner checkpoints                           |
| `--opponent-refresh-every`|   `250` | Add the current learner to the opponent pool every N episodes |
| `--opponent-pool-size`    |     `8` | Maximum number of older checkpoint opponents to keep       |
| `--random-opponent-prob`  |  `0.25` | Chance of using a random opponent instead of a checkpoint  |
| `--initial-opponent`      |    none | Seed the pool with one or more compatible checkpoint files |

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

### Individual Comparison Graphs

| Metric           | Comparison graph                                                                                         |
| ---------------- | -------------------------------------------------------------------------------------------------------- |
| Rewards          | ![agent1 vs newAgent reward comparison](assets/plots/comparison/agent1_vs_newAgent/reward_comparison.png) |
| Win Rate         | ![agent1 vs newAgent win rate comparison](assets/plots/comparison/agent1_vs_newAgent/winrate_comparison.png) |
| Goals            | ![agent1 vs newAgent goals comparison](assets/plots/comparison/agent1_vs_newAgent/goals_comparison.png) |
| Ball Touches     | ![agent1 vs newAgent touches comparison](assets/plots/comparison/agent1_vs_newAgent/touches_comparison.png) |
| Connected Kicks  | ![agent1 vs newAgent kicks comparison](assets/plots/comparison/agent1_vs_newAgent/kicks_comparison.png) |
| Steals           | ![agent1 vs newAgent steals comparison](assets/plots/comparison/agent1_vs_newAgent/steals_comparison.png) |
| Distance to Ball | ![agent1 vs newAgent distance comparison](assets/plots/comparison/agent1_vs_newAgent/distance_to_ball_comparison.png) |
| Episode Length   | ![agent1 vs newAgent episode length comparison](assets/plots/comparison/agent1_vs_newAgent/episode_length_comparison.png) |
| Epsilon Decay    | ![agent1 vs newAgent epsilon comparison](assets/plots/comparison/agent1_vs_newAgent/epsilon_comparison.png) |

### Training Comparison

`newAgent` gets much higher shaped reward under V12, but `agent1` still scored and won more often in the final 100 episodes. This suggests V12 changed the reward incentives substantially, but goal conversion still needs work.

| Last 100 Episodes                | agent1 run | newAgent run |
| -------------------------------- | ---------: | -----------: |
| Average reward                   |    `73.67` |     `158.11` |
| Win rate                         |      `54%` |         `7%` |
| Goals / episode                  |     `0.54` |       `0.07` |
| Touches / episode                |     `1.65` |       `0.75` |
| Connected kicks / episode        |     `0.06` |       `0.13` |
| Steals / episode                 |     `0.01` |       `0.00` |
| Average distance to ball         |   `171.94` |     `202.49` |
| Average episode length           |   `554.33` |     `925.31` |
| Final epsilon                    |     `0.05` |       `0.05` |

## Output Data

Every logged run creates a folder under:

```text
data/training_runs/<run_name>/
```

Example V12 training output:

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

## Repository Structure

```text
RLStriker/
├── agents/
│   ├── checkpoint_opponent.py
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
├── curriculum/
│   ├── curriculum_manager.py
│   └── stages.py
├── visual/
│   └── debug_overlay.py
├── data/
│   └── training_runs/
├── demo.py
├── main.py
├── run_random.py
├── self_play.py
├── train.py
├── requirements.txt
└── README.md
```

## Development Roadmap

- V14: Human vs AI mode
- V15: Multi-agent 2v2 expansion
- V16: Final portfolio polish

## License

MIT. See [LICENSE](LICENSE).
