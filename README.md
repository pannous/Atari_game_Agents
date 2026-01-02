# Atari Game Agents

Deep Reinforcement Learning agents for mastering Atari games through DQN and Double Dueling DQN algorithms.

## Overview

This project implements reinforcement learning agents that learn to play Atari games using state-of-the-art deep Q-learning techniques. The agents are trained using Gymnasium (formerly OpenAI Gym) and PyTorch, showcasing the practical application of deep reinforcement learning in gaming environments.

## Features

### Implemented Algorithms
- **DQN (Deep Q-Network)**: Classic deep reinforcement learning approach with experience replay and target networks
- **D3QN (Double Dueling DQN)**: Advanced architecture combining double Q-learning with dueling network architecture for improved performance

### Supported Games
- Pong-v5 (current implementation)
- Additional Atari games coming soon: Breakout, Enduro, Assault

## Project Structure

```
Atari_game_Agents/
├── AI_gameplay/
│   ├── Agents/
│   │   ├── DQN_Agent/
│   │   │   ├── DeepQAgent.py
│   │   │   └── DeepQNetwork.py
│   │   └── D3QN_Agent/
│   │       ├── DDDeepQAgent.py
│   │       └── DualingDeepQNetwork.py
│   ├── Model_for_Agents/      # Trained model checkpoints
│   ├── Videos/                # Gameplay recordings
│   ├── atari_wrapper.py       # Environment preprocessing wrappers
│   ├── main.py               # Training and testing interface
│   └── Visualize.py          # Visualization utilities
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Installation

### Requirements
- Python 3.10 or higher
- PyTorch 2.0+
- Gymnasium with Atari support

### Setup

```bash
# Clone the repository
git clone https://github.com/QuangNguyen2910/Atari_game_Agents.git
cd Atari_game_Agents

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"
```

## Usage

### Training an Agent

```bash
cd AI_gameplay
python main.py
```

When prompted:
1. Choose agent type: `DQN` or `D3QN`
2. Select option: `1` for training, `2` for testing
3. If training, choose: `1` for training from scratch, `2` to continue from saved model

### Testing a Trained Agent

Run the same command and select option `2` to test a trained agent. The agent will play the game and save a video of the gameplay.

## Architecture

### DQN Agent
- Experience replay memory for stable learning
- Separate target network for reducing overestimation
- Epsilon-greedy exploration strategy
- Huber loss for robust training

### D3QN Agent
- Dueling network architecture separating value and advantage streams
- Double Q-learning to reduce value overestimation
- All features from DQN agent

## Hyperparameters

Default configuration:
- Learning rate: 0.0001
- Discount factor (gamma): 0.99
- Batch size: 32
- Memory size: 10,000 transitions
- Epsilon decay: 5e-6
- Target network update frequency: 1000 steps

## Future Work

- Additional agent architectures (Deep NeuroEvolution, A3C, PPO)
- More Atari games support
- Hyperparameter optimization
- Distributed training support
- Performance benchmarking against published results

## Author

Quang Nguyen

## License

MIT License

## References

- Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature
- Van Hasselt et al. (2016). Deep Reinforcement Learning with Double Q-learning
- Wang et al. (2016). Dueling Network Architectures for Deep Reinforcement Learning
