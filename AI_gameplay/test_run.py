#!/usr/bin/env python3
"""Quick test run of the DQN agent"""

import gymnasium as gym
import ale_py
import numpy as np
from Agents.DQN_Agent.DeepQAgent import DQAgent
from atari_wrapper import wrap_deepmind

# Register ALE environments
gym.register_envs(ale_py)

def quick_test():
    """Run a quick 5-episode test"""
    print("="*60)
    print("Quick Test: DQN Agent on Pong")
    print("="*60)

    env = gym.make("ALE/Pong-v5", render_mode='rgb_array')
    env = wrap_deepmind(env, frame_stack=True, scale=True)
    env.reset(seed=42)

    agent = DQAgent(
        gamma=0.99,
        epsilon=1.0,
        batch_size=32,
        n_actions=6,
        eps_end=0.01,
        input_dims=(4, 84, 84),
        lr=0.0001
    )

    print(f"\nDevice: {agent.Q_eval.device}")
    print(f"Network parameters: {sum(p.numel() for p in agent.Q_eval.parameters()):,}")
    print("\nRunning 5 test episodes...\n")

    scores = []
    n_games = 5

    for i in range(n_games):
        score = 0
        terminated = truncated = False
        observation, info = env.reset()
        observation = np.array(observation).reshape(4, 84, 84)
        steps = 0

        while not (terminated or truncated) and steps < 100:  # Limit steps for quick test
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            observation_ = np.array(observation_).reshape(4, 84, 84)
            score += reward

            # Store and learn
            done = terminated or truncated
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()

            observation = observation_
            steps += 1

        scores.append(score)
        print(f"Episode {i+1}/5: Score={score:.1f}, Steps={steps}, Epsilon={agent.epsilon:.3f}, Memory={agent.mem_cntr}")

    avg_score = np.mean(scores)
    print(f"\n{'='*60}")
    print(f"Test Complete!")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Final Epsilon: {agent.epsilon:.3f}")
    print(f"Memory Size: {agent.mem_cntr}/{agent.mem_size}")
    print(f"{'='*60}")

    env.close()

if __name__ == '__main__':
    quick_test()
