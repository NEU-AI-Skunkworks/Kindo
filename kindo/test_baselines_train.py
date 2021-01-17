import gym
from stable_baselines import A2C, ACER, ACKTR
from stable_baselines.deepq import DQN
from stable_baselines.ppo2 import PPO2

from kindo.baselines_train import train_a_couple_of_models

# replace policy and other parameters here as constants:
ENV_NAME = "CartPole-v0"

if __name__ == "__main__":
    models = [
        # Simple DQN
        DQN(
            policy="MlpPolicy",
            env=gym.make(ENV_NAME),
            learning_rate=1e-3,
            prioritized_replay=True,
            verbose=1,
        ),
        # Deep Q-Network with prioritized replay
        DQN(
            policy="MlpPolicy",
            env=gym.make(ENV_NAME),
            verbose=1,
            double_q=False,
            prioritized_replay=True,
            policy_kwargs=dict(dueling=False),
        ),
        # Double Q-network with all the extensions
        DQN(
            policy="MlpPolicy",
            env=gym.make(ENV_NAME),
            verbose=1,
            double_q=True,
            prioritized_replay=True,
            policy_kwargs=dict(dueling=True),
        ),
        # A2C
        A2C(policy="MlpPolicy", env=gym.make(ENV_NAME), verbose=1),
        # ACER
        ACER(policy="MlpPolicy", env=gym.make(ENV_NAME), verbose=1),
        # ACKTR
        ACKTR(policy="MlpPolicy", env=gym.make(ENV_NAME), verbose=1),
        # PPO
        PPO2(policy="MlpPolicy", env=gym.make(ENV_NAME), verbose=1),
    ]
    model_names = [
        "Simple DQN",
        "Prioritizied Replay DQN",
        "Double DQN",
        "A2C",
        "ACER",
        "ACKTR",
        "PPO",
    ]
    train_a_couple_of_models(
        models=models,
        total_timesteps=4000,
        model_names=model_names,
        maximum_episode_reward=200,
        stop_training_threshold=35,
    )
