import gym
import tensorflow as tf
from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks import actor_distribution_network, value_network

from kindo import environment_converter
from kindo.tf_agents_api.tf_agents_train import train_on_policy_agent

LEARNING_RATE = 1e-3
ACTOR_FC_LAYERS = (200, 100)
VALUE_FC_LAYERS = (200, 100)
ENV_NAME = "CartPole-v0"

if __name__ == "__main__":
    # TF required calls (If I don't do that, it will raise an error)
    tf.compat.v1.enable_resource_variables()
    tf.compat.v1.enable_v2_behavior()

    # Define Environment:
    gym_env = gym.make(ENV_NAME)
    train_env = environment_converter.gym_to_tf(gym_env)
    # Initialize Model (Agent)
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=ACTOR_FC_LAYERS,
    )

    value_net = value_network.ValueNetwork(
        train_env.observation_spec(), fc_layer_params=VALUE_FC_LAYERS
    )

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_step_counter = tf.compat.v2.Variable(0)

    agent = ppo_agent.PPOAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        optimizer,
        actor_net=actor_net,
        value_net=value_net,
    )
    agent.initialize()

    # Train
    train_on_policy_agent(model=agent, train_env=train_env, total_timesteps=15000)
