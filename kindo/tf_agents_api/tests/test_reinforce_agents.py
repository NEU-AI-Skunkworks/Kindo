import gym
import tensorflow as tf
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.networks import actor_distribution_network

from kindo import environment_converter
from kindo.tf_agents_api.tf_agents_train import train_on_policy_tf_agent

LEARNING_RATE = 1e-3
FC_LAYER_PARAMS = (100,)
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
        fc_layer_params=FC_LAYER_PARAMS,
    )

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_step_counter = tf.compat.v2.Variable(0)
    agent = reinforce_agent.ReinforceAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        actor_network=actor_net,
        optimizer=optimizer,
        normalize_returns=True,
        train_step_counter=train_step_counter,
    )

    # tf_agent = ppo_agent.PPOAgent(
    #     tf_env.time_step_spec(),
    #     tf_env.action_spec(),
    #     optimizer,
    #     actor_net=actor_net,
    #     value_net=value_net,
    #     num_epochs=num_epochs,
    #     debug_summaries=debug_summaries,
    #     summarize_grads_and_vars=summarize_grads_and_vars,
    #     train_step_counter=global_step)

    agent.initialize()

    # Train
    train_on_policy_tf_agent(model=agent, train_env=train_env, total_timesteps=15000)
