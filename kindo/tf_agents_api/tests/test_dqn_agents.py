import gym
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.utils import common

from kindo import environment_converter
from kindo.tf_agents_api.tf_agents_train import train_off_policy_tf_agent

if __name__ == "__main__":
    # TF required calls (If I don't do that, it will raise an error)
    tf.compat.v1.enable_resource_variables()
    tf.compat.v1.enable_v2_behavior()

    # Define Environment:
    env_name = "CartPole-v0"
    train_env = environment_converter.gym_to_tf(gym.make(env_name))
    # Initialize Model (Agent)
    fc_layer_params = (100,)
    q_net = q_network.QNetwork(
        input_tensor_spec=train_env.observation_spec(),
        action_spec=train_env.action_spec(),
        fc_layer_params=fc_layer_params,
    )
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
    # train_step_counter = tf.Variable(0)
    agent = dqn_agent.DqnAgent(
        time_step_spec=train_env.time_step_spec(),
        action_spec=train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
    )
    agent.initialize()

    # Train
    train_off_policy_tf_agent(model=agent, train_env=train_env, total_timesteps=10000)
