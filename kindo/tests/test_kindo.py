"""
 isort:skip_file
"""
# fmt: off
import os
import shutil
import sys
import unittest

import gym  # noqa
import tensorflow as tf  # noqa
from tf_agents.networks import q_network, actor_distribution_network, value_network  # noqa
from tf_agents.agents.dqn import dqn_agent  # noqa
from tf_agents.agents.ppo import ppo_agent  # noqa
from tf_agents.agents.reinforce import reinforce_agent  # noqa
from tf_agents.utils import common  # noqa
from stable_baselines3 import DQN, PPO, A2C  # noqa

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../", "../")))  # noqa
from kindo import train, train_multiple  # noqa
from kindo.paths import get_saved_environments, get_trained_model_names, save_path  # noqa
from kindo import environment_converter  # noqa


# fmt: on


class TestingKindoMethods(unittest.TestCase):
    def test_train_stable_baselines(self):
        env = gym.make("CartPole-v0")
        model_name = "dqn_test"
        model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=1e-3,
            verbose=1,
        )
        train(
            model=model,
            env=env,
            total_timesteps=1500,
            stop_threshold=4000,
            model_name=model_name,
            maximum_episode_reward=195,
        )
        trained_env = get_saved_environments()[0]
        trained_models = get_trained_model_names(trained_env)
        model_saved = model_name in trained_models
        shutil.rmtree(save_path)
        self.assertTrue(model_saved)

    def test_train_tf_agetns(self):
        env_name = "CartPole-v0"
        model_name = "tf_agents_dqn"
        env = gym.make(env_name)
        train_env = environment_converter.gym_to_tf(env)
        fc_layer_params = (100,)
        q_net = q_network.QNetwork(
            input_tensor_spec=train_env.observation_spec(),
            action_spec=train_env.action_spec(),
            fc_layer_params=fc_layer_params,
        )
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
        agent = dqn_agent.DqnAgent(
            time_step_spec=train_env.time_step_spec(),
            action_spec=train_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
        )
        agent.initialize()

        train(
            model=agent,
            env=env,
            total_timesteps=1500,
            stop_threshold=4000,
            model_name=model_name,
            maximum_episode_reward=195,
        )
        trained_env = get_saved_environments()[0]
        trained_models = get_trained_model_names(trained_env)
        model_saved = model_name in trained_models
        shutil.rmtree(save_path)
        self.assertTrue(model_saved)

    def test_tf_agents_on_policy_agent(self):
        learning_rate = 1e-3
        actor_fc_layers = (200, 100)
        value_fc_layers = (200, 100)
        env_name = "CartPole-v0"
        gym_env = gym.make(env_name)
        model_name = "ppo_tf_agent"
        train_env = environment_converter.gym_to_tf(gym_env)
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            train_env.observation_spec(),
            train_env.action_spec(),
            fc_layer_params=actor_fc_layers,
        )
        value_net = value_network.ValueNetwork(
            train_env.observation_spec(), fc_layer_params=value_fc_layers
        )

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        agent = ppo_agent.PPOAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            optimizer,
            actor_net=actor_net,
            value_net=value_net,
        )
        agent.initialize()

        # Train
        train(agent, gym_env, 2000, 195, model_name, 200)
        trained_env = get_saved_environments()[0]
        trained_models = get_trained_model_names(trained_env)
        model_saved = model_name in trained_models
        shutil.rmtree(save_path)
        self.assertTrue(model_saved)

    def test_multiple_stable_baselines(self):
        env_name = "CartPole-v0"
        env = gym.make(env_name)
        models = [
            DQN("MlpPolicy", gym.make(env_name), learning_rate=1e-3),
            A2C(policy="MlpPolicy", env=gym.make(env_name), verbose=1),
            PPO(policy="MlpPolicy", env=gym.make(env_name), verbose=1),
        ]
        model_names = ["Simple DQN", "A2C", "PPO"]
        train_multiple(models, env, 1470, 195, model_names, 200)
        trained_env = get_saved_environments()[0]
        trained_models = get_trained_model_names(trained_env)
        model_saved = set(model_names) == set(trained_models)
        shutil.rmtree(save_path)
        self.assertTrue(model_saved)

    def test_multiple_tf_agents(self):
        env_name = "CartPole-v0"
        # DQN
        env = gym.make(env_name)
        train_env = environment_converter.gym_to_tf(env)
        fc_layer_params = (100,)
        q_net = q_network.QNetwork(
            input_tensor_spec=train_env.observation_spec(),
            action_spec=train_env.action_spec(),
            fc_layer_params=fc_layer_params,
        )
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
        dqn_tf_agent = dqn_agent.DqnAgent(
            time_step_spec=train_env.time_step_spec(),
            action_spec=train_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
        )
        dqn_tf_agent.initialize()

        # PPO
        env = gym.make(env_name)
        actor_fc_layers = (200, 100)
        value_fc_layers = (200, 100)
        learning_rate = 1e-3
        train_env = environment_converter.gym_to_tf(env)
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            train_env.observation_spec(),
            train_env.action_spec(),
            fc_layer_params=actor_fc_layers,
        )
        value_net = value_network.ValueNetwork(
            train_env.observation_spec(), fc_layer_params=value_fc_layers
        )
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

        ppo_tf_agent = ppo_agent.PPOAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            optimizer,
            actor_net=actor_net,
            value_net=value_net,
        )
        ppo_tf_agent.initialize()

        # REINFORCE:
        env = gym.make(env_name)
        train_env = environment_converter.gym_to_tf(env)
        learning_rate = 1e-3
        fc_layer_params = (100,)
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            train_env.observation_spec(),
            train_env.action_spec(),
            fc_layer_params=fc_layer_params,
        )
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        train_step_counter = tf.compat.v2.Variable(0)
        reinforce_tf_agent = reinforce_agent.ReinforceAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            actor_network=actor_net,
            optimizer=optimizer,
            normalize_returns=True,
            train_step_counter=train_step_counter,
        )
        reinforce_tf_agent.initialize()

        agents = [dqn_tf_agent, ppo_tf_agent, reinforce_tf_agent]
        agent_names = ["dqn_agent", "ppo_agent", "reinforce_agent"]

        train_multiple(agents, env, 1470, 195, agent_names, 200)

        trained_env = get_saved_environments()[0]
        trained_models = get_trained_model_names(trained_env)
        model_saved = set(agent_names) == set(trained_models)
        shutil.rmtree(save_path)
        self.assertTrue(model_saved)


# running the tests
if __name__ == "__main__":
    unittest.main()
