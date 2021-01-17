import typing
from pathlib import Path

import gym
from tf_agents.agents.ddpg.ddpg_agent import DdpgAgent
from tf_agents.agents.dqn.dqn_agent import DdqnAgent, DqnAgent
from tf_agents.agents.ppo.ppo_agent import PPOAgent
from tf_agents.agents.reinforce.reinforce_agent import ReinforceAgent
from tf_agents.agents.sac.sac_agent import SacAgent
from tf_agents.agents.td3.td3_agent import Td3Agent
from tf_agents.agents.tf_agent import TFAgent
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.replay_buffers import tf_uniform_replay_buffer

import kindo
from kindo import callbacks, environment_converter
from kindo.tf_agents_api import utils


class WrongModelError(Exception):
    pass


def train_off_policy_tf_agent(
    model: TFAgent,
    train_env: TFPyEnvironment,
    total_timesteps: int,
    callback: callbacks.BaseKindoRLCallback = None,
):
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        model.collect_data_spec, batch_size=train_env.batch_size, max_length=100000
    )
    collect_policy = model.collect_policy

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=64, num_steps=2
    ).prefetch(3)
    iterator = iter(dataset)

    if callback is not None:
        callback.init_callback(model=model, train_env=train_env)

    # Metrics over all the episodes
    locals_ = {"episode_rewards": [], "episode_losses": [], "episode_lengths": []}
    # Metrics for current episode
    curr_episode_losses, curr_episode_rewards, curr_episode_length = [], [], 0

    utils.step(
        environment=train_env, policy=collect_policy, replay_buffer=replay_buffer
    )

    if callback is not None:
        callback.on_training_start(locals_=locals_, globals_={})

    for _ in range(total_timesteps):
        reward, done = utils.step(
            environment=train_env, policy=collect_policy, replay_buffer=replay_buffer
        )
        experience, unused_info = next(iterator)
        train_loss = model.train(experience).loss.numpy()

        curr_episode_losses.append(train_loss)
        curr_episode_rewards.append(reward)
        curr_episode_length += 1

        if done:
            locals_["episode_rewards"].append(sum(curr_episode_rewards))
            locals_["episode_losses"].append(sum(curr_episode_losses))
            locals_["episode_lengths"].append(curr_episode_length)

            curr_episode_rewards = []
            curr_episode_losses = []
            curr_episode_length = 0

        if callback is not None:
            callback.update_locals(locals_)
            continue_training = callback.on_step()

            if not continue_training:
                break

    if callback is not None:
        callback.on_training_end()


def train_on_policy_tf_agent(
    model: TFAgent,
    train_env: TFPyEnvironment,
    total_timesteps: int,
    callback: callbacks.BaseKindoRLCallback = None,
):
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        model.collect_data_spec, batch_size=train_env.batch_size, max_length=100000
    )

    if callback is not None:
        callback.init_callback(model, train_env=train_env)

    collect_policy = model.collect_policy
    locals_ = {"episode_rewards": [], "episode_losses": [], "episode_lengths": []}
    passed_timesteps = 0

    if callback is not None:
        callback.on_training_start(locals_, {})

    while passed_timesteps < total_timesteps:
        episode_reward, episode_length = utils.step_episode(
            train_env, collect_policy, replay_buffer
        )
        passed_timesteps += episode_length
        locals_["episode_rewards"].append(episode_reward)
        locals_["episode_lengths"].append(episode_length)

        experience = replay_buffer.gather_all()
        train_loss = model.train(experience).loss.numpy()
        locals_["episode_losses"].append(train_loss)
        replay_buffer.clear()

        if callback is not None:
            callback.update_locals(locals_)
            continue_training = callback.on_steps(num_steps=episode_length)
            if not continue_training:
                break

    if callback is not None:
        callback.on_training_end()


def train_tf_agent(
    model: TFAgent,
    env: gym.Env,
    total_timesteps: int,
    model_name: typing.Optional[str] = None,
    maximum_episode_reward: int = 200,
    stop_training_threshold: int = 195,
):
    train_env = environment_converter.gym_to_tf(env)
    environment_name = env.__class__.__name__
    model_dir = f"{kindo.globals.save_path}/{environment_name}/{model_name}"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    stop_training_callback = callbacks.StopTrainingWhenMean100EpReward(
        reward_threshold=stop_training_threshold
    )
    history_saving_callback = callbacks.HistorySavingCallback(
        total_timesteps=total_timesteps,
        history_save_dir=model_dir,
        maximum_episode_reward=maximum_episode_reward,
        stop_callback=stop_training_callback,
    )

    if model.__class__ in [DqnAgent, DdqnAgent, DdpgAgent, SacAgent]:
        train_off_policy_tf_agent(
            model, train_env, total_timesteps, history_saving_callback
        )
    elif model.__class__ in [PPOAgent, ReinforceAgent, Td3Agent]:
        train_on_policy_tf_agent(
            model, train_env, total_timesteps, history_saving_callback
        )
    else:
        raise WrongModelError(
            f"Model of class `{model.__class__.__name__}` is not supported by Kindo API"
        )

    collect_policy = model.collect_policy
    saver = PolicySaver(collect_policy, batch_size=None)
    saver.save(f"{model_dir}/model")
