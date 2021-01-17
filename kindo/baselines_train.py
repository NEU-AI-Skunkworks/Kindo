import logging
import secrets
import typing
from pathlib import Path

from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.vec_env import DummyVecEnv

from kindo import abs_path, globals
from kindo.callbacks import HistorySavingCallback, StopTrainingWhenMean100EpReward

logger = logging.getLogger(__name__)


def compile_random_model_name(model: BaseRLModel) -> str:
    return f"{model.__class__.__name__}_{secrets.token_hex(2)}"


def _train_model(
    model: BaseRLModel,
    total_timesteps: int = 100000,
    log_interval: int = 10,
    model_name: typing.Optional[str] = None,
    save_history=True,
    maximum_episode_reward=None,
    stop_training_threshold=195,
):
    model_name = model_name or compile_random_model_name(model)

    if isinstance(model.env, DummyVecEnv):
        model_dir = f"{globals.save_path}/{model.env.envs[0].env.__class__.__name__}/{model_name}"
    else:
        model_dir = (
            f"{globals.save_path}/{model.env.env.__class__.__name__}/{model_name}"
        )

    Path(model_dir).mkdir(parents=True, exist_ok=True)

    stop_callback = StopTrainingWhenMean100EpReward(
        reward_threshold=stop_training_threshold, timestep_activation_threshold=5000
    )
    history_saving_callback = HistorySavingCallback(
        total_timesteps=total_timesteps,
        history_save_dir=model_dir,
        stop_callback=stop_callback,
        maximum_episode_reward=maximum_episode_reward,
    )

    model = model.learn(
        total_timesteps=total_timesteps,
        log_interval=log_interval,
        callback=history_saving_callback,
    )
    model.save(abs_path(f"{model_dir}/model.pkl"), cloudpickle=True)
    model.env.close()
    logger.info(f"Model {model.__class__.__name__} is trained and saved to {model_dir}")


def train_a_model(
    model: BaseRLModel,
    total_timesteps: int = 100000,
    log_interval: int = 10,
    model_name: typing.Optional[str] = None,
    save_history: bool = True,
    maximum_episode_reward: int = None,
):
    _train_model(
        model,
        total_timesteps,
        log_interval,
        model_name,
        save_history,
        maximum_episode_reward,
    )


def train_a_couple_of_models(
    models: typing.List[BaseRLModel],
    total_timesteps: int = 100000,
    log_interval: int = 10,
    model_names: typing.Optional[typing.List[str]] = None,
    save_history: bool = True,
    maximum_episode_reward: int = None,
    stop_training_threshold: int = 195,
):
    if model_names is not None:
        assert len(models) == len(model_names), (
            "The length of the `model_names` list should be the "
            "same as `models` list"
        )
    else:
        model_names = [compile_random_model_name(model) for model in models]

    for model, model_name in zip(models, model_names):
        print(f"===Training {model_name}===")
        _train_model(
            model=model,
            total_timesteps=total_timesteps,
            log_interval=log_interval,
            model_name=model_name,
            save_history=save_history,
            maximum_episode_reward=maximum_episode_reward,
            stop_training_threshold=stop_training_threshold,
        )
        print(f"===Finished training {model_name}===")
