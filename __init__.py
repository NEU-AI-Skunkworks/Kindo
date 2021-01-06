import typing
import glob
from pathlib import Path
import tensorflow as tf
from kindo import globals

# Turn off Tensorflow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def set_history_dir(path):
    global globals
    globals.history_save_dir = path


def set_save_path(path):
    global globals
    globals.save_path = path


def abs_path(local_path):
    return str(Path(local_path).resolve())


def get_saved_environments() -> typing.List[str]:
    return [
        env_path.replace("saved/", "") for env_path in glob.glob(f"{globals.save_path}/*")
    ]


def get_trained_model_paths(env_name):
    return [
        model_path for model_path in glob.glob(f"{globals.save_path}/{env_name}/*")
    ]


def get_trained_model_names() -> typing.List[str]:
    return [
        model_path.replace("saved/", "").replace(".pkl", "")
        for model_path in glob.glob(f"{globals.save_path}/*.pkl")
    ]


from kindo.baselines_train import train_a_model, train_a_couple_of_models
