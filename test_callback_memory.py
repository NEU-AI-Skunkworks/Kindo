from memory_profiler import profile
from stable_baselines.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback
import gym


class StopTrainingCallback(StopTrainingOnRewardThreshold):
    def __init__(self, *args, **kwargs):
        self.memory_weight = " " * 128 * 1024 * 1024
        super().__init__(*args, **kwargs)


@profile
def monitor():
    env = gym.make("CartPole-v0")
    for _ in range(60):
        stop_callback = StopTrainingCallback(reward_threshold=5000)
        eval_callback = EvalCallback(callback_on_new_best=stop_callback, eval_env=env)

    print("monitoring at the last point")


if __name__ == "__main__":
    monitor()
