## Table of Contents
- [Common Trainer Configurations](#common-trainer-configurations)
    - [train](#train)
    - [train_multiple](#train_multiple)
- [Trainer-specific Configurations](#trainer-specific-configurations)
    - [tf_agents_api](#tf_agents_trainer)
        - [train_on_policy_tf_agent](#train_on_policy_tf_agent)
        - [train_off_policy_tf_agent](#train_off_policy_tf_agent)
        - [train_tf_agent](#train_tf_agent)
    - [stable_baselines_api]
- [Result Plots](#result_plots)
    - history_manager(#history_manager)


## Common Trainer Configurations
### [model_trainer]
**train**
```python
def train(
    model: Union[BaseAlgorithm, TFAgent, Type[BaseAlgorithm], Type[TFAgent]],
    env: Union[Env, TimeLimit],
    total_timesteps: int,
    stop_threshold: int,
    model_name: Optional[str] = None,
    maximum_episode_reward: Optional[int] = None,
)
```
| **Setting**              | **Description**                                                                                   |
| :----------------------- | :-------------------------------------------------------------------------------------------------|
| `model` |  Type of model class              
| `env` | Environment from the OpenAI Gym library              
| `total_timesteps` | Number of time steps/ training iterations
| `stop_threshold` |  Average reward per episode that considered solved defined by the OpenAI Gym environments   
| `model_name` | Name of model class
| `maximum_episode_reward` | Maximum reward per episode

**train_multiple**
```python
def train_multiple(
    models: List[Union[BaseAlgorithm, TFAgent, ABCMeta]],
    env: Env,
    total_timesteps: int,
    stop_threshold: int,
    model_names: Optional[List[str]] = None,
    maximum_episode_reward: int = None,
)
```
| **Setting**              | **Description**                                                                                   |
| :----------------------- | :-------------------------------------------------------------------------------------------------|
| `model` |  A list of model classes types             
| `env` | Environment from the OpenAI Gym library              
| `total_timesteps` | Number of time steps/ training iterations
| `stop_threshold` |  Average reward per episode that considered solved defined by the OpenAI Gym environments   
| `model_name` | A list of model classes names
| `maximum_episode_reward` | Maximum reward per episode

## Trainer-specific Configurations
### [tf_agents_trainer]
**train_off_policy_tf_agent**
```python
def train_off_policy_tf_agent(
    model: TFAgent,
    train_env: TFPyEnvironment,
    total_timesteps: int,
    callback: callbacks.BaseKindoRLCallback = None,
)
```
| **Setting**              | **Description**                                                                                   |
| :----------------------- | :-------------------------------------------------------------------------------------------------|
| `model` | (default = `TFAgent`) Training model class type           
| `train_env` | Environment from the OpenAI Gym library              
| `total_timesteps` | Number of time steps/ training iterations
| `callback` | 

**train_on_policy_tf_agent**
```python
def train_on_policy_tf_agent(
    model: TFAgent,
    train_env: TFPyEnvironment,
    total_timesteps: int,
    callback: callbacks.BaseKindoRLCallback = None,
)
```
| **Setting**              | **Description**                                                                                   |
| :----------------------- | :-------------------------------------------------------------------------------------------------|
| `model` | (default = `TFAgent`) Training model class type           
| `train_env` | Environment from the OpenAI Gym library              
| `total_timesteps` | Number of time steps/ training iterations
| `callback` | 

**initialize_tf_agent**
```python
def initialize_tf_agent(
    model_class: ABCMeta, 
    train_env: TFPyEnvironment
)
```
| **Setting**              | **Description**                                                                                   |
| :----------------------- | :-------------------------------------------------------------------------------------------------|
| `model_class` | Training model class type           
| `train_env` | Environment from the OpenAI Gym library              

**train_tf_agent**
```python
def train_tf_agent(
    model: typing.Union[TFAgent, typing.Type[TFAgent]],
    env: gym.Env,
    total_timesteps: int,
    model_name: typing.Optional[str] = None,
    maximum_episode_reward: int = 200,
    stop_training_threshold: int = 195,
)
```
| **Setting**              | **Description**                                                                                   |
| :----------------------- | :-------------------------------------------------------------------------------------------------|
| `model` | Training model class type           
| `env` | Environment from the OpenAI Gym library    
| `total_timesteps` | Number of time steps/ training iterations
| `model_name` | Name of model class    
| `maximum_episode_reward` | (default = `200`) Maximum reward per episode
| `stop_training_threshold` | (default = `195`) Average reward per episode that considered solved defined by the OpenAI Gym environments



## Result Plots
### [history_manager]
| **Method**              | **Description**                                                                                   |
| :----------------------- | :------------------------------------------------------------------------------------------------|
| `plot_mean_rewards` | Mean reward over last 100 episodes of each trained model, presented in bar chart      
| `plot_mean_regrets` | Mean regrets over last 100 episodes of each trained model, presented in bar chart `episode_regret = maximum_episode_reward - episode_reward`
| `plot_time_spent_on_training` | Total time spent on each trained model in seconds, presented in bar chart
| `plot_episode_rewards` | The reward of every episode in each trained model, presented in line chart
| `plot_episode_regrets` | The regrets of every episode in each trained model, presented in line chart

[//]: # (reference links)
[model_trainer]: <https://github.com/NEU-AI-Skunkworks/kindo/blob/master/kindo/model_trainer.py>
[tf_agents_trainer]: <https://github.com/NEU-AI-Skunkworks/kindo/blob/master/kindo/tf_agents_api/tf_agents_trainer.py>
[history_manager]:<https://github.com/NEU-AI-Skunkworks/kindo/blob/master/kindo/history_manager.py>
