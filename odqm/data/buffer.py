import numpy as np
import torch
import os


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size: int, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self._pointer = 0
        self.size = max_size

        self._observations = np.zeros((max_size, state_dim))
        self._actions = np.zeros((max_size, action_dim))
        self._next_observations = np.zeros((max_size, state_dim))
        self._rewards = np.zeros((max_size,))
        self._done = np.zeros((max_size,))

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        """_summary_

        Args:
            state (np.ndarray): _description_
            action (np.ndarray): _description_
            next_state (np.ndarray): _description_
            reward (float): _description_
            done (bool): _description_

        Raises:
            BufferOverflowException: _description_
        """

        self._observations[self._pointer] = state
        self._actions[self._pointer] = action
        self._next_observations[self._pointer] = next_state
        self._rewards[self._pointer] = reward
        self._done[self._pointer] = done

        self._pointer += 1

    def load_dict(self, data: dict) -> None:
        """
        Load dataset from dictionary.

        Args:
            data (dict): Dictionary have to contain next keys:
            - "observations": 2D-array array with the shape (N, state_dim) where each element of the array is a observation on the current timestep
            - "actions": 2D-array with the shape (N, action_dim), where each element of the array is an action agent performed of the current timestep
            - "next_observations": 2D-array with the shape (N, state_dim), where each element of the array is an observation, agent got after performing an action
            - "rewards": 1D-array with the shape (N, ), where each element is a reward agent got after performing an action
            - "done": 1D-array with the shape (N, ), where each element indicates whether episode is done or not

            N must not exceed buffer max capacity.
        """

        assert (
            data["observations"].shape[0]
            == data["actions"].shape[0]
            == data["next_observations"].shape[0]
            == data["rewards"].shape[0]
            == data["done"].shape[0]
        ), "Data have inconsistent size of first dimension"

        N = data["observations"].shape[0]

        assert N <= self.size, "Array size exceeds max capacity"

        self._observations[:N] = data["observations"][:N]
        self._actions[:N] = data["actions"][:N]
        self._next_observations = data["next_observations"][:N]
        self._rewards = data["rewards"][:N]
        self._done = data["done"][:N]

        self._pointer = N

    def sample(
        self, batch_size: int
    ):
        idx = np.random.randint(low=0, high=self._pointer, size=(batch_size,))

        return {
            'state': torch.FloatTensor(self._observations[idx]).to(self.device),
            'action': torch.FloatTensor(self._actions[idx]).to(self.device),
            'reward': torch.FloatTensor(self._rewards[idx]).to(self.device),
            'next_state': torch.FloatTensor(self._next_observations[idx]).to(self.device),
            'done': torch.FloatTensor(self._done[idx]).to(self.device),
        }

    @property
    def state_dim(self):
        return self._observations.shape[-1]

    @property   
    def action_dim(self):
        return self._actions.shape[-1]

    def to(self, device):
        self.device = device

    def __len__(self):
        return self._pointer

# from d4rl.offline_env import OfflineEnv

# def buffer_from_gym(
#     env: OfflineEnv, dataset: dict = None, terminate_on_end=False, **kwargs
# ) -> ReplayBuffer:
#     if dataset is None:
#         dataset = env.get_dataset(**kwargs)

#     N = dataset["rewards"].shape[0]
#     obs_ = []
#     next_obs_ = []
#     action_ = []
#     reward_ = []
#     done_ = []

#     # The newer version of the dataset adds an explicit
#     # timeouts field. Keep old method for backwards compatability.
#     use_timeouts = "timeouts" in dataset

#     episode_step = 0
#     for i in range(N - 1):
#         obs = dataset["observations"][i].astype(np.float32)
#         new_obs = dataset["observations"][i + 1].astype(np.float32)
#         action = dataset["actions"][i].astype(np.float32)
#         reward = dataset["rewards"][i].astype(np.float32)
#         done_bool = bool(dataset["terminals"][i])

#         if use_timeouts:
#             final_timestep = dataset["timeouts"][i]
#         else:
#             final_timestep = episode_step == env._max_episode_steps - 1
#         if (not terminate_on_end) and final_timestep:
#             # Skip this transition and don't apply terminals on the last step of an episode
#             episode_step = 0
#             continue
#         if done_bool or final_timestep:
#             episode_step = 0

#         obs_.append(obs)
#         action_.append(action)
#         reward_.append(reward)
#         next_obs_.append(new_obs)
#         done_.append(done_bool)

#         obs_ = np.array(obs_)
#         action_ = np.array(action_)
#         reward_ = np.array(reward_)
#         next_obs_ = np.array(next_obs_)
#         done_ = np.array(done_)

#         state_dim = obs.shape[-1]
#         action_dim = action.shape[-1]
#         max_size = obs.shape[0]

#         buffer = ReplayBuffer(
#             state_dim=state_dim, action_dim=action_dim, max_size=max_size
#         )

#         buffer.load_dict(
#             {
#                 "observations": obs_,
#                 "next_observations": next_obs_,
#                 "rewards": reward_,
#                 "next_obs_": next_obs_,
#                 "done": done_,
#             }
#         )

#         return buffer


def make_buffer(data_conf: dict):
    data_name = data_conf['name']

    if os.path.isdir(data_name) and os.path.exists(data_name):
        return buffer_from_path(data_name)


def buffer_from_path(path: str):

    states_path = os.path.join(path, "states.npy")
    actions_path = os.path.join(path, "actions.npy")
    rewards_path = os.path.join(path, "rewards.npy")
    dones_path = os.path.join(path, "dones.npy")

    states = np.load(states_path)
    actions = np.load(actions_path)
    rewards = np.load(rewards_path)
    dones = np.load(dones_path)

    N = rewards.shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    for i in range(N - 1):
        obs = states[i].astype(np.float32)
        new_obs = states[i + 1].astype(np.float32)
        action = actions[i].astype(np.float32)
        reward = rewards[i].astype(np.float32)
        done_bool = bool(dones[i])

        if done_bool:
            episode_step = 0

        obs_.append(obs)
        action_.append(action)
        reward_.append(reward)
        next_obs_.append(new_obs)
        done_.append(done_bool)

    obs_ = np.array(obs_)
    action_ = np.array(action_)
    reward_ = np.array(reward_)
    next_obs_ = np.array(next_obs_)
    done_ = np.array(done_)

    state_dim = obs_.shape[-1]
    action_dim = action_.shape[-1]
    max_size = obs_.shape[0]

    buffer = ReplayBuffer(
        state_dim=state_dim, action_dim=action_dim, max_size=max_size
    )

    buffer.load_dict(
        {
            "observations": obs_,
            "next_observations": next_obs_,
            "rewards": reward_,
            "done": done_,
            "actions": action_,
        }
    )

    return buffer
