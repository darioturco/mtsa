import numpy as np
import random


class ReplayBuffer(object):
    def __init__(self, env, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self._env = env

    def __len__(self):
        return len(self._storage)

    def add(self, action_features, reward, obs2):
        data = (action_features, reward, obs2)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        action_featuress, rewards, obss = [], [], []
        for i in idxes:
            data = self._storage[i]
            action_features, reward, obs = data
            action_featuress.append(action_features)
            rewards.append(reward)
            obss.append(obs)

        return action_featuress, np.array(rewards), obss

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def __repr__(self):
        return " - ".join([str(data[:2]) for data in self._storage])

    """
    # Esta funcion fallaba si n_step era igual a 1 (quedo vieja, BORRAR)
    def get_experience_from_random_policy(self, total_steps, nstep=1):
        states = []
        last_steps = []
        obs = self._env.reset()
        done = False

        for steps in range(total_steps):
            action = np.random.randint(len(obs))
            if len(last_steps) >= nstep:
                last_steps.pop(0)

            last_steps.append(obs[action])
            obs, reward, done, info = self._env.step(action)

            if done:
                for j in range(len(last_steps)):
                    states.append((self._env.context.compute_features(last_steps[j]), -len(last_steps) + j, None))
                last_steps = []
                obs = self._env.reset()

        if not done:
            for j in range(len(last_steps)):
                states.append((self._env.context.compute_features(last_steps[j]), -len(last_steps) + j, None))

        return states
    """
    def get_experience_from_random_policy(self, total_steps, nstep=1):
        # A random policy is run for total_steps steps, saving the observations in the format of the replay buffer
        states = []
        obs = self._env.reset()
        steps = 0

        last_steps = []
        while steps < total_steps:
            action = np.random.randint(len(obs))
            last_steps.append(obs[action])

            obs2, reward, done, info = self._env.step(action)

            if done:

                for j in range(len(last_steps)):
                    states.append((self._env.context.compute_features(last_steps[j]), -len(last_steps) + j, None))
                last_steps = []
                obs = self._env.reset()
            else:
                if len(last_steps) >= nstep:
                    aux = self._env.context.compute_feature_of_list(obs2)
                    states.append((self._env.context.compute_features(last_steps[0]), -nstep, aux))
                last_steps = last_steps[len(last_steps) - nstep + 1:]
                obs = obs2
            steps += 1
        return states
