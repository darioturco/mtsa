import torch
import torch.nn as nn
import time
import numpy as np
import csv
from torch.optim import Adam
from torch.distributions import Categorical

from src.agents.replay_buffer import ReplayBuffer
from src.agents.agent import Agent

class Model:
    def __init__(self):
        pass

    def predict(self, s):
        raise NotImplementedError()

    def eval_batch(self, obss):
        raise NotImplementedError()

    def eval(self, s):
        raise NotImplementedError()

    def best(self, s):
        raise NotImplementedError()

    def current_loss(self):
        raise NotImplementedError()



class TorchModel(Model):
    def __init__(self, nfeatures, network, args):
        super().__init__()
        self.nfeatures = nfeatures

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = network
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=args["learning_rate"],
                                         momentum=args["momentum"],
                                         nesterov=args["nesterov"],
                                         weight_decay=args["weight_decay"])

        self.has_learned_something = False
        self.losses = []

    def eval_batch(self, ss):
        return np.array([self.eval(s) for s in ss])

    def eval(self, s):
        if not self.has_learned_something or s is None:
            return 0
        return float(self.predict(s).max())

    def best(self, s):
        if not self.has_learned_something or s is None:
            return 0
        return int(self.predict(s).argmax())

    def predict(self, s):
        if not self.has_learned_something or s is None:
            return 0
        return self.model(torch.tensor(s).to(self.device))

    def single_update(self, s, value):
        return self.batch_update(np.array([s]), np.array([value]))

    def batch_update(self, ss, values):
        ss = torch.tensor(ss).to(self.device)
        values = torch.tensor(values, dtype=torch.float, device=self.device).reshape(len(ss), 1)

        self.optimizer.zero_grad()
        pred = self.model(ss)

        loss = self.loss_fn(pred, values)
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())
        self.has_learned_something = True



    def nfeatures(self):
        return self.nfeatures

    # should be called only at the end of each episode
    def current_loss(self):
        avg_loss = np.mean(self.losses)
        self.losses = []
        return avg_loss

    def save(self, path):
        torch.save(self.model, path)
    @classmethod
    def load(cls, nfeatures, path, args):
        network = torch.load(path)
        new_model = cls(nfeatures, network, args)
        new_model.has_learned_something = True
        return new_model


class NeuralNetwork(nn.Module):
    def __init__(self, nfeatures, nnsize):
        super(NeuralNetwork, self).__init__()
        nnsize = list(nnsize) + [1]
        layers = [nn.Linear(nfeatures, nnsize[0])]
        for i in range(len(nnsize)-1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(nnsize[i], nnsize[i+1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x.to(torch.float))
        return x

class DQN(Agent):
    def __init__(self, env, nn_model, args, verbose=False):
        assert nn_model is not None

        self.env = env
        self.args = args
        self.model = nn_model

        self.target = None
        self.buffer = None

        self.trained = False

        self.training_start = None
        self.training_steps = 0
        self.epsilon = args["first_epsilon"]

        self.verbose = verbose

        self.training_data = []

        self.best_training_perf = {}
        self.last_best = None
        self.converged = False
        self.freq_save = 100

    def initializeBuffer(self):
        """ Initialize replay buffer uniformly with experiences """
        buffer_size = self.args["buffer_size"]

        print(f"Initializing buffer with {buffer_size} observations...")

        self.buffer = ReplayBuffer( self.env, buffer_size)
        random_experience = self.buffer.get_experience_from_random_policy(total_steps=buffer_size, nstep=self.args["n_step"])
        for action_features, reward, obs2 in random_experience:
            self.buffer.add(action_features, reward, obs2)



        print("Done.")

    def train(self, seconds=None, max_steps=None, max_eps=100000, early_stopping=False, pth_path=None):
        self.initializeBuffer()
        if self.training_start is None:
            self.training_start = time.time()
            self.last_best = 0

        instance, n, k = self.env.get_instance_info()
        csv_path = f"./results/training/{instance}-{n}-{k}-partial.csv"
        saved = False

        steps, eps = 1, 1
        epsilon_step = (self.args["first_epsilon"] - self.args["last_epsilon"])
        epsilon_step /= self.args["epsilon_decay_steps"]

        obs = self.env.reset()

        rewards = 0
        all_rewards = []
        last_steps = []

        while not (max_eps is not None and eps >= max_eps):
            a = self.get_action(obs, self.epsilon)
            last_steps.append(obs[a])

            obs2, reward, done, info = self.env.step(a)
            rewards += reward

            if self.args["exp_replay"]:
                if done:
                    for j in range(len(last_steps)):
                        self.buffer.add(self.env.context.compute_features(last_steps[j]), -len(last_steps) + j, None)
                    last_steps = []
                else:
                    if len(last_steps) >= self.args["n_step"]:
                        aux = self.env.context.compute_feature_of_list(obs2)
                        self.buffer.add(self.env.context.compute_features(last_steps[0]), -self.args["n_step"], aux)
                    last_steps = last_steps[len(last_steps) - self.args["n_step"] + 1:]
                self.batch_update()
            else:
                self.update(obs, a, reward, obs2)

            if done:
                eps += 1

                all_rewards.append(-rewards)
                print(f"Epsode: {eps} - Acumulated Reward: {-rewards} - Acumulated: {np.mean(all_rewards[-32:])} - Epsilon: {self.epsilon}")
                rewards = 0
                instance = (self.env.info["problem"], self.env.info["n"], self.env.info["k"])
                info.update({
                    "training time": time.time() - self.training_start,
                    "training steps": self.training_steps,
                    "instance": instance,
                    "loss": self.model.current_loss(),
                    })
                self.training_data.append(info)
                obs = self.env.reset()
            else:
                obs = obs2

            if eps % self.freq_save == 0 and not saved and pth_path is not None:
                if len(all_rewards) > 1000:
                    all_rewards = all_rewards[100:]

                saved = True
                DQN.save(self, f"{pth_path[:-4]}-{eps}.pth", partial=True)

                with open(csv_path, 'a') as f:
                    writer = csv.writer(f)
                    for i in range(self.freq_save-1):
                        writer.writerow([steps, all_rewards[-self.freq_save+1+i]])

                print("Partial Saved!")
            else:
                if eps % self.freq_save != 0:
                    saved = False

            steps += 1
            self.training_steps += 1

            if seconds is not None and time.time() - self.training_start > seconds:
                break

            if max_steps is not None and not early_stopping and steps >= max_steps:
                break

            if max_eps is not None and eps >= max_eps:
                break

            if max_steps is not None and self.training_steps > max_steps and (self.training_steps - self.last_best) / self.training_steps > 0.33:
                print("Converged since steps are", self.training_steps, "and max_steps is", max_steps, "and last best was", self.last_best)
                self.converged = True

            if early_stopping and self.converged:
                print("Converged!")
                break

            if self.epsilon > self.args["last_epsilon"] + 1e-10:
                self.epsilon -= epsilon_step

        self.trained = True
        return obs

    def get_action(self, s, epsilon, env=None):
        """ Gets epsilon-greedy action using self.model """
        if env is None:
            env = self.env
        if np.random.rand() <= epsilon:
            return np.random.randint(len(s))
        else:
            features = env.actions_to_features(s)
            return self.model.best(features)

    def update(self, obs, action, reward, obs2):
        """ Gets epsilon-greedy action using self.model """
        if self.target is not None:
            value = self.target.eval(obs2)
        else:
            value = self.model.eval(obs2)

        self.model.single_update(obs[action], value+reward)

        if self.verbose:
            print("Single update. Value:", value+reward)

    def batch_update(self):
        action_featuress, rewards, obss2 = self.buffer.sample(self.args["batch_size"])
        if self.target is not None:
            values = self.target.eval_batch(obss2)
        else:
            values = self.model.eval_batch(obss2)

        if self.verbose:
            print("Batch update. Values:", rewards+values)

        self.model.batch_update(np.array(action_featuress), rewards + values)


    @staticmethod
    def save(agent, path, partial=False):
        if partial:
            path = path[:-4] + f"-partial.pth"
        agent.model.save(path)

import random
class TRPO(object):
    def __init__(self, env, args):
        self.env = env
        self.args = args

        self.state_size = env.observation_space_dim
        self.num_actions = 1#len(env.action_space)

        ### Abstraerlo en una clase
        self.buffer_max_size = 500   # Sacarlo de args
        self.batch_size = 10 # Sacarlo de args
        self.buffer_i = 0
        self.buffer = [None] * self.buffer_max_size

        # delta, maximum KL divergence
        self.max_d_kl = 0.01

        self.actor = nn.Sequential(nn.Linear(self.state_size, args['actor_size']),
                                   nn.ReLU(),
                                   nn.Linear(args['actor_size'], self.num_actions),
                                   nn.Softmax(dim=1))

        # Critic takes a state and returns its values
        self.critic = nn.Sequential(nn.Linear(self.state_size, args['critic_size']),
                                    nn.ReLU(),
                                    nn.Linear(args['critic_size'], 1))
        self.critic_optimizer = Adam(self.critic.parameters(), lr=args['learning_rate'])

    def add(self, state, action, reward, next_state):
        i = self.buffer_i % self.buffer_max_size
        self.buffer[i] = (state, action, reward, next_state)
        self.buffer_i += 1

    def sample(self):
        idexes = [random.randint(0, len(self.buffer) - 1) for _ in range(self.batch_size)]
        states, actions, rewards, next_states = [], [], [], []
        for i in idexes:
            state, action, reward, next_state = self.buffer[i]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)

        return states, actions, rewards, next_states

    def is_buffer_full(self):
        return self.buffer_i > self.buffer_max_size


    def train(self, episodes=100, freq_update=10):
        all_rewards = []

        for eps in range(episodes):

            total_reward_in_batch = []

            state = self.env.reset()
            done = False
            r = 0

            samples = []

            while not done:
                with torch.no_grad():
                    action = self.predict(state)

                next_state, reward, done, info = self.env.step(action)

                # Collect samples
                #samples.append((state, action, reward, next_state))
                for s in state:
                    self.add(s, action, reward, next_state)

                state = next_state
                r += reward

            # Transpose our samples
            #states, actions, rewards, next_states = zip(*samples)

            #states = torch.stack([torch.from_numpy(np.array(state)) for state in states], dim=0).float()
            #next_states = torch.stack([torch.from_numpy(np.array(state)) for state in next_states], dim=0).float()
            #states = [torch.from_numpy(np.array(state)) for state in states]
            #next_states = [torch.from_numpy(np.array(state)) for state in next_states]
            #actions = torch.as_tensor(actions).unsqueeze(1)
            #rewards = torch.as_tensor(rewards).unsqueeze(1)

            all_rewards.append(r)
            if not self.is_buffer_full():
                continue

            states, actions, rewards, next_states =  self.sample()

            # el baffer tiene es un diccionario de listas, cada elemento de la lista es una posible transicion en un estado
            #for s, a, r, ns in zip(states, actions, rewards, next_states):
                #buffer = []
                #buffer.append({'state': states, 'action': actions, 'reward': rewards, 'next_states': next_states})
                #self.update_agent(buffer)
            self.update_agent([{'state': torch.as_tensor(states), 'action': actions, 'reward': rewards, 'next_states': torch.as_tensor(next_states[random.randint(0, len(next_states)-1)])}])

            print(f"Episode: {eps} - Acumulated Reward: {-r} - Acumulated: {np.mean(all_rewards[-32:])}")

    def predict(self, state):
        state = torch.tensor(state).float()  # Turn state into a batch with a single element
        dist = Categorical(self.actor(state))  # Create a distribution from probabilities for actions
        return dist.sample().argmax().item()

    def update_critic(self, advantages):
        loss = .5 * (advantages ** 2).mean()  # MSE
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

    def update_agent(self, buffer):
        actions = torch.cat([torch.as_tensor(r['action']) for r in buffer], dim=0).flatten()

        advantages = [self.estimate_advantages(r['state'], r['next_states'][-1], r['reward']) for r in buffer]
        advantages = torch.cat(advantages, dim=0).flatten()

        # Normalize advantages to reduce skewness and improve convergence
        advantages = (advantages - advantages.mean()) / advantages.std()

        self.update_critic(advantages)

        states = torch.cat([r['state'] for r in buffer], dim=0)
        distribution = self.actor(states)
        distribution = torch.distributions.utils.clamp_probs(distribution)
        probabilities = distribution[range(distribution.shape[0]), actions]

        # Now we have all the data we need for the algorithm

        # We will calculate the gradient wrt to the new probabilities (surrogate function),
        # so second probabilities should be treated as a constant
        L = self.surrogate_loss(probabilities, probabilities.detach(), advantages)
        KL = self.kl_div(distribution, distribution)

        parameters = list(self.actor.parameters())

        g = self.flat_grad(L, parameters, retain_graph=True)
        d_kl = self.flat_grad(KL, parameters,
                              create_graph=True)  # Create graph, because we will call backward() on it (for HVP)

        def HVP(v):
            return self.flat_grad(d_kl @ v, parameters, retain_graph=True)

        search_dir = self.conjugate_gradient(HVP, g)
        max_length = torch.sqrt(2 * self.max_d_kl / (search_dir @ HVP(search_dir)))
        max_step = max_length * search_dir

        def criterion(step):
            self.apply_update(step)

            with torch.no_grad():
                distribution_new = self.actor(states)
                distribution_new = torch.distributions.utils.clamp_probs(distribution_new)
                probabilities_new = distribution_new[range(distribution_new.shape[0]), actions]

                L_new = self.surrogate_loss(probabilities_new, probabilities, advantages)
                KL_new = self.kl_div(distribution, distribution_new)

            L_improvement = L_new - L

            if L_improvement > 0 and KL_new <= self.max_d_kl:
                return True

            self.apply_update(-step)
            return False

        i = 0
        while not criterion((0.9 ** i) * max_step) and i < 10:
            i += 1

    def estimate_advantages(self, states, last_state, rewards):
        values = self.critic(states)
        last_value = self.critic(last_state.unsqueeze(0))
        rewards = torch.tensor(rewards)
        next_values = torch.zeros_like(rewards)
        for i in reversed(range(rewards.shape[0])):
            last_value = next_values[i] = rewards[i] + 0.99 * last_value
        advantages = next_values - values
        return advantages

    def surrogate_loss(self, new_probabilities, old_probabilities, advantages):
        return (new_probabilities / old_probabilities * advantages).mean()

    def kl_div(self, p, q):
        p = p.detach()
        return (p * (p.log() - q.log())).sum(-1).mean()

    def flat_grad(self, y, x, retain_graph=False, create_graph=False):
        if create_graph:
            retain_graph = True

        g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
        g = torch.cat([t.view(-1) for t in g])
        return g

    def conjugate_gradient(self, A, b, delta=0., max_iterations=10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()

        i = 0
        while i < max_iterations:
            AVP = A(p)

            dot_old = r @ r
            alpha = dot_old / (p @ AVP)

            x_new = x + alpha * p

            if (x - x_new).norm() <= delta:
                return x_new

            i += 1
            r = r - alpha * AVP

            beta = (r @ r) / dot_old
            p = r + beta * p

            x = x_new
        return x

    def apply_update(self, grad_flattened):
        n = 0
        for p in self.actor.parameters():
            numel = p.numel()
            g = grad_flattened[n:n + numel].view(p.shape)
            p.data += g
            n += numel



############################################
############################################
############################################
#https://github.com/ericyangyu/PPO-for-Beginners/blob/master/part4/ppo_for_beginners/ppo_optimized.py
############################################



import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch.nn.functional as F

class FeedForwardNN(nn.Module):
	"""
		A standard in_dim-64-64-out_dim Feed Forward Neural Network.
	"""
	def __init__(self, in_dim, out_dim):
		"""
			Initialize the network and set up the layers.

			Parameters:
				in_dim - input dimensions as an int
				out_dim - output dimensions as an int

			Return:
				None
		"""
		super(FeedForwardNN, self).__init__()

		self.layer1 = nn.Linear(in_dim, 64)
		self.layer2 = nn.Linear(64, 64)
		self.layer3 = nn.Linear(64, out_dim)

	def forward(self, obs):
		"""
			Runs a forward pass on the neural network.

			Parameters:
				obs - observation to pass as input

			Return:
				output - the output of our forward pass
		"""
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)

		activation1 = F.relu(self.layer1(obs))
		activation2 = F.relu(self.layer2(activation1))
		output = self.layer3(activation2)

		return output

class PPO:
    """
        This is the PPO class we will use as our model in main.py
    """

    def __init__(self, env, **hyperparameters):
        """
            Initializes the PPO model, including hyperparameters.

            Parameters:
                policy_class - the policy class to use for our actor/critic networks.
                env - the environment to train on.
                hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

            Returns:
                None
        """
        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)

        # Extract environment information
        self.env = env

        self.obs_dim = env.observation_space_dim
        self.act_dim = 1

        policy_class = FeedForwardNN
        # Initialize actor and critic networks
        self.actor = policy_class(self.obs_dim, self.act_dim)  # ALG STEP 1
        self.critic = policy_class(self.obs_dim, 1)

        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,  # timesteps so far
            'i_so_far': 0,  # iterations so far
            'batch_lens': [],  # episodic lengths in batch
            'batch_rews': [],  # episodic returns in batch
            'actor_losses': [],  # losses of actor network in current iteration
            'lr': 0,
        }

    def learn(self, total_timesteps):
        """
            Train the actor and critic networks. Here is where the main PPO algorithm resides.

            Parameters:
                total_timesteps - the total number of timesteps to train for

            Return:
                None
        """
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far
        while t_so_far < total_timesteps:  # ALG STEP 2
            # Autobots, roll out (just kidding, we're collecting our batch simulations here)
            batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones = self.rollout()  # ALG STEP 3

            # Calculate advantage using GAE
            A_k = self.calculate_gae(batch_rews, batch_vals, batch_dones)
            V = self.critic(batch_obs).squeeze()
            batch_rtgs = A_k + V.detach()

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
            # isn't theoretically necessary, but in practice it decreases the variance of
            # our advantages and makes convergence much more stable and faster. I added this because
            # solving some environments was too unstable without it.
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            step = batch_obs.size(0)
            inds = np.arange(step)
            minibatch_size = step // self.num_minibatches
            loss = []

            for _ in range(self.n_updates_per_iteration):  # ALG STEP 6 & 7
                # Learning Rate Annealing
                frac = (t_so_far - 1.0) / total_timesteps
                new_lr = self.lr * (1.0 - frac)

                # Make sure learning rate doesn't go below 0
                new_lr = max(new_lr, 0.0)
                self.actor_optim.param_groups[0]["lr"] = new_lr
                self.critic_optim.param_groups[0]["lr"] = new_lr
                # Log learning rate
                self.logger['lr'] = new_lr

                # Mini-batch Update
                np.random.shuffle(inds)  # Shuffling the index
                for start in range(0, step, minibatch_size):
                    end = start + minibatch_size
                    idx = inds[start:end]
                    # Extract data at the sampled indices
                    mini_obs = batch_obs[idx]
                    mini_acts = batch_acts[idx]
                    mini_log_prob = batch_log_probs[idx]
                    mini_advantage = A_k[idx]
                    mini_rtgs = batch_rtgs[idx]

                    # Calculate V_phi and pi_theta(a_t | s_t) and entropy
                    V, curr_log_probs, entropy = self.evaluate(mini_obs, mini_acts)

                    # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                    # NOTE: we just subtract the logs, which is the same as
                    # dividing the values and then canceling the log with e^log.
                    # For why we use log probabilities instead of actual probabilities,
                    # here's a great explanation:
                    # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
                    # TL;DR makes gradient descent easier behind the scenes.
                    logratios = curr_log_probs - mini_log_prob
                    ratios = torch.exp(logratios)
                    approx_kl = ((ratios - 1) - logratios).mean()

                    # Calculate surrogate losses.
                    surr1 = ratios * mini_advantage
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * mini_advantage

                    # Calculate actor and critic losses.
                    # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                    # the performance function, but Adam minimizes the loss. So minimizing the negative
                    # performance function maximizes it.
                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    critic_loss = nn.MSELoss()(V, mini_rtgs)

                    # Entropy Regularization
                    entropy_loss = entropy.mean()
                    # Discount entropy loss by given coefficient
                    actor_loss = actor_loss - self.ent_coef * entropy_loss

                    # Calculate gradients and perform backward propagation for actor network
                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    # Gradient Clipping with given threshold
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optim.step()

                    # Calculate gradients and perform backward propagation for critic network
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optim.step()

                    loss.append(actor_loss.detach())
                # Approximating KL Divergence
                if approx_kl > self.target_kl:
                    break  # if kl aboves threshold
            # Log actor loss
            avg_loss = sum(loss) / len(loss)
            self.logger['actor_losses'].append(avg_loss)

            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')

    def calculate_gae(self, rewards, values, dones):
        batch_advantages = []  # List to store computed advantages for each timestep

        # Iterate over each episode's rewards, values, and done flags
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []  # List to store advantages for the current episode
            last_advantage = 0  # Initialize the last computed advantage

            # Calculate episode advantage in reverse order (from last timestep to first)
            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    # Calculate the temporal difference (TD) error for the current timestep
                    delta = ep_rews[t] + self.gamma * ep_vals[t + 1] * (1 - ep_dones[t + 1]) - ep_vals[t]
                else:
                    # Special case at the boundary (last timestep)
                    delta = ep_rews[t] - ep_vals[t]

                # Calculate Generalized Advantage Estimation (GAE) for the current timestep
                advantage = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage  # Update the last advantage for the next timestep
                advantages.insert(0, advantage)  # Insert advantage at the beginning of the list

            # Extend the batch_advantages list with advantages computed for the current episode
            batch_advantages.extend(advantages)

        # Convert the batch_advantages list to a PyTorch tensor of type float
        return torch.tensor(batch_advantages, dtype=torch.float)

    def rollout(self):

        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        batch_vals = []
        batch_dones = []

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []
        ep_vals = []
        ep_dones = []
        t = 0  # Keeps track of how many timesteps we've run so far this batch

        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.timesteps_per_batch:
            ep_rews = []  # rewards collected per episode
            ep_vals = []  # state values collected per episode
            ep_dones = []  # done flag collected per episode
            # Reset the environment. Note that obs is short for observation.
            obs, _ = self.env.reset()
            # Initially, the game is not done
            done = False

            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.max_timesteps_per_episode):
                # Track done flag of the current state
                ep_dones.append(done)

                t += 1  # Increment timesteps ran this batch so far

                # Track observations in this batch
                batch_obs.append(obs)
                obs = torch.Tensor(obs)

                # Calculate action and make a step in the env.
                # Note that rew is short for reward.
                action, log_prob = self.get_action(obs)
                val = self.critic(obs)

                obs, rew, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                # Track recent reward, action, and action log probability
                ep_rews.append(rew)
                ep_vals.append(val.flatten())
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                # If the environment tells us the episode is terminated, break
                if done:
                    break

            # Track episodic lengths, rewards, state values, and done flags
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)
        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).flatten()

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        # Here, we return the batch_rews instead of batch_rtgs for later calculation of GAE
        return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones

    def get_action(self, obs):
        """
            Queries an action from the actor network, should be called from rollout.

            Parameters:
                obs - the observation at the current timestep

            Return:
                action - the action to take, as a numpy array
                log_prob - the log probability of the selected action in the distribution
        """
        # Query the actor network for a mean action
        #obs = torch.tensor(obs, dtype=torch.float)
        mean = self.actor(obs)

        # Create a distribution with the mean action and std from the covariance matrix above.
        # For more information on how this distribution works, check out Andrew Ng's lecture on it:
        # https://www.youtube.com/watch?v=JjB58InuTqM
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution
        action = dist.sample()

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # If we're testing, just return the deterministic action. Sampling should only be for training
        # as our "exploration" factor.
        if self.deterministic:
            return mean.detach().numpy(), 1

        # Return the sampled action and the log probability of that action in our distribution
        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        """
            Estimate the values of each observation, and the log probs of
            each action in the most recent batch with the most recent
            iteration of the actor network. Should be called from learn.

            Parameters:
                batch_obs - the observations from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of observation)
                batch_acts - the actions from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of action)
                batch_rtgs - the rewards-to-go calculated in the most recently collected
                                batch as a tensor. Shape: (number of timesteps in batch)
        """
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        # if batch_obs.size(0) == 1:
        #     V = self.critic(batch_obs)
        # else:
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs, dist.entropy()

    def _init_hyperparameters(self, hyperparameters):
        """
            Initialize default and custom values for hyperparameters

            Parameters:
                hyperparameters - the extra arguments included when creating the PPO model, should only include
                                    hyperparameters defined below with custom values.

            Return:
                None
        """
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.timesteps_per_batch = 4800  # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 1600  # Max number of timesteps per episode
        self.n_updates_per_iteration = 5  # Number of times to update actor/critic per iteration
        self.lr = 0.005  # Learning rate of actor optimizer
        self.gamma = 0.95  # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2  # Recommended 0.2, helps define the threshold to clip the ratio during SGA
        self.lam = 0.98  # Lambda Parameter for GAE
        self.num_minibatches = 6  # Number of mini-batches for Mini-batch Update
        self.ent_coef = 0  # Entropy coefficient for Entropy Regularization
        self.target_kl = 0.02  # KL Divergence threshold
        self.max_grad_norm = 0.5  # Gradient Clipping threshold

        # Miscellaneous parameters
        self.render = False  # If we should render during rollout
        self.save_freq = 10  # How often we save in number of iterations
        self.deterministic = False  # If we're testing, don't sample actions
        self.seed = None  # Sets the seed of our program, used for reproducibility of results

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert (type(self.seed) == int)

            # Set the seed
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _log_summary(self):
        """
            Print to stdout what we've logged so far in the most recent batch.

            Parameters:
                None

            Return:
                None
        """
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        lr = self.logger['lr']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"Learning rate: {lr}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []
