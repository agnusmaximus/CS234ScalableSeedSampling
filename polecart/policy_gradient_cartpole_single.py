import tensorflow as tf
import sys
import time
import numpy as np
import random
import gym
import polecart_env_harder
import math
import matplotlib.pyplot as plt
from gym import envs

#gym.make("polecart_env_harder:polecart_harder-v0")


class ReplayBuffer(object):
    def __init__(self, buffer_size=10000):
        self.states = []
        self.actions = []
        self.advantages = []
        self.transitions = []
        self.update_vals = []
        self.buffer_size = buffer_size

    def add_examples(self, states, actions, advantages, transitions, update_vals):

        # Add examples
        self.states += states
        self.actions += actions
        self.advantages += advantages
        self.transitions += transitions
        self.update_vals += update_vals

        # Drop if more than buffer size
        self.states = self.states[-self.buffer_size:]
        self.actions = self.actions[-self.buffer_size:]
        self.advantages = self.advantages[-self.buffer_size:]
        self.transitions = self.transitions[-self.buffer_size:]
        self.update_vals = self.update_vals[-self.buffer_size:]

    def extract_index_examples(self, arr, indices):
        return [arr[i] for i in indices]

    def sample_examples(self, n):
        buffer_length = len(self.actions)
        replace = n > buffer_length
        random_indices = np.random.choice(buffer_length, n, replace=replace)
        return (self.extract_index_examples(x, random_indices) for x in
                [self.states, self.actions, self.advantages,
                 self.transitions, self.update_vals])

class CartPoleAgentPolicyGradient(object):

    def __init__(self, sess, env, replay_buffer, agent_id, seed_sample=False):
        self.agent_id = agent_id
        self.policy_grad = self.policy_gradient()
        self.value_grad = self.value_gradient()
        self.env = env
        self.sess = sess
        self.seed_sample = seed_sample
        self.replay_buffer = replay_buffer

    def policy_gradient(self):
        with tf.variable_scope("policy_agent_%d" % self.agent_id):
            params = tf.get_variable("policy_parameters",[4,2])
            state = tf.placeholder("float",[None,4])
            actions = tf.placeholder("float",[None,2])
            advantages = tf.placeholder("float",[None,1])
            linear = tf.matmul(state,params)
            probabilities = tf.nn.softmax(linear)
            good_probabilities = tf.reduce_sum(tf.multiply(probabilities, actions),
                                               reduction_indices=[1])
            eligibility = tf.log(good_probabilities) * advantages
            loss = -tf.reduce_sum(eligibility)
            optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
            return probabilities, state, actions, advantages, optimizer

    def value_gradient(self):
        with tf.variable_scope("value_agent_%d" % self.agent_id):
            state = tf.placeholder("float",[None,4])
            newvals = tf.placeholder("float",[None,1])
            w1 = tf.get_variable("w1",[4,10])
            b1 = tf.get_variable("b1",[10])
            h1 = tf.nn.relu(tf.matmul(state,w1) + b1)
            w2 = tf.get_variable("w2",[10,1])
            b2 = tf.get_variable("b2",[1])
            calculated = tf.matmul(h1,w2) + b2
            diffs = calculated-newvals
            loss = tf.nn.l2_loss(diffs)
            optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
            return calculated, state, newvals, optimizer, loss

    def simulate(self, n_episodes=1, render=False, reset_args={}):
        env, policy_grad, value_grad, sess = self.env, self.policy_grad, self.value_grad, self.sess
        pl_calculated, pl_state, pl_actions, pl_advantages, pl_optimizer = policy_grad
        vl_calculated, vl_state, vl_newvals, vl_optimizer, vl_loss = value_grad
        totalreward = 0

        # Shared buffer of examples
        states = []
        actions = []
        advantages = []
        transitions = []
        update_vals = []

        for ii in range(n_episodes):
            observation = env.reset(**reset_args)
            # Run multiple episodes and fill up buffer
            for _ in range(1000):

                # calculate policy
                obs_vector = np.expand_dims(observation, axis=0)
                probs = sess.run(pl_calculated,feed_dict={pl_state: obs_vector})
                action = 0 if random.uniform(0,1) < probs[0][0] else 1

                # record the transition
                states.append(observation)
                actionblank = np.zeros(2)
                actionblank[action] = 1
                actions.append(actionblank)

                # take the action in the environment
                old_observation = observation
                observation, reward, done, info = env.step(action)
                transitions.append((old_observation, action, reward))
                totalreward += reward

                if render:
                    env.render()

                if done:
                    break
            for index, trans in enumerate(transitions):
                obs, action, reward = trans

                # calculate discounted monte-carlo return
                future_reward = 0
                future_transitions = len(transitions) - index
                decrease = 1
                for index2 in range(future_transitions):
                    future_reward += transitions[(index2) + index][2] * decrease
                    decrease = decrease * 0.97
                obs_vector = np.expand_dims(obs, axis=0)
                currentval = sess.run(vl_calculated,feed_dict={vl_state: obs_vector})[0][0]

                # advantage: how much better was this action than normal
                advantages.append(future_reward - currentval)

                # update the value function towards new return
                update_vals.append(future_reward)

        return states, actions, advantages, transitions, update_vals, totalreward / n_episodes

    def simulate_fill_buffer(self, n_episodes_simulation=1):
        # Simulate the episodes
        states, actions, advantages, transitions, update_vals, totalreward = (
            self.simulate(n_episodes_simulation))

        # Update replay buffer
        self.replay_buffer.add_examples(states, actions, advantages, transitions, update_vals)
        return totalreward

    def update(self,
               n_updates=10,
               batch_size=32,
               render=False):

        env, policy_grad, value_grad, sess = self.env, self.policy_grad, self.value_grad, self.sess
        pl_calculated, pl_state, pl_actions, pl_advantages, pl_optimizer = policy_grad
        vl_calculated, vl_state, vl_newvals, vl_optimizer, vl_loss = value_grad

        def batch(iterable, n=1):
            l = len(iterable)
            for ndx in range(0, l, n):
                yield iterable[ndx:min(ndx + n, l)]

        # Do n_updates updates
        for i in range(n_updates):

            # Sample examples
            states, actions, advantages, transitions, update_vals = (
                self.replay_buffer.sample_examples(batch_size))

            advantages = []

            # Re-calculate advantages
            for trans_batch, future_reward_batch in zip(batch(transitions, batch_size),
                                                        batch(update_vals, batch_size)):
                obs_batch = [x[0] for x in trans_batch]
                action_batch = [x[1] for x in trans_batch]
                reward_batch = [x[2] for x in trans_batch]
                obs_vectors = np.stack(obs_batch, axis=0)
                currentvals = sess.run(vl_calculated, feed_dict={vl_state: obs_vectors})

                for ind, future_reward in enumerate(future_reward_batch):
                    advantages.append(future_reward-currentvals[ind][0])

            # update value function
            update_vals_vector = np.expand_dims(update_vals, axis=1)
            sess.run(vl_optimizer, feed_dict={vl_state: states, vl_newvals: update_vals_vector})

            # Update policy function
            advantages_vector = np.expand_dims(advantages, axis=1)
            sess.run(pl_optimizer, feed_dict={pl_state: states, pl_advantages: advantages_vector, pl_actions: actions})

n_agents = int(sys.argv[1])

# Create tf session and env
sess = tf.Session()
#env = gym.make('CartPole-v0')
env = gym.make('polecart_harder-v0')

# Create cartpole agent policy grad
replay_buffer = ReplayBuffer()
agents = [CartPoleAgentPolicyGradient(sess, env, replay_buffer, i) for i in range(n_agents)]
#replay_buffer = ReplayBuffer()
#agents = [CartPoleAgentPolicyGradient(sess, env, ReplayBuffer(), i) for i in range(n_agents)]

# Initialize sess variables
sess.run(tf.initialize_all_variables())

reward_sequences = [[] for i in range(n_agents)]

data = []
iteration = 0

# Train agents
while True:

    rewards = []
    for agent in agents:
        rewards.append(agent.simulate_fill_buffer())
    for agent in agents:
        agent.update()

    # Keep track of running average reward per agent
    for reward_sequence, reward in zip(reward_sequences, rewards):
        reward_sequence.append(reward)
        if len(reward_sequence) >= 100:
            reward_sequence.pop(0)

    # Calculate running averages
    running_averages = [np.mean(x) for x in reward_sequences]

    if iteration % 100 == 0:
        print("Iteration %d avg reward %f" % (iteration, np.max(running_averages)))

    data.append((iteration, np.max(running_averages)))

    if np.max(running_averages) >= 195 or iteration >= 5000:
        break

    iteration += 1

agents[0].simulate(render=True, reset_args={"start_angle" : 90})
#print(data)
