import gym
import tensorflow as tf
import numpy as np
import random
import pandas as pd
from collections import deque

env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env, 'rl-cartpole-experiment', force=True)  

# Constants defining our neural network
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

dis = 0.9
REPLAY_MEMORY = 2000


class DQN:

	def __init__(self, session, input_size, output_size, name="main"):
		self.session = session
		self.input_size = input_size
		self.output_size = output_size
		self.net_name = name
		self._build_network()

	def _build_network(self, h_size=10, l_rate=1e-1):

		with tf.variable_scope(self.net_name):
			
			# Input Layer
			self._X = tf.placeholder(tf.float32, [None, self.input_size], name='input_x')
			
			# First Hidden Layer
			W1 = tf.get_variable("W1", shape=[self.input_size, h_size], initializer=tf.contrib.layers.xavier_initializer())
			layer1 = tf.nn.tanh(tf.matmul(self._X, W1))

			# Second Hidden Layer
			W2 = tf.get_variable("W2", shape=[h_size, self.output_size], initializer=tf.contrib.layers.xavier_initializer())

			# Output Layer - Q Prediction
			self._Qpred = tf.matmul(layer1, W2)

		# We need to define the parts of the network needed for learning a policy
		self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)

		# Loss function
		self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))

		# Learning
		self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)

	def predict(self, state):
		x = np.reshape(state, [1, self.input_size])
		return self.session.run(self._Qpred, feed_dict={self._X: x})

	def update(self, x_stack, y_stack):
		return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})


def replay_train(mainDQN, targetDQN, train_batch):
	x_stack = np.empty(0).reshape(0, input_size)
	y_stack = np.empty(0).reshape(0, output_size)

	# Get stored information from the buffer
	for state, action, reward, next_state, done in train_batch:
		Q = mainDQN.predict(state)

		if done: # Game over?
			Q[0, action] = reward
		else:
			# get target from target DQN (Q')
			Q[0, action] = reward + dis * np.max(targetDQN.predict(next_state))

		y_stack = np.vstack([y_stack, Q])
		x_stack = np.vstack([x_stack, state])

	# Train our network using target and predicted Q values on each episode
	return mainDQN.update(x_stack, y_stack)


def get_copy_var_ops(dest_scope_name="target", src_scope_name="main"):
	# Copy variables src_scope to dest_scope
	op_holder = []
	
	src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
	dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

	for src_var, dest_var in zip(src_vars, dest_vars):
		op_holder.append(dest_var.assign(src_var.value()))

	return op_holder


def bot_play(mainDQN):
	# See our trained network in action
	s = env.reset()
	reward_sum = 0
	while True:
		env.render()
		a = np.argmax(mainDQN.predict(s))
		s, reward, done, _ = env.step(a)
		reward_sum += reward
		if done:
			print("Total score: {}".format(reward_sum))
			break

			
def main():
	max_episodes = 5000 # 5000
	# store the previous observations in replay memory 
	replay_buffer = deque()
	pandas_reward = []
	pandas_loss = []
	pandas_stepcount = []	

	with tf.Session() as sess:
		mainDQN = DQN(sess, input_size, output_size, name="main")
		targetDQN = DQN(sess, input_size, output_size, name="target")
		tf.global_variables_initializer().run()
	
		# initial copy q_net -> target_net
		copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
		sess.run(copy_ops)

		for episode in range(max_episodes):
			e = 1. / ((episode / 10) + 1)
			done = False
			step_count = 0 
			state = env.reset()
			reward_sum = 0
			
			while not done:
				if np.random.rand(1) < e:
					action = env.action_space.sample()
				else:
					# Choose an action by greedily from the Q-network
					action = np.argmax(mainDQN.predict(state))

				# Get new state and reward from environment
				next_state, reward, done, _ = env.step(action)

				if done: # Penalty
					reward = -100
				
				reward_sum += reward # accumulated rewards

				# Save the experience to our buffer
				replay_buffer.append((state, action, reward, next_state, done))
				if len(replay_buffer) > REPLAY_MEMORY:
					replay_buffer.popleft()

				state = next_state
				step_count += 1
				if step_count > 10000: # Good enough. Let's move on
					break
			
			# Game over
			pandas_reward.append(reward_sum)
			pandas_stepcount.append(step_count)			

			print("Episode: {} steps: {}".format(episode, step_count))

			#if step_count > 10000:
			#	pass
				# break

			if episode % 10 == 1: # train every 10 episode
				# Get a random batch of experiences
				for _ in range(50):
					minibatch = random.sample(replay_buffer, 10)
					loss, _ = replay_train(mainDQN, targetDQN, minibatch)

				print("--------------------> Loss: ", loss)
				pandas_loss.append(loss)
				# copy q_net -> target_net
				sess.run(copy_ops)


		# END OF EPISODES
		### SAVE PANDAS (EXCEL) FOR VISUALIZATION
		df1 = pd.DataFrame(pandas_loss)
		df2 = pd.DataFrame(pandas_reward)
		df3 = pd.DataFrame(pandas_stepcount)
		
		df1.to_csv('loss.csv')
		df2.to_csv('reward.csv')
		df3.to_csv('stepcount.csv')
		
		bot_play(mainDQN)


if __name__ == "__main__":
	main()























