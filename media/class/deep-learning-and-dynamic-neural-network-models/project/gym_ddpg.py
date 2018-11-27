import filter_env
from ddpg import *
import gc
gc.enable()

ENV_NAME = 'BipedalWalker-v2' #'InvertedPendulum-v1'
EPISODES = 100000
TEST = 10

OUT_DIR = 'DDPG-bipedalwalker-experiment'

# env.spec.timestep_limit = 1600


def main():
    env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    agent = DDPG(env)
    env = gym.wrappers.Monitor(env, OUT_DIR, force=True)
    #env.monitor.start('experiments/' + ENV_NAME,force=True)
    print(env.spec.timestep_limit)

    for episode in xrange(EPISODES):
        state = env.reset()
        print "episode:",episode
        # Train
        for step in xrange(env.spec.timestep_limit):
            action = agent.noise_action(state)
            next_state,reward,done,_ = env.step(action)
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            if done:
                break
        # Testing:
        if episode % 50 == 0 and episode > 40: # 100
			total_reward = 0
			for i in xrange(TEST):
				state = env.reset()
				for j in xrange(env.spec.timestep_limit):
					#env.render()
					action = agent.action(state) # direct action for test
					state,reward,done,_ = env.step(action)
					total_reward += reward
					if done:
						break
			ave_reward = total_reward/TEST
			print 'episode: ',episode,'Evaluation Average Reward:',ave_reward, 'total_reward', total_reward, 'TEST', TEST
    env.monitor.close()

if __name__ == '__main__':
    main()
