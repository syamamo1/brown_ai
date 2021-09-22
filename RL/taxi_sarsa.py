import gym
import numpy as np
import random
import tabular_sarsa as Tabular_SARSA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys


class SARSA(Tabular_SARSA.Tabular_SARSA):
    def __init__(self):
        super(SARSA, self).__init__()

    # TODO
    def learn_policy(
        self, env, gamma, learning_rate, epsilon, lambda_value, num_episodes
    ):
        """
        Implement Sarsa algorithm to update qtable and learning policy.
        Input:
            all parameters

        Output:
            This function returns the updated qtable, learning policy and the reward after each episode.

        """
        self.alpha = learning_rate
        self.epsilon = 0.15
        self.gamma = gamma
        rewards_each_learning_episode = []
        for i in range(num_episodes):
            state = env.reset()
            action = self.LearningPolicy(state)
            episodic_reward = 0
            while True:
                next_state, reward, done, info = env.step(action)  # take a random action

                "*** Fill in the rest of the algorithm!! ***"
                next_action = self.LearningPolicy(next_state)
                if done is True:  # if you reach the goal state
                    episodic_reward = episodic_reward + reward
                    rewards_each_learning_episode = rewards_each_learning_episode + [episodic_reward]
                    delta = reward - self.qtable[state][action]
                    self.qtable[state][action] = self.qtable[state][action] + learning_rate * delta
                    break

                else:  # not at goal state
                    episodic_reward = episodic_reward + reward
                    delta = reward + gamma*self.qtable[next_state][next_action] - self.qtable[state][action]
                    self.qtable[state][action] = self.qtable[state][action] + learning_rate * delta
                    action = next_action
                    state = next_state

        np.save("qvalues_taxi_sarsa_grading.npy", self.qtable)
        np.save("policy_taxi_sarsa_grading.npy", self.policy)

        return self.policy, self.qtable, rewards_each_learning_episode

    def LearningPolicy(self, state, testing=False):
        return Tabular_SARSA.Tabular_SARSA.learningPolicy(self, state, testing=testing)

def plot_rewards(episode_rewards):
    """
    Plots a learning curve for SARSA

    Input:
        episode_rewards: a list of episode rewards

    """
    plt.plot(episode_rewards)
    plt.ylabel("rewards per episode")
    plt.ion()
    plt.savefig("rewards_plot_taxi_sarsa.png")

def render_visualization(learned_policy):
    """
    Renders a taxi problem visualization

    Input:
        learned_policy: the learned SARSA policy to be used by the taxi

    """
    env = gym.make("Taxi-v2")
    state = env.reset()
    env.render()
    while True:
        next_state, reward, done, info = env.step(learned_policy[state, 0])
        env.render()
        print("Reward: {}".format(reward))
        state = next_state
        if done:
            break

def avg_episode_rewards(num_runs):
    """
    Runs the learner algorithms a number of times and averages the episodic rewards
    from all runs for each episode

    Input:
        num_runs: the number of times to run the SARSA learner

    Output:
        episode_rewards: a list of averaged rewards per episode over a num_runs number of times
        learned_policy: the policy learned by the last run of sarsaLearner.learn_policy() to be
        used in problem visualization
    """
    episode_rewards = []
    for i in range(num_runs):
        env = gym.make("Taxi-v2")
        env.reset()
        sarsaLearner = SARSA()
        learned_policy, q_values, single_run_er = sarsaLearner.learn_policy(
            env, 0.95, 0.2, 0.1, 0.1, 1000
        )         # single_run_er is the episodic reward for each run

        if not episode_rewards: # on the first iteration of this loop, episodeRewards will be empty
            episode_rewards = single_run_er
        else: # add this run's ERs to previous runs in order to calculate the average later
            episode_rewards = [episode_rewards[i] + single_run_er[i] for i in range(len(single_run_er))]

    # Get the average over ten runs
    episode_rewards = [er / num_runs for er in episode_rewards]

    return episode_rewards, learned_policy

def test_policy():
    policy = np.load("policy_taxi_sarsa_grading.npy")
    env = gym.make("Taxi-v2")
    env.reset()
    num_episodes = 1000
    rewards_each_test_episode = []
    steps_each_test_episode = []

    for i in range(num_episodes):
        state = env.reset()
        episodic_reward = 0
        steps = 0
        while True:
            action = policy[state][0]
            next_state, reward, done, info = env.step(action)
            steps += 1

            episodic_reward += reward

            state = next_state

            if done:
                break

        rewards_each_test_episode.append(episodic_reward)
        steps_each_test_episode.append(steps)

    print("Rewards each test episode: {}".format(rewards_each_test_episode))
    print("Steps each test episode: {}".format(steps_each_test_episode))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        "*** run without input arguments to learn ***"
        episode_rewards, learned_policy = avg_episode_rewards(10)
        plot_rewards(episode_rewards)
        render_visualization(learned_policy)
    elif sys.argv[1] == "test":
        "*** run to test the saved policy with input argument test ***"
        test_policy()
    else:
        print("unknown input argument")

