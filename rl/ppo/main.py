# coding:utf-8

import gym
import numpy as np
from ppo import Agent

if __name__ == "__main__":

    env = gym.make("CartPole-v0")
    N = 20
    bsz = 5
    n_epochs = 4
    agent = Agent(
        n_actions=env.action_space.n,
        input_dim=env.observation_space.shape,
        hidden_dim=256,
        batch_size=bsz,
    )
    n_games = 300

    best_score = env.reward_range[0]
    score_history = []
    
    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        obs = env.reset()[0]
        done = False
        score = 0
        while not done:
            log_prob, action, value = agent.choose_actions(obs)
            next_obs, reward, done, _, _ = env.step(action)
            n_steps +=1
            score +=reward
            agent.remember(obs, action, log_prob, value, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters +=1
            obs = next_obs
        
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        
        print(f"N_game: {i}, score: {score:.2f}, avg_score: {avg_score:.2f}, time_steps: {n_steps}, learning_steps: {learn_iters}")