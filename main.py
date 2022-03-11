#!/usr/bin/env python3

import time
from policy.Sarsa import Sarsa
from policy.QLearning import QLearning
from policy.MonteCarlo import MonteCarlo
import environments.blackjack
import environments.cliff
import environments.cart_pole
import matplotlib.pyplot as plt

# env = environments.cart_pole.Env()
env = environments.blackjack.Env()

learning_rate = 0.02
discount_factor = 0.999
randomness = 0.05
# pol = MonteCarlo(env, gamma=discount_factor, epsilon=randomness, max_steps=1000)
# pol = Sarsa(env, alpha=learning_rate, gamma=discount_factor, epsilon=randomness)
pol = QLearning(env, alpha=learning_rate, gamma=discount_factor, epsilon=randomness)

training_iterations = 0
while training_iterations < 100000:
    training_iterations += pol.Train(1.05)
    env.plot(pol)

pol.Play()
while input("Press enter to play again ['q' + enter to exit]: ") != "q":
    pol.Play()
