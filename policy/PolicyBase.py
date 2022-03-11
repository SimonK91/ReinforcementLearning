import numpy
import random
import time

from .containers.QTable import QTable


class Base:
    def __init__(self, environment):
        self._env = environment
        self.__policy = {}
        self.__q = QTable()

    def Play(self, render=True, max_steps=-1):
        StateActions = []
        Rewards = []
        S = self._env.reset()
        if render:
            self._env.render()
        score = 0
        done = False
        while max_steps != 0:
            max_steps -= 1
            A = self.GetAction(S)
            StateActions.append((S, A))
            S, R, done, _ = self._env.step(A)
            score += R
            Rewards.append(R)
            if render:
                self._env.render()
            if done:
                return (score, StateActions, Rewards, True)
        return (score, StateActions, Rewards, False)

    def Train(self, duration):
        iterations = 0
        startTime = time.time()
        while time.time() - startTime < duration:
            iterations += 1
            self.DoTraining()
        return iterations

    def GetAction(self, state):
        return self.GetPolicy(state)

    def SetValue(self, state, action, value):
        self.__q.SetValue(state, action, value)

    def GetValue(self, state, action=None):
        return self.__q.GetValue(state, action)

    def BestAction(self, state):
        return self.__q.BestAction(state)

    def GetPolicy(self, state):
        if not state in self.__policy:
            self.__policy[state] = self._env.action_space.sample()
        return self.__policy[state]

    def SetPolicy(self, state, action):
        self.__policy[state] = action

    def GetTable(self):
        return self.__q._QTable__table

    def GetEncounters(self):
        return self.__q._QTable__counter


class ExploringBase(Base):
    def __init__(self, environment, epsilon):
        Base.__init__(self, environment)
        self.__epsilon = epsilon

    def GetAction(self, state):
        if random.random() < self.__epsilon:  # Random action
            return self._env.action_space.sample()
        return self.GetPolicy(state)
