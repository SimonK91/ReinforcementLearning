# Policy Iteration class
import numpy
from .PolicyBase import Base


class PolicyIteration(Base):
    def __init__(self, environment, theta=0.01, gamma=0.7):
        Base.__init__(self, environment)
        self.theta = theta
        self.gamma = gamma

    def DoTraining(self):
        stabilized = False
        while not stabilized:
            self.Evaluate()
            stabilized = self.Improve()

    def Evaluate(self):
        delta = self.theta + 1
        while delta > self.theta:
            delta = 0
            for state in range(self._env.nS):
                old_value = self.values[state]
                action = self.GetPolicy(state)
                value = 0
                for prob, state_t1, reward, done in self._env.P[state][action]:
                    if done:
                        value = value + prob * reward
                    else:
                        value = value + prob * (
                            reward + self.gamma * self.values[state_t1]
                        )
                self.values[state] = value
                delta = max(delta, abs(old_value - value))

    def Improve(self):
        stabilized = True
        for state in range(self._env.nS):
            old_action = self.GetPolicy(state)
            best_value = None
            for a in range(self._env.nA):
                value = 0
                for prob, state_t1, reward, done in self._env.P[state][a]:
                    if done:
                        value = value + prob * (reward)
                    else:
                        value = value + prob * (
                            reward + self.gamma * self.values[state_t1]
                        )
                if best_value == None or value > best_value:
                    self.SetPolicy(state, a)
                    best_value = value
            if old_action != self.GetPolicy(state):
                stabilized = False
        return stabilized
