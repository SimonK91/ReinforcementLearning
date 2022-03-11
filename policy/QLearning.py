import numpy
from .PolicyBase import ExploringBase
from .containers.QTable import QTable


class QLearning(ExploringBase):
    def __init__(self, environment, alpha=0.5, gamma=0.7, epsilon=0.05, max_steps=-1):
        ExploringBase.__init__(self, environment, epsilon)
        self.returns = {}

        self.gamma = gamma
        self.alpha = alpha
        self.max_steps = max_steps

    def DoTraining(self):
        S = self._env.reset()
        A = self.GetAction(S)
        rem_steps = self.max_steps
        done = False
        while not done:
            rem_steps -= 1
            NS, R, done, _ = self._env.step(A)
            if rem_steps == 0:
                done = True
                R -= 100
            NA = self.GetAction(NS)
            efv = self.GetValue(NS)
            V = self.GetValue(S, A)
            error = R + self.gamma * efv - V
            self.SetValue(S, A, V + self.alpha * error)
            self.SetPolicy(S, self.BestAction(S))
            S = NS
            A = NA
