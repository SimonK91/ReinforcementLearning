import numpy
from .PolicyBase import ExploringBase
from .containers.QTable import QTable


class MonteCarlo(ExploringBase):
    def __init__(self, environment, gamma=0.7, epsilon=0.05, max_steps=-1):
        ExploringBase.__init__(self, environment, epsilon)
        self.returns = {}

        self.gamma = gamma
        self.max_steps = max_steps

    def DoTraining(self):
        _, SA, R, _ = self.Play(False, self.max_steps)
        self.UpdateStateAction(SA, R)

    def UpdateStateAction(self, StateActions, Rewards):
        G = 0
        # Get last index, since the function is updated in reverse
        idx = len(StateActions) - 1
        while idx >= 0:
            SA = StateActions[idx]
            Rt_1 = Rewards[idx]
            G = self.gamma * G + Rt_1
            if StateActions.index(SA) == idx:  # first-visit check
                if not SA in self.returns:
                    self.returns[SA] = []
                self.returns[SA].append(G)
                St, At = SA
                self.SetValue(St, At, numpy.average(self.returns[SA]))
                self.SetPolicy(St, self.BestAction(St))
            idx -= 1
