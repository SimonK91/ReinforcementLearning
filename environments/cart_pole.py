import gym
import time
import pandas, numpy
import matplotlib.pyplot as plt


def discretize(value):
    return numpy.round(numpy.sign(value) * ((numpy.abs(value) * 17) ** 0.5))


class Env:
    def __init__(self):
        self.__env = gym.make("CartPole-v1")
        self.action_space = self.__env.action_space
        self.fig = None
        self.__actionNames = ["Left ", "Right"]
        self.reset()

    def reset(self):
        self.__last_action = -1
        self.__score = 0
        return tuple(discretize(self.__env.reset()))

    def step(self, action):
        self.__last_action = action
        s, r, d, p = self.__env.step(action)
        self.__score += r
        # r = -1 if d else 0
        return tuple(discretize(s)), r, d, p

    def render(self):
        self.__env.render()

    def plot(self, policy):
        policy.Play()
        return
        if self.fig == None:
            self.fig = plt.figure()
            self.subp = [None, None, None]
            self.subp[0] = self.fig.add_subplot(311)

            self.subp[1] = self.fig.add_subplot(312)

            self.subp[2] = self.fig.add_subplot(313)

        env = self.__env.render(mode="ansi").split()
        env = pandas.factorize(env)[0]
        env[self.__env.start_state_index] = 4
        env = env.reshape(self.__env.shape)
        values = numpy.zeros(self.__env.shape)

        self.subp[0].cla()
        self.subp[0].imshow(env, interpolation="none")

        # self.subp[1].table(env)

        self.subp[1].cla()
        self.subp[1].grid()
        self.subp[1].set_xlim(0, self.__env.shape[1])
        self.subp[1].set_ylim(0, self.__env.shape[0])
        self.subp[1].set_xticks(numpy.arange(self.__env.shape[1]))
        self.subp[1].set_yticks(numpy.arange(self.__env.shape[0]))
        for y in range(4):
            for x in range(12):
                c = x + y * 12
                values[y][x] = policy.GetValue(c)
                self.subp[1].text(
                    x + 0.5,
                    3.5 - y,
                    self.__actionNames[policy.GetPolicy(c) + 1],
                    va="center",
                    ha="center",
                )

        self.subp[2].cla()
        self.subp[2].imshow(values, interpolation="none")

        plt.ion()
        plt.show()
        plt.pause(0.01)