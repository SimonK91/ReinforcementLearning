import gym.envs.toy_text as toy_text
import time
import pandas, numpy
import matplotlib.pyplot as plt


class bcolors:
    ResetAll = "\033[0m"

    Default = "\033[39m"
    Black = "\033[30m"
    Red = "\033[31m"
    Green = "\033[32m"
    Yellow = "\033[33m"
    Blue = "\033[34m"
    Magenta = "\033[35m"
    Cyan = "\033[36m"
    LightGray = "\033[37m"
    DarkGray = "\033[90m"
    LightRed = "\033[91m"
    LightGreen = "\033[92m"
    LightYellow = "\033[93m"
    LightBlue = "\033[94m"
    LightMagenta = "\033[95m"
    LightCyan = "\033[96m"
    White = "\033[97m"

    BG_Default = "\033[49m"
    BG_Black = "\033[40m"
    BG_Red = "\033[41m"
    BG_Green = "\033[42m"
    BG_Yellow = "\033[43m"
    BG_Blue = "\033[44m"
    BG_Magenta = "\033[45m"
    BG_Cyan = "\033[46m"
    BG_LightGray = "\033[47m"
    BG_DarkGray = "\033[100m"
    BG_LightRed = "\033[101m"
    BG_LightGreen = "\033[102m"
    BG_LightYellow = "\033[103m"
    BG_LightBlue = "\033[104m"
    BG_LightMagenta = "\033[105m"
    BG_LightCyan = "\033[106m"
    BG_White = "\033[107m"


class Env:
    def __init__(self):
        self.__env = toy_text.CliffWalkingEnv()
        self.action_space = self.__env.action_space
        self.fig = None
        self.__actionNames = ["", "Up   ", "Right", "Down ", "Left "]
        self.reset()

    def reset(self):
        self.__last_action = -1
        self.__score = 0
        return self.__env.reset()

    def step(self, action):
        self.__last_action = action
        s, r, d, p = self.__env.step(action)
        self.__score += r
        return s, r, d, p

    def render(self):
        if self.__last_action != -1:
            print("\33[8A", end="")
        print("")
        board = self.__env.render(mode="ansi")
        board = board.replace("C", bcolors.Red + "C" + bcolors.ResetAll)
        board = board.replace("x", bcolors.LightBlue + "x" + bcolors.ResetAll)
        print(board[:-2])
        print("Action: {}".format(self.__actionNames[self.__last_action + 1]))
        print("Score: {}".format(self.__score))
        print("")

        time.sleep(0.1)

    def plot(self, policy):
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