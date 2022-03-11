import gym.envs.toy_text as toy_text
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy
import random


class Env:
    def __init__(self):
        self.__env = toy_text.BlackjackEnv()
        self.action_space = self.__env.action_space
        self.fig = None
        self.reset()

    def reset(self):
        self.__last_action = -1
        return self.__env.reset()

    def step(self, action):
        self.__last_action = action
        return self.__env.step(action)

    def render(self):
        reward = toy_text.blackjack.cmp(
            toy_text.blackjack.score(self.__env.player),
            toy_text.blackjack.score(self.__env.dealer),
        )
        # Print action
        if self.__last_action < 0:
            print("\n\nStarting Hands:")
        else:
            print("Action: {}".format(["Stop", "Hit"][self.__last_action]))

        dealerBust = " [BUST!]" if toy_text.blackjack.is_bust(self.__env.dealer) else ""
        playerBust = " [BUST!]" if toy_text.blackjack.is_bust(self.__env.player) else ""
        if self.__last_action == 0 or playerBust != "":
            dealerHand = self.__env.dealer
            dealerSum = toy_text.blackjack.sum_hand(dealerHand)
        else:
            dealerHand = [self.__env.dealer[0]]
            dealerSum = toy_text.blackjack.sum_hand(dealerHand)
            dealerHand.append("?")
        # Print hands
        print(
            "Dealer{}: {}, {}\nPlayer{}: {}, {}".format(
                dealerBust,
                dealerSum,
                str(dealerHand),
                playerBust,
                toy_text.blackjack.sum_hand(self.__env.player),
                str(self.__env.player),
            )
        )

        if self.__last_action == 0 or playerBust != "":
            if reward == 1:
                print("You Win!")
            if reward == -1:
                print("You Lost!")
            if reward == 0:
                print("Draw!")
        print("")

        time.sleep(0.2)

    def plot(self, policy):
        if self.fig == None:
            self.fig = plt.figure()
            self.subp = [[None, None], [None, None]]
            self.subp[0][0] = self.fig.add_subplot(321)
            self.subp[0][1] = self.fig.add_subplot(322, projection="3d")
            self.subp[1][0] = self.fig.add_subplot(325)
            self.subp[1][1] = self.fig.add_subplot(326, projection="3d")
        x = [x for x in range(1, 11)]
        y = [y for y in range(11, 22)]
        X, Y = numpy.meshgrid(x, y)

        P1 = numpy.ones(X.shape) * (-1)
        V1 = numpy.ones(X.shape) * (-1)
        P2 = numpy.ones(X.shape) * (-1)
        V2 = numpy.ones(X.shape) * (-1)
        for a in range(X.shape[0]):
            for b in range(X.shape[1]):
                dealer = X[a][b]
                own = Y[a][b]
                state1 = (own, dealer, False)
                state2 = (own, dealer, True)
                try:
                    P1[a][b] = policy.GetPolicy(state1)
                    P2[a][b] = policy.GetPolicy(state2)
                    V1[a][b] = policy.GetValue(state1)
                    V2[a][b] = policy.GetValue(state2)
                except:
                    pass

        dealer_loc = numpy.arange(10)
        dealer_labels = [str(x + 1) for x in dealer_loc]
        dealer_labels[0] = "A"

        player_loc = numpy.arange(len(y))
        player_labels = [str(21 - x) for x in player_loc]

        self.subp[0][0].cla()
        self.subp[0][0].set_xlabel("Dealer")
        self.subp[0][0].set_ylabel("Player")
        self.subp[0][0].title.set_text("No active Ace")
        self.subp[0][0].imshow(numpy.flip(P1, 0), interpolation="none")
        self.subp[0][0].set_xticks(dealer_loc)
        self.subp[0][0].set_xticklabels(dealer_labels)
        self.subp[0][0].set_yticks(player_loc)
        self.subp[0][0].set_yticklabels(player_labels)

        self.subp[1][0].cla()
        self.subp[1][0].set_xlabel("Dealer")
        self.subp[1][0].set_ylabel("Player")
        self.subp[1][0].title.set_text("Active Ace")
        self.subp[1][0].imshow(numpy.flip(P2, 0), interpolation="none")
        self.subp[1][0].set_xticks(dealer_loc)
        self.subp[1][0].set_xticklabels(dealer_labels)
        self.subp[1][0].set_yticks(player_loc)
        self.subp[1][0].set_yticklabels(player_labels)

        self.subp[0][1].cla()
        self.subp[0][1].set_xlabel("Dealer")
        self.subp[0][1].set_ylabel("Player")
        self.subp[0][1].set_zlabel("Value")
        self.subp[0][1].title.set_text("No active Ace")
        self.subp[0][1].plot_surface(
            X, Y, V1, rstride=1, cstride=1, cmap="viridis", edgecolor="none"
        )

        self.subp[1][1].cla()
        self.subp[1][1].set_xlabel("Dealer")
        self.subp[1][1].set_ylabel("Player")
        self.subp[1][1].set_zlabel("Value")
        self.subp[1][1].title.set_text("Active Ace")
        self.subp[1][1].plot_surface(
            X, Y, V2, rstride=1, cstride=1, cmap="viridis", edgecolor="none"
        )

        plt.ion()
        plt.show()
        plt.pause(0.01)
