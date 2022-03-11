class QTable:
    def __init__(self):
        self.__table = {}
        self.__counter = {}

    def GetValue(self, state, action=None):
        if not state in self.__table:
            return 0

        if action == None:
            value = None
            for a in self.__table[state]:
                if value == None or value < self.__table[state][a]:
                    value = self.__table[state][a]
            return value

        if not action in self.__table[state]:
            return 0

        return self.__table[state][action]

    def SetValue(self, state, action, value):
        if not state in self.__table:
            self.__table[state] = {}
            self.__counter[state] = {}
        if not action in self.__counter[state]:
            self.__counter[state][action] = 0

        self.__table[state][action] = value
        self.__counter[state][action] += 1

    def BestAction(self, state):
        if not state in self.__table:
            return None

        value = None
        action = None
        for a in self.__table[state]:
            if value == None or value < self.__table[state][a]:
                value = self.__table[state][a]
                action = a
        return action
