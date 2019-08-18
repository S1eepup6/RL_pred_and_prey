import random
import pprint
import torch

class Env:
    def __init__(self):
        self.board = [ ['_'] * 7 for i in range(7) ]
        self.max = 6
        self.min = 0

        self.pred = [3,3]   # spawn on center
        self.prey = [0,0]   # temporary point

        # Spawn point
        self.ul = [0,0]    # upper left
        self.ur = [0,6]    # upper right
        self.dl = [6,0]    # down left
        self.dr = [6,6]    # down right
        self.center = [3,3]    # center

        self.step_number = 1
        self.avail = list()

        self.reward = 0

    def get_reward(self):
        return self.reward

    def spawn(self):
        #spawn prey on random point
        spawn_number = random.randrange(1,5,1)
        if spawn_number == 1:
            self.prey = self.ul
        elif spawn_number == 2:
            self.prey = self.ur
        elif spawn_number == 3:
            self.prey = self.dl
        elif spawn_number == 4:
            self.prey = self.dr
        else:
            self.prey = self.dr

        #spawn pred on center
        self.pred = self.center

        self.step_number = 1

        self.reward = 0

    def state(self):
        stat = torch.tensor([ self.pred[0], self.pred[1], self.prey[0], self.prey[1] ], dtype=torch.float32)
        return stat

    def prey_move(self):
        avail = list()
        if not ((self.prey[0] - 1 == self.pred[0] or self.prey[0] - 2 == self.pred[0]) and (abs(self.prey[1] - self.pred[1]) <= 2)):
            if self.prey[0] - 1 >= self.min:
                move = [self.prey[0] - 1, self.prey[1]]
                avail.append(move)

        if not ((self.prey[1] + 1 == self.pred[1] or self.prey[1] + 2 == self.pred[1]) and (abs(self.prey[0] - self.pred[0]) <= 2)):
            if self.prey[1] + 1 <= self.max:
                move = [self.prey[0], self.prey[1] + 1]
                avail.append(move)

        if not ((self.prey[0] + 1 == self.pred[0] or self.prey[0] + 2 == self.pred[0]) and (abs(self.prey[1] - self.pred[1]) <= 2)):
            if self.prey[0] + 1 <= self.max:
                move = [self.prey[0] + 1, self.prey[1]]
                avail.append(move)

        if not ((self.prey[1] - 1 == self.pred[1] or self.prey[1] - 2 == self.pred[1]) and (abs(self.prey[0] - self.pred[0]) <= 2)):
            if self.prey[1] - 1 >= self.min:
                move = [self.prey[0], self.prey[1] - 1]
                avail.append(move)

        self.avail = avail
        if len(avail) > 0:
            dice = random.randrange(0, len(avail), 1)
            self.prey = avail[dice]

    def step(self, action):
        #clockwise movement
        if action == 0 and self.pred[0] - 1 >= self.min:
            self.pred[0] -= 1
        elif action == 1 and self.pred[1] + 1 <= self.max:
            self.pred[1] += 1
        elif action == 2 and self.pred[0] + 1 <= self.max:
            self.pred[0] += 1
        elif action == 3 and self.pred[1] - 1 >= self.min:
            self.pred[1] -= 1
        elif action == 4:       # stop on original position
            pass
        else:
            pass

        self.step_number += 1
        self.prey_move()
        if self.pred[0] == self.prey[0] and self.pred[1] == self.prey[1]:
            self.reward = 50 - self.step_number
            return True
        elif abs(self.pred[0] - self.prey[0]) <= 1 and abs(self.pred[1] - self.prey[1]) <= 1:
            self.reward += 1
            return False
        elif self.step_number == 20:
            return True
        else:
            return False

    def show_board(self):
        self.board[self.prey[0]][self.prey[1]] = '#' #prey position
        self.board[self.pred[0]][self.pred[1]] = '@' #predator position
        if self.pred[0] == self.prey[0] and self.pred[1] == self.prey[1]:
            self.board[self.pred[0]][self.pred[1]] = '*'
        pp = pprint.PrettyPrinter(indent = 3)
        pp.pprint(self.board)
        print("")
        self.board = [ ['_'] * 7 for i in range(7) ]