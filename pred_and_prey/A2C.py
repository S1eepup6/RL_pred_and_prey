import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy

from Env import Env

max_epi = 1000
print_interval = 1000
gamma = 0.9

class A2C(nn.Module):
    def __init__(self, n_actions):
        super(A2C,self).__init__()
        self.n_actions = n_actions

        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, self.n_actions)
        self.fc_v = nn.Linear(256, 1)

    def pi(self, x, softmax_dim = 0):
        x = self.fc1(x)
        x = self.fc2(x)
        prob = F.softmax(x, dim = softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc_v(x)
        return x

def compute_target(v_final, r_lst, mask_lst):
    G = v_final.reshape(-1)
    td_target = list()

    for r, mask in zip(r_lst[::-1], mask_lst[::-1]):
        G = r + gamma * G * mask
        td_target.append(G)

    return torch.tensor(td_target[::-1]).float()

def test(epi_idx, model, LOG_PATH):
    env = Env()
    score = 0.0
    done = False
    num_test = 5

    for _ in range(num_test):
        env.spawn()
        while not done:
            s = env.state()
            prob = model.pi(s)
            a = Categorical(prob).sample().numpy()
            done = env.step(a)

        r = env.get_reward()
        score += r
        done = False

    f = open(LOG_PATH, 'w')
    f.write(f"episode # :{epi_idx}, avg score : {score/num_test:.1f}")
    print(f"episode # :{epi_idx}, avg score : {score/num_test:.1f}")

def _test(model):
    env=Env()
    done = False
    env.spawn()
    while not done:
        env.show_board()
        s = env.state()
        prob = model.pi(s)
        print(prob)
        a = Categorical(prob).sample().numpy()
        done = env.step(a)

#############
# MAIN PART #
#############

print("Put number over 0 for loading model")
mode = int(input())

print("PATH for model file")

MODEL_PATH = input()
LOG_PATH =  "saved\\" + MODEL_PATH + ".txt"
MODEL_PATH = "saved\\" + MODEL_PATH + ".pt"

print("Test model will be saved in " + MODEL_PATH)

env = Env()
model = A2C(5)

if mode > 0:
    model.load_state_dict(torch.load(MODEL_PATH))
    param = list(model.parameters())
    print(param)

print("Into test mode? (y/else)")
test_mode = input()
if test_mode == 'y':
    _test(model)
else:
    optimizer = optim.Adam(model.parameters(), lr = 0.005)

    for epi in range(max_epi):
        env.spawn()

        done = False
        s_lst = list()
        a_lst = list()
        r_lst = list()
        d_lst = list()

        while not done:
            s = env.state()
            act = Categorical(model.pi(s)).sample().numpy()
            done = env.step(act)

            s_prime = env.state()
            r = env.get_reward()

            s_lst.append(s.tolist())
            a_lst.append(act.tolist())
            r_lst.append(r)
            d_lst.append(done)

        s_final = s_prime.float()
        v_final = model.v(s_final).detach().clone().numpy()

        td_target = compute_target(v_final, r_lst, d_lst)
        td_target_vec = td_target.reshape(-1)

        s_vec = torch.tensor(s_lst).reshape(-1, 4)
        a_vec = torch.tensor(a_lst).reshape(-1).unsqueeze(1)
        advantage = td_target_vec - model.v(s_vec).reshape(-1)

        pi = model.pi(s_vec, softmax_dim=1)
        pi_a = pi.gather(1, a_vec).reshape(-1)
        loss = -(torch.log(pi_a) * advantage.detach()).mean() + F.smooth_l1_loss(model.v(s_vec).reshape(-1), td_target_vec)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epi % print_interval == 0:
            test(epi, model, LOG_PATH)

    torch.save(model.state_dict(), MODEL_PATH)