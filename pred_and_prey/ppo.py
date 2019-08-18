import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from Env import Env

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self.fc1   = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,5)
        self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s.tolist())
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

def main():
    print("Load? (y/else)")
    load = input()
    test = 'n'
    if load == 'y':
        print("test? (y/else)")
        test = input()
    print("Model name (overwrite)")
    MODEL_PATH = "saved\\" + input() +".pt"

    print("path is " + MODEL_PATH)

    env = Env()
    model = PPO()
    if load == 'y':
        model.load_state_dict(torch.load(MODEL_PATH))
        param = list(model.parameters())
        print(param)
    score = 0.0

    if test == 'y':
        done = False
        env.spawn()
        while not done:
            env.show_board()
            s = env.state()
            prob = model.pi(s)
            m = Categorical(prob)
            a = m.sample().item()
            done = env.step(a)

            s_prime = env.state()
            r = env.get_reward()
            score += r
        print("test reward is " + str(score))

    else:
        print_interval = 50

        for n_epi in range(10000):
            env.spawn()
            done = False
            while not done:
                s = env.state()
                prob = model.pi(s)
                m = Categorical(prob)
                a = m.sample().item()
                done = env.step(a)

                s_prime = env.state()
                r = env.get_reward()

                model.put_data((s, a, r/100.0, s_prime.tolist(), prob[a].item(), done))
                s = s_prime

                score += r
                if done:
                    break

            model.train_net()

            if n_epi % print_interval == 0:
                print("# of episode :{}, score : {:.1f}".format(n_epi, score/print_interval))
                score = 0.0

        torch.save(model.state_dict(), MODEL_PATH)

main()