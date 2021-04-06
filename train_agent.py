from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

from knowledge_graph import KnowledgeGraph
from kg_env import BatchKGEnvironment
from utils import *

logger = None

SavedAction = namedtuple("SavedAction", ["log_prob", "value"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        gamma=0.99,
        hidden_sizes=[512, 256],
        pretrained_model_path=None,
    ):
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.gamma = gamma

        self.l1 = nn.Linear(state_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.actor = nn.Linear(hidden_sizes[1], act_dim)
        self.critic = nn.Linear(hidden_sizes[1], 1)

        self.saved_actions = []
        self.rewards = []
        self.entropy = []

        if pretrained_model_path is not None:
            with open(pretrained_model_path, "r") as f:
                self.load_state_dict(torch.load(f))

    def forward(self, inputs):
        state, act_mask = inputs  # state: [bs, state_dim], act_mask: [bs, act_dim]
        x = self.l1(state)
        x = F.dropout(F.elu(x), p=0.5)
        out = self.l2(x)
        x = F.dropout(F.elu(out), p=0.5)

        actor_logits = self.actor(x)
        actor_logits[act_mask.logical_not()] = -999999.0
        act_probs = F.softmax(actor_logits, dim=-1)  # Tensor of [bs, act_dim]

        state_values = self.critic(x)  # Tensor of [bs, 1]
        return act_probs, state_values

    def evaluate(self, inputs):
        action_probs, state_values = self.forward(inputs)
        dist = Categorical(action_probs)
        acts = dist.sample()
        state, mask = inputs
        valid_idx = mask.gather(1, acts.view(-1, 1)).view(-1)
        acts[valid_idx == 0] = 0
        action_logprobs = dist.log_prob(acts)

        dist_entropy = dist.entropy()

        return action_logprobs, state_values, dist_entropy

    def act(self, batch_state, batch_act_mask, memory: Memory):
        state = torch.FloatTensor(batch_state).to(device)  # Tensor [bs, state_dim]
        act_mask = torch.BoolTensor(batch_act_mask).to(
            device
        )  # Tensor of [bs, act_dim]

        model_input = (state, act_mask)

        probs, value = self(model_input)

        m = Categorical(probs)
        acts = m.sample()

        valid_idx = act_mask.gather(1, acts.view(-1, 1)).view(-1)
        acts[valid_idx == 0] = 0

        memory.states.append(model_input)
        memory.actions.append(acts)
        memory.logprobs.append(m.log_prob(acts))

        self.saved_actions.append(SavedAction(m.log_prob(acts), value))
        self.entropy.append(m.entropy())
        return acts.cpu().numpy().tolist()

    def select_action(self, batch_state, batch_act_mask, device):
        state = torch.FloatTensor(batch_state).to(device)  # Tensor [bs, state_dim]
        act_mask = torch.BoolTensor(batch_act_mask).to(
            device
        )  # Tensor of [bs, act_dim]

        probs, value = self(
            (state, act_mask)
        )  # act_probs: [bs, act_dim], state_value: [bs, 1]
        m = Categorical(probs)
        acts = m.sample()  # Tensor of [bs, ], requires_grad=False
        # [CAVEAT] If sampled action is out of action_space, choose the first action in action_space.
        valid_idx = act_mask.gather(1, acts.view(-1, 1)).view(-1)
        acts[valid_idx == 0] = 0

        self.saved_actions.append(SavedAction(m.log_prob(acts), value))
        self.entropy.append(m.entropy())
        return acts.cpu().numpy().tolist()

    def update(self, optimizer, device, ent_weight):
        if len(self.rewards) <= 0:
            del self.rewards[:]
            del self.saved_actions[:]
            del self.entropy[:]
            return 0.0, 0.0, 0.0

        batch_rewards = np.vstack(self.rewards).T  # numpy array of [bs, #steps]
        batch_rewards = torch.FloatTensor(batch_rewards).to(device)
        num_steps = batch_rewards.shape[1]
        for i in range(1, num_steps):
            batch_rewards[:, num_steps - i - 1] += (
                self.gamma * batch_rewards[:, num_steps - i]
            )

        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        for i in range(0, num_steps):
            log_prob, value = self.saved_actions[
                i
            ]  # log_prob: Tensor of [bs, ], value: Tensor of [bs, 1]
            advantage = batch_rewards[:, i] - value.squeeze(1)  # Tensor of [bs, ]
            actor_loss += -log_prob * advantage.detach()  # Tensor of [bs, ]
            critic_loss += advantage.pow(2)  # Tensor of [bs, ]
            entropy_loss += -self.entropy[i]  # Tensor of [bs, ]
        actor_loss = actor_loss.mean()
        critic_loss = critic_loss.mean()
        entropy_loss = entropy_loss.mean()
        loss = actor_loss + critic_loss + ent_weight * entropy_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del self.rewards[:]
        del self.saved_actions[:]
        del self.entropy[:]

        return loss.item(), actor_loss.item(), critic_loss.item(), entropy_loss.item()

    def clear(self):
        del self.rewards[:]
        del self.saved_actions[:]
        del self.entropy[:]


class ACDataLoader(object):
    def __init__(self, uids, batch_size):
        self.uids = np.array(uids)
        self.num_users = len(uids)
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self._rand_perm = np.random.permutation(self.num_users)
        self._start_idx = 0
        self._has_next = True

    def has_next(self):
        return self._has_next

    def get_batch(self):
        if not self._has_next:
            return None
        # Multiple users per batch
        end_idx = min(self._start_idx + self.batch_size, self.num_users)
        batch_idx = self._rand_perm[self._start_idx : end_idx]
        batch_uids = self.uids[batch_idx]
        self._has_next = self._has_next and end_idx < self.num_users
        self._start_idx = end_idx
        return batch_uids.tolist()


class PPO:
    def __init__(self, ac_config, args):
        self.K_epochs = args.K_epochs
        self.eps_clip = args.eps_clip

        self.policy = ActorCritic(**ac_config).to(device)
        self.policy_old = ActorCritic(**ac_config).to(device)

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.gamma = args.gamma

    def calc_loss(self, y_predict, y_true):
        pass

    def update(self, optimizer, ent_weight: float, memory: Memory):

        batch_rewards = np.vstack(memory.rewards).T  # numpy array of [bs, #steps]
        batch_rewards = torch.FloatTensor(batch_rewards).to(device)
        num_steps = batch_rewards.shape[1]
        batch_size = batch_rewards.shape[0]
        for i in range(1, num_steps):
            batch_rewards[:, num_steps - i - 1] += (
                self.gamma * batch_rewards[:, num_steps - i]
            )

        ret_loss = []
        ret_actor_loss = []
        ret_critic_loss = []
        ret_entropy_loss = []

        for _ in range(self.K_epochs):
            actor_loss = torch.zeros(batch_size).to(device)
            critic_loss = torch.zeros(batch_size).to(device)
            entropy_loss = torch.zeros(batch_size).to(device)
            for i in range(0, num_steps):
                logprob, value, entropy = self.policy.evaluate(memory.states[i])

                advantage = batch_rewards[:, i] - value.squeeze(1)
                old_log_prob = memory.logprobs[i]
                ratios = torch.exp(logprob - old_log_prob.detach())

                surr1 = ratios * advantage
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * advantage
                )
                # print(surr1.shape, surr2.shape, actor_loss.shape)
                actor_loss += -torch.min(surr1, surr2)
                critic_loss += advantage.pow(2)  # Tensor of [bs, ]
                entropy_loss += -entropy  # Tensor of [bs, ]

            mean_actor_loss = actor_loss.mean()
            mean_critic_loss = critic_loss.mean()
            mean_entropy_loss = entropy_loss.mean()

            loss = mean_actor_loss + mean_critic_loss + ent_weight * mean_entropy_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ret_loss.append(loss.item())
            ret_actor_loss.append(mean_actor_loss.item())
            ret_critic_loss.append(mean_critic_loss.item())
            ret_entropy_loss.append(mean_entropy_loss.item())

        self.policy_old.clear()

        self.policy_old.load_state_dict(self.policy.state_dict())
        ret = [ret_loss, ret_actor_loss, ret_critic_loss, ret_entropy_loss]

        ret = [np.mean(l) for l in ret]

        return ret


def ppo_train(args):
    env = BatchKGEnvironment(
        args.dataset,
        args.max_acts,
        max_path_len=args.max_path_len,
        state_history=args.state_history,
    )
    ppo_model = PPO(
        {
            "state_dim": env.state_dim,
            "act_dim": env.act_dim,
            "gamma": args.gamma,
            "hidden_sizes": args.hidden,
            "pretrained_model_path": args.pretrained_model_path,
        },
        args,
    )

    uids = list(env.kg(USER).keys())
    dataloader = ACDataLoader(uids, args.batch_size)

    total_losses, total_plosses, total_vlosses, total_entropy, total_rewards = (
        [],
        [],
        [],
        [],
        [],
    )

    step = 0
    ppo_model.policy.train()
    ppo_model.policy_old.train()
    memory = Memory()
    optimizer = torch.optim.Adam(ppo_model.policy.parameters(), lr=args.lr)

    for i_epoch in range(1, args.epochs + 1):
        dataloader.reset()
        while dataloader.has_next():
            batch_uids = dataloader.get_batch()
            batch_state = env.reset(batch_uids)

            done = False
            while not done:
                batch_act_mask = env.batch_action_mask(dropout=args.act_dropout)

                batch_action = ppo_model.policy_old.act(
                    batch_state, batch_act_mask, memory
                )

                batch_state, batch_reward, done = env.batch_step(batch_action)

                memory.rewards.append(batch_reward)
                memory.is_terminals.append(done)

            lr = args.lr * max(
                1e-4, 1.0 - float(step) / (args.epochs * len(uids) / args.batch_size)
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            total_rewards.append(np.sum(memory.rewards))
            loss, ploss, vloss, eloss = ppo_model.update(
                optimizer, args.ent_weight, memory
            )

            total_losses.append(loss)
            total_plosses.append(ploss)
            total_vlosses.append(vloss)
            total_entropy.append(eloss)
            step += 1

            if step > 0 and step % 100 == 0:
                avg_reward = np.mean(total_rewards) / args.batch_size
                avg_loss = np.mean(total_losses)
                avg_ploss = np.mean(total_plosses)
                avg_vloss = np.mean(total_vlosses)
                avg_entropy = np.mean(total_entropy)
                (
                    total_losses,
                    total_plosses,
                    total_vlosses,
                    total_entropy,
                    total_rewards,
                ) = ([], [], [], [], [])
                logger.info(
                    "epoch/step={:d}/{:d}".format(i_epoch, step)
                    + " | loss={:.5f}".format(avg_loss)
                    + " | ploss={:.5f}".format(avg_ploss)
                    + " | vloss={:.5f}".format(avg_vloss)
                    + " | entropy={:.5f}".format(avg_entropy)
                    + " | reward={:.5f}".format(avg_reward)
                )

        policy_file = "{}/policy_model_epoch_{}.ckpt".format(args.log_dir, epoch)
        logger.info("Save model to " + policy_file)
        torch.save(ppo_model.policy.state_dict(), policy_file)


def train(args):
    env = BatchKGEnvironment(
        args.dataset,
        args.max_acts,
        max_path_len=args.max_path_len,
        state_history=args.state_history,
    )
    uids = list(env.kg(USER).keys())
    dataloader = ACDataLoader(uids, args.batch_size)
    model = ActorCritic(
        env.state_dim, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden
    ).to(args.device)
    logger.info("Parameters:" + str([i[0] for i in model.named_parameters()]))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    total_losses, total_plosses, total_vlosses, total_entropy, total_rewards = (
        [],
        [],
        [],
        [],
        [],
    )
    step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        ### Start epoch ###
        dataloader.reset()
        while dataloader.has_next():
            batch_uids = dataloader.get_batch()
            ### Start batch episodes ###
            batch_state = env.reset(batch_uids)  # numpy array of [bs, state_dim]
            done = False
            while not done:
                batch_act_mask = env.batch_action_mask(
                    dropout=args.act_dropout
                )  # numpy array of size [bs, act_dim]
                batch_act_idx = model.select_action(
                    batch_state, batch_act_mask, args.device
                )  # int
                batch_state, batch_reward, done = env.batch_step(batch_act_idx)
                model.rewards.append(batch_reward)
            ### End of episodes ###

            lr = args.lr * max(
                1e-4, 1.0 - float(step) / (args.epochs * len(uids) / args.batch_size)
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Update policy
            total_rewards.append(np.sum(model.rewards))
            loss, ploss, vloss, eloss = model.update(
                optimizer, args.device, args.ent_weight
            )
            total_losses.append(loss)
            total_plosses.append(ploss)
            total_vlosses.append(vloss)
            total_entropy.append(eloss)
            step += 1

            # Report performance
            if step > 0 and step % args.step == 0:
                avg_reward = np.mean(total_rewards) / args.batch_size
                avg_loss = np.mean(total_losses)
                avg_ploss = np.mean(total_plosses)
                avg_vloss = np.mean(total_vlosses)
                avg_entropy = np.mean(total_entropy)
                (
                    total_losses,
                    total_plosses,
                    total_vlosses,
                    total_entropy,
                    total_rewards,
                ) = ([], [], [], [], [])
                logger.info(
                    "epoch/step={:d}/{:d}".format(epoch, step)
                    + " | loss={:.5f}".format(avg_loss)
                    + " | ploss={:.5f}".format(avg_ploss)
                    + " | vloss={:.5f}".format(avg_vloss)
                    + " | entropy={:.5f}".format(avg_entropy)
                    + " | reward={:.5f}".format(avg_reward)
                )
        ### END of epoch ###

        policy_file = "{}/policy_model_epoch_{}.ckpt".format(args.log_dir, epoch)
        logger.info("Save model to " + policy_file)
        torch.save(model.state_dict(), policy_file)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name", type=str, default="train_agent", help="directory name."
    )
    parser.add_argument("--seed", type=int, default=123, help="random seed.")
    parser.add_argument("--gpu", type=str, default="0", help="gpu device.")
    parser.add_argument("--epochs", type=int, default=50, help="Max number of epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate.")
    parser.add_argument(
        "--max_acts", type=int, default=250, help="Max number of actions."
    )
    parser.add_argument("--max_path_len", type=int, default=3, help="Max path length.")
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="reward discount factor."
    )
    parser.add_argument(
        "--ent_weight", type=float, default=1e-3, help="weight factor for entropy loss"
    )
    parser.add_argument(
        "--act_dropout", type=float, default=0.5, help="action dropout rate."
    )
    parser.add_argument(
        "--state_history", type=int, default=1, help="state history length"
    )
    parser.add_argument(
        "--hidden", type=str, default="[512, 256]", help="number of samples"
    )
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--pretrained_model_path", type=str, default=None)

    parser.add_argument("--K_epochs", type=int, default=5)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    args = parser.parse_args()
    for k in config:
        v = config[k]
        if v is not None:
            setattr(args, k, v)
    args.hidden = eval(args.hidden)
    args.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    args.log_dir = "{}/{}".format(TMP_DIR[args.dataset], args.name)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    global logger
    logger = get_logger(args.log_dir + "/train_log.txt")
    logger.info(args)

    set_random_seed(args.seed)
    ppo_train(args)


if __name__ == "__main__":
    main()
