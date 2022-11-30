import pickle
import os
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


class IRLWrapper:
    def __init__(self, model_path="../external_repos/guided-cost-learning/trained_models/best_model/model"):

        data_path = "prevmode" if "prevmode" in model_path else "noprevmode"
        with open(
            os.path.join(
                "..", "external_repos", "guided-cost-learning", "expert_samples", data_path, "mobis_train.pkl",
            ),
            "rb",
        ) as infile:
            (_, self.feat_mean, self.feat_std) = pickle.load(infile)

        with open("config.json", "r") as infile:
            self.included_modes = np.array(json.load(infile)["included_modes"])

        f = open(os.path.join("..", "data", "mobis", "trips_features.csv"), "r")
        data_columns = f.readlines(1)[0][:-1].split(",")
        prev_mode_cols = [col for col in data_columns if col.startswith("feat_prev_")]
        self.feat_columns = [col for col in data_columns if col.startswith("feat")]
        if "prevmode" not in model_path:
            for x in prev_mode_cols:
                self.feat_columns.remove(x)
        self.feat_mean = self.feat_mean[self.feat_columns]
        self.feat_std = self.feat_std[self.feat_columns]

        self.policy = PG(len(self.feat_columns), len(self.included_modes))
        self.policy.load_model(os.path.join(model_path))

    def __call__(self, feature_vec):
        feature_vec = np.array(feature_vec[self.feat_columns]).astype(float)
        feature_vec = (feature_vec - self.feat_mean) / self.feat_std

        action_probs = self.policy.predict_probs(np.array([feature_vec]))[0]
        a = np.random.choice(self.policy.n_actions, p=action_probs)
        return self.included_modes[a]


class PG(nn.Module):
    def __init__(self, state_shape, n_actions):

        super().__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.model = nn.Sequential(
            nn.Linear(in_features=state_shape, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=self.n_actions),
        )

    def forward(self, x):
        logits = self.model(x)
        return logits

    def predict_probs(self, states):
        states = torch.FloatTensor(states)
        logits = self.model(states).detach()
        probs = F.softmax(logits, dim=-1).numpy()
        # print(states, logits, probs)
        return probs

    def generate_session(self, env, t_max=1000):
        states, traj_probs, actions, rewards = [], [], [], []
        s = env.reset()
        q_t = 1.0
        for t in range(t_max):
            action_probs = self.predict_probs(np.array([s]))[0]
            a = np.random.choice(self.n_actions, p=action_probs)
            new_s, r, done, info = env.step(a)

            q_t *= action_probs[a]

            states.append(s)
            traj_probs.append(q_t)
            actions.append(a)
            rewards.append(r)

            s = new_s
            if done:
                break

        return states, actions, rewards, traj_probs

    def _get_cumulative_rewards(self, rewards, gamma=0.99):
        G = np.zeros_like(rewards, dtype=float)
        G[-1] = rewards[-1]
        for idx in range(-2, -len(rewards) - 1, -1):
            G[idx] = rewards[idx] + gamma * G[idx + 1]
        return G

    def _to_one_hot(self, y_tensor, ndims):
        y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
        y_one_hot = torch.zeros(y_tensor.size()[0], ndims).scatter_(1, y_tensor, 1)
        return y_one_hot

    def train_on_env(self, env, gamma=0.99, entropy_coef=1e-2):
        states, actions, rewards = self.generate_session(env)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int32)
        cumulative_returns = np.array(self._get_cumulative_rewards(rewards, gamma))
        cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32)

        logits = self.model(states)
        probs = nn.functional.softmax(logits, -1)
        log_probs = nn.functional.log_softmax(logits, -1)

        log_probs_for_actions = torch.sum(log_probs * self._to_one_hot(actions, env.action_space.n), dim=1)

        entropy = -torch.mean(torch.sum(probs * log_probs), dim=-1)
        loss = -torch.mean(log_probs_for_actions * cumulative_returns - entropy * entropy_coef)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return np.sum(rewards)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))


if __name__ == "__main__":
    # TEST CODE
    from v2g4carsharing.mode_choice_model.evaluate import mode_share_plot

    mobis_data = pd.read_csv("../data/mobis/trips_features.csv")

    irl = IRLWrapper()

    act_real = np.argmax(np.array(mobis_data[irl.included_modes]), axis=1)
    act_real = np.array(irl.included_modes)[act_real]

    # mobis_data["feat_caraccess"] = 1 # for testing the influence of the caraccess feature

    mode_sim = []
    for i in range(len(mobis_data)):
        mode_sim.append(irl(mobis_data.iloc[i]))
    mode_share_plot(act_real, mode_sim, out_path=".")

