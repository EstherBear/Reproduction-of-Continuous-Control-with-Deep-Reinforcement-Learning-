from copy import deepcopy
from models import *


class DDPGAgent(object):

    def __init__(self, state_dim, action_dim, max_action, mode, device, discount=0.99, tau=0.005):
        self.device = device
        self.discount = discount
        self.tau = tau
        self.mode = mode
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.target_actor = deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(state_dim, action_dim).to(device)
        self.target_critic = deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

    def select_action(self, state):
        if self.mode == 'bn' or self.mode == 'bntn':
            self.actor.eval()
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state, mode=self.mode).cpu().data.numpy().flatten()
        if self.mode == 'bn' or self.mode == 'bntn':
            self.actor.train()
        return action

    @staticmethod
    def soft_update(current_network, target_network, tau):
        for current_param, target_param in zip(current_network.parameters(), target_network.parameters()):
            target_param.data.copy_(tau * current_param.data + (1.0 - tau) * target_param.data)

    def save_checkpoint(self, filename):
        torch.save(self.critic.state_dict(), filename + '_critic')
        torch.save(self.critic_optimizer.state_dict(), filename + '_critic_optimizer')
        torch.save(self.actor.state_dict(), filename + '_actor')
        torch.save(self.actor_optimizer.state_dict(), filename + '_actor_optimizer')

    def load_checkpoint(self, filename):
        self.critic.load_state_dict(
            torch.load(
                filename + "_critic",
                map_location=torch.device('cpu')
            )
        )
        self.critic_optimizer.load_state_dict(
            torch.load(
                filename + "_critic_optimizer",
                map_location=torch.device('cpu')
            )
        )
        self.target_critic = deepcopy(self.critic)
        self.actor.load_state_dict(
            torch.load(
                filename + "_actor",
                map_location=torch.device('cpu')
            )
        )
        self.actor_optimizer.load_state_dict(
            torch.load(
                filename + "_actor_optimizer",
                map_location=torch.device('cpu')
            )
        )
        self.target_actor = deepcopy(self.actor)

    def train(self, replay_buffer, batch_size=100):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        # Q'(si+1, μ'(si+1))
        if self.mode == 'bn':
            self.actor.eval()
            self.critic.eval()
            Q_target = self.critic(
                next_state, self.actor(next_state, mode=self.mode), mode=self.mode)
            self.actor.train()
            self.critic.train()
        else:
            Q_target = self.target_critic(
                next_state, self.target_actor(next_state, mode=self.mode), mode=self.mode)
        # ri+γQ'(si+1, μ'(si+1))
        Q_target = reward + (not_done * self.discount * Q_target).detach()
        Q_value = self.critic(state, action, mode=self.mode)
        
        # update critic
        critic_loss = F.mse_loss(Q_value, Q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # update actor
        actor_loss = -self.critic(state, self.actor(state,  mode=self.mode),  mode=self.mode).mean()
        estimated_reward = (-actor_loss).detach()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # update target network softly
        DDPGAgent.soft_update(self.critic, self.target_critic, self.tau)
        DDPGAgent.soft_update(self.actor, self.target_actor, self.tau)
        
        return estimated_reward
