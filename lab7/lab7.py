import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Actor-Critic网络
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Linear(input_dim, 128)
        self.policy = nn.Linear(128, action_dim)
        self.value = nn.Linear(128, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc(x))
        policy_dist = torch.softmax(self.policy(x), dim=-1)
        value = self.value(x)
        return policy_dist, value

# Hyperparameters
env = gym.make('CartPole-v1')
input_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
lr = 0.01
gamma = 0.99
episodes = 1000

# 初始化网络和优化器
model = ActorCritic(input_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=lr)

# 训练过程
def train():
    episode_rewards = []
    
    for episode in range(episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        done = False
        total_reward = 0
        
        while not done:
            policy_dist, value = model(state)
            action = np.random.choice(action_dim, p=policy_dist.detach().numpy().flatten())
            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            _, next_value = model(next_state)
            
            # 计算优势
            advantage = reward + (1 - done) * gamma * next_value - value
            
            # 损失函数
            policy_loss = -torch.log(policy_dist.squeeze(0)[action]) * advantage
            value_loss = advantage ** 2
            loss = policy_loss + value_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
            total_reward += reward
            
        episode_rewards.append(total_reward)
        if (episode + 1) % 100 == 0:
            print(f'Episode {episode + 1}, Total Reward: {total_reward}')
    
    return episode_rewards

# 运行训练并绘制结果
rewards = train()
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Performance')
plt.show()

def play_agent():
    env2 = gym.make('CartPole-v1', render_mode='human')
    state, info = env2.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    done = False
    while not done:
        policy_dist, _ = model(state)
        action = torch.multinomial(policy_dist, 1).item()
        next_state, reward, done, _, _  = env2.step(action)
        env2.render()
        state = torch.tensor(next_state, dtype=torch.float32)

if __name__ == '__main__':
    # 训练代理
    # train()
    # 使用训练好的代理进行演示
    play_agent()
