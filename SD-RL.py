import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt


class WirelessSequenceEnv:
    def __init__(self, sequence_length=5, snr_db=10):
        self.sequence_length = sequence_length
        self.snr_db = snr_db
        self.snr_linear = 10 ** (snr_db / 10)
        self.original_sequence = self.generate_sequence()
        self.current_index = 0
        self.noisy_sequence = self.add_noise(self.original_sequence)

    def generate_sequence(self):
        return np.random.randint(0, 2, self.sequence_length)

    def add_noise(self, sequence):
        noise = np.random.normal(0, 1 / np.sqrt(self.snr_linear), self.sequence_length)
        return 2 * sequence - 1 + noise  # BPSK: Map {0,1} → {-1,+1}

    def reset(self):
        self.original_sequence = self.generate_sequence()
        self.noisy_sequence = self.add_noise(self.original_sequence)
        self.current_index = 0
        return self.noisy_sequence

    def step(self, action):
        correct_symbol = self.original_sequence[self.current_index]
        reward = 1 if action == correct_symbol else -1
        self.current_index += 1
        done = self.current_index == self.sequence_length
        return self.noisy_sequence, reward, done

# Define the Deep Q-Network (DQN)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Training parameters
gamma = 0.99
learning_rate = 0.001
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 32
memory = []
mem_size = 10000

def train_dqn():
    env = WirelessSequenceEnv(sequence_length=5, snr_db=10)
    state_dim = env.sequence_length
    action_dim = 2
    model = DQN(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    global epsilon
    rewards_history = []
    
    for episode in range(1000):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            if random.random() < epsilon:
                action = random.randint(0, 1)
            else:
                with torch.no_grad():
                    action = torch.argmax(model(state_tensor)).item()
            
            next_state, reward, done = env.step(action)
            memory.append((state, action, reward, next_state, done))
            if len(memory) > mem_size:
                memory.pop(0)
            
            state = next_state
            total_reward += reward
            
            if len(memory) > batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1)
                
                q_values = model(states).gather(1, actions)
                next_q_values = model(next_states).max(1, keepdim=True)[0]
                target_q_values = rewards + gamma * next_q_values * (1 - dones)
                
                loss = criterion(q_values, target_q_values.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_history.append(total_reward)
        print(f"Episode {episode+1}: Total Reward: {total_reward}")
    
    torch.save(model.state_dict(), "dqn_sequence_detector.pth")
    print("Model saved.")
    
    # Plot training reward curve
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward Curve")
    plt.show()

train_dqn()
