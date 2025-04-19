import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

# ------------------------------
# 1. Wireless Environment
# ------------------------------
class WirelessEnv:
    def __init__(self, noisy_sequences, clean_sequences, window_size=5):
        self.noisy_sequences = noisy_sequences
        self.clean_sequences = clean_sequences
        self.window_size = window_size
        self.half_window = window_size // 2
        self.current_seq = 0
        self.current_pos = self.half_window

    def reset(self, seq_idx=None):
        if seq_idx is None:
            self.current_seq = np.random.randint(0, len(self.clean_sequences))
        else:
            self.current_seq = seq_idx
        self.current_pos = self.half_window
        return self._get_window()

    def _get_window(self):
        noisy = self.noisy_sequences[self.current_seq]
        padded = np.pad(noisy, (self.half_window, self.half_window), mode='constant')
        return padded[self.current_pos:self.current_pos + self.window_size]

    def step(self, action):
        true_bit = self.clean_sequences[self.current_seq][self.current_pos - self.half_window]
        reward = 1 if action == true_bit else -1
        self.current_pos += 1
        done = self.current_pos >= len(self.clean_sequences[self.current_seq]) + self.half_window - 1
        return self._get_window(), reward, done, true_bit

# ------------------------------
# 2. SBRNN Model
# ------------------------------
class SBRNN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=2, window_size=5):
        super(SBRNN, self).__init__()
        self.window_size = window_size
        self.rnn = nn.GRU(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(-1)  # Add channel dimension
        out, _ = self.rnn(x)
        center_out = out[:, self.window_size // 2, :]
        return self.fc(center_out)

# ------------------------------
# 3. Training + Validation
# ------------------------------
def train_and_validate():
    # Load dataset
    data = np.load("channel_dataset.npz")
    noisy_data = data['noisy']
    clean_data = data['clean']

    # Split data
    X_train_noisy, X_val_noisy, X_train_clean, X_val_clean = train_test_split(
        noisy_data, clean_data, test_size=0.2, random_state=42
    )

    train_env = WirelessEnv(X_train_noisy, X_train_clean, window_size=5)
    val_env = WirelessEnv(X_val_noisy, X_val_clean, window_size=5)

    # Hyperparameters
    gamma = 0.99
    lr = 0.001
    epsilon_start = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 64
    memory_size = 10000
    episodes = 500

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SBRNN(window_size=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    memory = deque(maxlen=memory_size)
    epsilon = epsilon_start

    train_rewards = []
    val_accuracies = []
    val_episodes = []
    best_val_acc = 0.0

    for episode in range(episodes):
        state = train_env.reset()
        done = False
        total_reward = 0
        correct = 0
        total = 0

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, 1)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = model(state_tensor)
                    action = torch.argmax(q_values).item()

            next_state, reward, done, true_bit = train_env.step(action)
            memory.append((state, action, reward, next_state, done))
            total_reward += reward
            correct += (action == true_bit)
            total += 1
            state = next_state

            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                max_len = max(len(x[0]) for x in batch)
                states = np.array([np.pad(x[0], (0, max_len - len(x[0]))) for x in batch])
                next_states = np.array([np.pad(x[3], (0, max_len - len(x[3]))) for x in batch])
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor([x[1] for x in batch]).to(device)
                rewards = torch.FloatTensor([x[2] for x in batch]).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor([x[4] for x in batch]).to(device)

                current_q = model(states).gather(1, actions.unsqueeze(1))
                with torch.no_grad():
                    next_q = model(next_states).max(1)[0]
                    targets = rewards + gamma * next_q * (1 - dones)

                loss = criterion(current_q.squeeze(), targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        train_rewards.append(total_reward)

        print(f"Episode {episode+1}/{episodes} | Train Reward: {total_reward} | "
              f"Train Acc: {correct/total:.2%} | Epsilon: {epsilon:.3f}")

        # Validation
        if episode % 10 == 0:
            model.eval()
            val_correct = 0
            val_total = 0
            val_state = val_env.reset()
            val_done = False
            predicted_bits = []  # Store predicted bits
            actual_bits = []    # Store actual bits

            while not val_done:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(val_state).unsqueeze(0).to(device)
                    action = torch.argmax(model(state_tensor)).item()
                val_state, _, val_done, true_bit = val_env.step(action)
                predicted_bits.append(action)
                actual_bits.append(true_bit)
                val_correct += (action == true_bit)
                val_total += 1

            val_acc = val_correct / val_total
            val_accuracies.append(val_acc)
            val_episodes.append(episode)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "best_sbrnn_qlearning.pth")

            # Plot actual vs predicted bits
            plot_bit_comparison(np.array(actual_bits), np.array(predicted_bits), n_bits=100)

    # Plot training results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_rewards)
    plt.title("Training Rewards")
    plt.subplot(1, 2, 2)
    plt.plot(val_episodes, val_accuracies, marker='o')
    plt.title("Validation Accuracy")
    plt.ylim(0, 1)
    plt.show()

    # Final Evaluation
    model.load_state_dict(torch.load("best_sbrnn_qlearning.pth"))
    model.eval()
    total_correct = 0
    total_bits = 0

    for seq_idx in range(len(X_val_clean)):
        val_env.reset(seq_idx=seq_idx)
        done = False
        while not done:
            state = val_env._get_window()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = torch.argmax(model(state_tensor)).item()
            _, _, done, true_bit = val_env.step(action)
            total_correct += (action == true_bit)
            total_bits += 1

    print(f"\nOVERALL VALIDATION ACCURACY: {total_correct / total_bits:.4%}")
    print(f"BER: {1 - (total_correct / total_bits):.4%}")

def plot_bit_comparison(actual, predicted, n_bits=100):
    plt.figure(figsize=(15, 5))
    error_mask = (actual[:n_bits] != predicted[:n_bits])

    for i in np.where(error_mask)[0]:
        plt.axvspan(i - 0.5, i + 0.5, color='red', alpha=0.3)

    plt.step(np.arange(n_bits), actual[:n_bits], where='post',
             label='Actual', linewidth=2, color='blue')
    plt.step(np.arange(n_bits), predicted[:n_bits], where='post',
             linestyle=':', label='Predicted', linewidth=2, color='orange')

    plt.title(f'Bit Comparison (First {n_bits} Bits)', fontsize=14)
    plt.xlabel('Bit Position', fontsize=12)
    plt.ylabel('Bit Value', fontsize=12)
    plt.yticks([0, 1], ['0', '1'], fontsize=12)
    plt.xticks(fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_and_validate()
