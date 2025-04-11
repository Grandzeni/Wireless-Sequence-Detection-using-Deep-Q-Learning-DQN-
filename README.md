# Wireless Sequence Detection using Deep Q Learning DQN

This project explores the application of Deep Reinforcement Learning for symbol sequence detection over a noisy wireless channel. Using a custom environment that simulates Binary Phase Shift Keying (BPSK) transmission with additive Gaussian noise, we train a Deep Q-Network (DQN) to decode the original transmitted bit sequence based on noisy received values.

# What It Does
Simulates a wireless transmission environment using BPSK modulation.

Adds realistic channel noise based on a configurable Signal-to-Noise Ratio (SNR).

Uses a Deep Q-Network (DQN) to learn the best sequence decoding strategy through reinforcement learning.

Learns to maximize accuracy by receiving positive rewards for correct symbol predictions and penalties for incorrect ones.

Saves the trained model and visualizes the learning curve over training episodes.

# Applications
Symbol detection in noisy communication systems

AI-powered channel decoding

Exploration of RL in physical-layer communications
