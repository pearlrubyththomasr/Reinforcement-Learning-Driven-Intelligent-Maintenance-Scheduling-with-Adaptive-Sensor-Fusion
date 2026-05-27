# Reinforcement Learning for Intelligent Maintenance with Adaptive Sensor Fusion

This repository contains the code implementation for the paper:  
**"Reinforcement Learning–Driven Intelligent Maintenance Scheduling with Adaptive Sensor Fusion for Industrial Systems"**

## Overview

The project proposes a two‑stage RL framework:
- **Stage 1:** A PPO agent adaptively fuses vibration, temperature, and acoustic signals into a health indicator.
- **Stage 2:** A DQN agent uses the health indicator to schedule maintenance actions (operate/maintain).

The code includes:
- Synthetic multi‑sensor data generation (with concept drift)
- Feature extraction (moving RMS, mean, peak)
- PPO training for adaptive sensor fusion
- DQN training for maintenance scheduling
- Evaluation scripts and result plotting

## Requirements

- Python 3.8+
- `gym==0.26.2`
- `stable-baselines3`
- `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`
