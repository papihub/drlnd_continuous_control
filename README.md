# drlnd_continuous_control
Deep Reinforcement Learning - Project Continuous Control

![DDPG agent controlling a Double join arm](https://video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif)

This repository contains a Deep Reinforcement learning agent based on DDPG algorithm. This agent controls a double jointed arm. The goal of the agent is to maintain its position at the target location for as many time steps as possible. The target location is non stationary. The agent receives a reward of +0.1 for each step that the agent's hand is in the goal location and 0 otherwise.

## Dependencies
I developed code in this repository on a windows 10 64bit OS. So, I havent tested if this code works on any other OS.

**Miniconda**: Install miniconda3 from [miniconda download page](https://docs.conda.io/en/latest/miniconda.html)

**Python**: Follow the instructions in [DRLND Github repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your python environment. These instructions will guide you to install PyTorch, ML-Agents toolkit and a couple of other python packages required for this project.

**Unity Environment**: Download the unity environment from [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip). This is for windows 10 64bit OS. Please refer to the course material if you want the environment for a different OS.

## Instructions to train the agent:
Install the dependencies above.

open a jupyter notebook.

Run the notebook continuous_control.ipynb to train and test the agent. The notebook has instructions to load a saved model and to save a trained model.

Refer to the Readme.md for the approach used by the agent, training and evaluation.
