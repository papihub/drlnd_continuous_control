# drlnd_continuous_control
Deep Reinforcement Learning - Project Continuous Control

![DDPG agent controlling a Double join arm](https://video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif)

This repository contains a Deep Reinforcement learning agent based on DDPG algorithm. This agent controls a double jointed arm. The goal of the agent is to maintain its position at the target location for as many time steps as possible. The target location is non stationary. The agent receives a reward of +0.1 for each step that the agent's hand is in the goal location and 0 otherwise.

## Dependencies
I developed code in this repository on a windows 10 64bit OS. So, I havent tested if this code works on any other OS.

**Miniconda**: Install miniconda3 from [miniconda download page](https://docs.conda.io/en/latest/miniconda.html)

**Python**: Follow the instructions in [DRLND Github repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your python environment. These instructions will guide you to install PyTorch, ML-Agents toolkit and a couple of other python packages required for this project.

**Unity Environment**: Download the unity environment from [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip). This is for windows 10 64bit OS. Please refer to the course material if you want the environment for a different OS.

## Environment
In this environment the observation/state space has 33 dimensions corresponding to position, rotation, velocity, and angular velocities of the arms. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1. The action space in continuous.

The course provides two versions of this environment. Version 1 is for a single agent and version 2 has 20 identical agents each with its own copy of the environment. I chose to use version 2 with 20 agents.

The task is episodic. This means there is a distinct "done" state. In order to solve the environment, our agent must get an average of +30 reward over 100 consecutive episodes.

## Success criteria:
Our agent must get an average of +30 reward over 100 consecutive episodes.
After each episode, we get the mean reward (average of rewards from all 20 agents).
We need avearage of this mean reward to be > +30 over 100 consecutive episodes.

## Instructions to train the agent:
Install the dependencies above.

open a jupyter notebook.

Run the notebook continuous_control.ipynb to train and test the agent. The notebook has instructions to load a saved model and to save a trained model.

** The training step in this notebook took 7 hours on a CPU. I did not test the code on a GPU. **

## Approach : Reinforced Learning with the Deep Deterministic Policy Gradient (DDPG) approach
This agent trains using DDPG algorithm (link to llicrap) using four deep neural networks. Two of them correspond to an "Actor" and two to a "Critic". 

Actor takes current state as input and provides action as an output. There is a target actor network thats only used during training the actor. The target actor network is a time delayed copy of the actor network. Since we dont have a complete view of our environment, we use our actors current understanding of the environment as a target. Since our actor's understanding of the environment changes with every step, our target is non stationary. Neural networks learn better in a stable environment. So we try to keep the target for the actor relatively stable by making it a timedelayed copy of the actor.

Critic takes the action from the actor along with the current state as inputs. It provides a single value as an output. Critic's output is an evaluation of the actor's output. It helps the actor decide which direction to move while learning. Critic also uses a critic-target network while training. Simiar to actor-target, critic-target is a time delayed copy of critic and helps stablize critic's learning.

The agent maintains two replay buffers, a positive reward buffer and a negative reward buffer to store previously seen states, actions taken in those states, resulting rewards, next states returned by environment and whether its a done state or not. I noticed that having two buffers allowed the agent to learn faster. Since positive experiences are rare in the initial learning stages, I think maintaining two separate buffers and sampling evenly from them helped the agent learn better.

Initially the neural networks are initialized with random weights.

The agent interacts with the environment by taking actions and observing the reward and state changes.

When the agent takes an action in a current state, the environment returns the next state along with a reward and a *done* indicator if we reached the terminal state.

When the agent reaches the terminal state, the eposide terminates.

The agent maintains a tuple of ( state, action, reward, next state, done ) for all steps in an episode in two replay buffers. tuples with positive reward in a positive-replay buffer and tuples with negative rewards in a negative-replay buffer

The agent maintains as many episodes as its replay buffers can fit. In this case the replay buffers are set to hold 10,000 tuples.

The agent samples episodes from the positive and negative replay buffers evenly and trains its neural networks models.

The agent does not learn after every step. Instead it learns once every 100 steps ( hyper parameter )

The agent uses its 'actor' deep neural netowrks to come up with an action given the current state. It uses a 'critic' DNN to evaluate the actor's output.


**** update the below ****


For stability of the neural networks, the agent maintains a target network and a local network for both actor and critic networks. Actor local network is used to take action and is refeined in each learning step. The target network is only updated after a fixed number of learning steps.

In each learning step, the agent computes the difference between expected and predicted value and uses a learning rate along with a discount factor. It uses this difference (loss) to train the deep network.

### Network Architecture
The agent uses 2 identical deep neural networks to learn from the environment interactions.

The networks have 1 input layer, 1 hidden layer and 1 output layer.

All layers are fully connected.

Input layers has (state_size) inputs and (state_size)x10 outputs

Hidden layer has (state_size)x10 inputs and (state_size)x5 outputs

output layer has (state_size)x5 inputs and (action_size) outputs

State_size = 37 and action_size = 4 for this environment.

Input and hidden layers go thru an relu activation function.

We use a mean squared loss function to compute the loss values.

We use Adam optimizer to backpropogate the loss and update weights.

### Hyper parameters and their values
|Hyper parameter|Value|Comment|
|---------------|:---:|-------|
|Replay buffer size|1e5|BUFFER_SIZE|
|Discount Factor|0.99|GAMMA|
|How often do we learn?|4|UPDATE_EVERY|
|Minimum steps before we start learning|64|BATCH_SIZE|
|Factor for target network update|1e-3|TAU|
|Learning Rate|5e-4|ALPHA/LR|
|Initial epsilon|1.0|for exploration|
|Min epsilon|0.01||
|epsilon decay|0.995||
|*below are other parameters used in training the agent*|
|Max episodes|2000||
|Max steps in an episode|1000||
|window size to compute average score|100||

## Future actions / steps
In this implementation I sampled randomly from the replay buffer for training.

Implementing priority queues can allow some of the agent to learn from important steps more times than other steps.

Implement dual networks where we switch the target and local networks.

Learn directly from raw images than from encoded states!!
