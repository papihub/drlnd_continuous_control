# drlnd_continuous_control
Deep Reinforcement Learning - Project Continuous Control

![DDPG agent controlling a Double join arm](https://video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif)

This repository contains a Deep Reinforcement learning agent based on DDPG algorithm. This agent controls a double jointed arm. The goal of the agent is to maintain its position at the target location for as many time steps as possible. The target location is non stationary. The agent receives a reward of +0.1 for each step that the agent's hand is in the goal location and 0 otherwise.

## Environment
In this environment the observation/state space has 33 dimensions corresponding to position, rotation, velocity, and angular velocities of the arms. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1. The action space in continuous.

The task is episodic. This means there is a distinct "done" state. In order to solve the environment, our agent must get an average of +30 reward over 100 consecutive episodes.

## Success criteria:
Our agent must get an average of +30 reward over 100 consecutive episodes.
After each episode, we get the mean reward (average of rewards from all 20 agents).
We need avearage of this mean reward to be > +30 over 100 consecutive episodes.

## Approach : Reinforced Learning with the Deep Deterministic Policy Gradient (DDPG) approach
This agent trains using DDPG algorithm (https://arxiv.org/abs/1509.02971) using four deep neural networks. Two of them correspond to an "Actor" and two to a "Critic". 

Actor takes current state as input and provides action as an output. There is a target actor network thats only used during training the actor. The target actor network is a time delayed copy of the actor network. Since we dont have a complete view of our environment, we use our actors current understanding of the environment as a target. Since our actor's understanding of the environment changes with every step, our target is non stationary. Neural networks learn better in a stable environment. So we try to keep the target for the actor relatively stable by making it a timedelayed copy of the actor.

Critic takes the action from the actor along with the current state as inputs. It provides a single value as an output. Critic's output is an evaluation of the actor's output. It helps the actor decide which direction to move while learning. Critic also uses a critic-target network while training. Simiar to actor-target, critic-target is a time delayed copy of critic and helps stablize critic's learning.

The agent maintains two replay buffers, a positive reward buffer and a negative reward buffer to store previously seen states, actions taken in those states, resulting rewards, next states returned by environment and whether its a done state or not. I noticed that having two buffers allowed the agent to learn faster. Since positive experiences are rare in the initial learning stages, I think maintaining two separate buffers and sampling evenly from them helped the agent learn better.

Here is algorithm my agent is based on:

![DDPG Algorithm](https://miro.medium.com/max/542/1*BVST6rlxL2csw3vxpeBS8Q.png)

Initially the neural networks are initialized with random weights.

The agent interacts with the environment by taking actions and observing the reward and state changes.

When the agent takes an action in a current state, the environment returns the next state along with a reward and a *done* indicator if we reached the terminal state.

When the agent reaches the terminal state, the eposide terminates.

The agent maintains a tuple of ( state, action, reward, next state, done ) for all steps in an episode in two replay buffers. tuples with positive reward in a positive-replay buffer and tuples with negative rewards in a negative-replay buffer

The agent maintains as many episodes as its replay buffers can fit. In this case the replay buffers are set to hold 10,000 tuples.

The agent samples episodes from the positive and negative replay buffers evenly and trains its neural networks models.

The agent does not learn after every step. Instead it learns once every 100 steps ( hyper parameter )

The agent uses its 'actor' deep neural netowrks to come up with an action given the current state. It uses a 'critic' DNN to evaluate the actor's output.

For stability of the neural networks, the agent maintains a target network and a local network for both actor and critic networks. Actor local network is used to take action and is refeined in each learning step. The target network is only updated after a fixed number of learning steps. Target networks are not trained. They meerly get a scaled copy of the local networks.

In each learning step, the agents computes the difference between expected and predicted values and use a learning rate along with a discount factor to learn from the difference(loss) between expected and predicted values.

### Network Architecture
The agent uses 2 different deep neural networks to learn from the environment interactions.

Actor network:
1 input layer and 1 output layer.
All layers are fully connected.
Input layers has (state_size) inputs and (state_size)x(action_size) outputs
Output layer has (state_size)x(action_size) inputs and (action_size) outputs

State_size = 33 and action_size = 4 for this environment.

Input layer goes thru an leaky relu activation function.
Output layer goes thru a tanh activation function to ensure all output values are between -1 and 1

We use a mean squared loss function to compute the loss values.

We use Adam optimizer to backpropogate the loss and update weights.

Critic network:
4 layers all fully connected.
layer 1 takes state as input
layer 2 takes action as input
layer 3 takes the output from layer 1 & 2 concatenated
layer 4 is the output layer 

All these layers are fully connected.
layer 1 goes thru a leaky relu
layer 2 goes thru a tanh
layer 3 & 4 go thru a leaky relu

The critic also uses a mean squared loss function and an Adam optimizer to backpropogate the loss and udpate weights.


### Hyper parameters and their values
|Hyper parameter|Value|Comment|
|---------------|:---:|-------|
|Replay buffer size|10,000x2|BUFFER_SIZE. I used 2 buffers - one for positive rewards and another for zero rewards|
|Discount Factor|0.99|GAMMA|
|How often do we learn?|100|UPDATE_EVERY|
|No of experiences we use for learning|100|sample size|
|Factor for target network update|1e-3|TAU. Same TAU for both actor and critic|
|Learning Rate|1e-4|ALPHA/LR. Use the same learning rate for actor and critic. I used 1e-2 and the agent dint learn at all.|
|No of epochs during each training step|3| We sample 3 times from the buffers and learn from each of those sample sets|

## Training results and evaluation

The agent was trained for 200 episodes. During each training episode, in each step all 20 arms in the environment use the same copy of the agent to take actions. The results are captured and stored in replay buffers. After every 100 steps (hyper parameter), the agent perfroms 3 learning steps using 100 samples from the replay buffers. Sames are drawn evenly from each of the positive and negative buffers. At each step we collect the rewards from all 20 arms. At the end of each episode we average the scores from all 20 agents. The below graph shows these averaged scores for the first 200 Training episodes.
![Scores during Training](https://github.com/papihub/drlnd_continuous_control/blob/master/200_episode_scores.png)

Here is an evaluation run:
![Evaluation of trained agent](https://github.com/papihub/drlnd_continuous_control/blob/master/trained_agent.gif)

## Future actions / steps
In this implementation I sampled randomly from two replay buffer for training.

I want to try some of the other training methods like Proximal policy optimizing, TD3 and TRPO

May be create a real robotic arm and then use this agent to control it!! :)
