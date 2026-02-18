# Reinforcement Learning Project 

### Project Description
In this project, I implemented reinforcement learning algorithms learned during DS442 class using Python 3.9 to
improve the performance of the AI agent under two scenarios. 

I installed Gymnasium, an open source Python library (that used to be maintained by Open AI) for developing and comparing reinforcement learning algorithms by
providing a standard API to communicate between learning algorithms and environments. 
Use the command below: pip install gymnasium

#### Question-1 Solving Blackjack Using Model-Free RL
Blackjack is a card game where the goal is to beat the dealer by obtaining cards that
sum to closer to 21 (without going over 21) than the dealers cards. In the Gymnasium library, there is a separate
environment for Blackjack that we will be using for this question. The description of
this environment and corresponding parameters (action space, observation space, and
reward rules) can be checked though this [link](https://gymnasium.farama.org/environments/toy_text/blackjack/).

This is the environment to set up Blackjack correctly.

```
import gymnasium as gym

env = gym.make("Blackjack-v1", render_mode="human") # Initializing environments

observation, info = env.reset()

for _ in range(50):

  action = env.action_space.sample() # agent policy that uses the observation and info
  
  observation, reward, terminated, truncated, info = env.step(action)
  
  if terminated or truncated:
  
    observation, info = env.reset()

env.close()
```

After running test_q1.py, the GUI will appear (which means all packages are set correctly).

##### Q1.2
To win this game, your card sum should be greater than the dealers without exceeding 21. The AI agent needs to be trained through Q-learning and played optimally with this environment setting below (solution_q1.py):

```
import gymnasium as gym

env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode=”human”)

###YOUR Q-LEARNING CODE BEGINS

###YOUR Q-LEARNING CODE ENDS
```

Implement a Q-learning algorithm in solution_q1.py (print out whether the bot won or lost in each
episode and the total win rate), through which you can learn the optimal policy for playing Blackjack in
this environment. Write code that starts off by initializing the Q(s,a) table for all
possible values of s and a. And then act according to Q-learning, and every time you act, you will
have a chance to update your Q(s,a) table. Over time, if you implement Q-learning properly, you should
be able to arrive at the optimal Q(s,a) values, which should enable you to act optimally.

#### Questions-2 Solving Frozen Lake Using Model Based RL
The Frozen lake environment on Gymnasium involves crossing a frozen lake from start
to goal without falling into any holes by walking over the frozen lake. The player may
not always move in the intended direction due to the slippery nature of the frozen lake.
The description of this environment and corresponding parameters (action space,
observation space, and reward rules) can be checked though this link.

Create the test_q2.py, copy the content below into your test_q2.py, and run test_q2.py to make sure the
environment of Frozen Lake is set correctly.

```
import gymnasium as gym

env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", render_mode=”human”,is_slippery=True, ) #initialization

observation, info = env.reset()

for _ in range(50):

  action = env.action_space.sample() # agent policy that uses the observation and info

  observation, reward, terminated, truncated, info = env.step(action)

  if terminated or truncated:

    observation, info = env.reset()

env.close()
```
After running test_q2.py, you will see the GUI (which means all packages are set correctly).

##### Q2.2 Use a random policy to learn the underlying MDP
Write code that will execute a random policy for 1000 episodes, and collect the training data that you
observe in those 1000 episodes. You can use the code that is given in Q2.1 as a starting point.
Using the training data for 1000 episodes, estimate the transition function T(s’/s,a) and the reward
function R(s,a,s’).

##### Q2.3 Implement value iteration to output the optimal value function
Using the transition function T(s’/s,a) and the reward function R(s,a,s’) learned in Q2.2, implement the
value iteration algorithm which will help you uncover the optimal value function.

##### Q2.4 Implement policy extraction
Using the learned value function in Q2.3, implement the policy extraction algorithm which will help you
uncover the optimal policy (that is associated with the learned value function of Q2.3).

##### Q2.5 Write code for acting inside the Frozen Lake policy (which follows the optimal policy you extract)
Using the extracted policy in Q2.4, write code similar to the sample code given in Q2.1 (except for the
fact that in Q2.1, the action is chosen randomly at each state, whereas in your code for Q2.5, we expect
that each action is chosen according to the optimal policy that was extracted in Q2.4).
At a high level, we expect solution_q2.py to look like the following. Your code will run the 1000 episodes
for random exploration without the GUI and start the GUI to show how the optimal policy performs.

```
import gymnasium as gym

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode=”human”)

#Your code for Q2.2 which Executes Random Policy until 1000 episodes
#Your code for Q2.3 which implements Value Iteration
#Your code for Q2.4 which implements Policy Extraction
#Your code for Q2.5 which executes the optimal policy
```
