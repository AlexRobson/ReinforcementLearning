# ReinforcementLearning
A repo to attempt to solve the environments in OpenAI gym using different algorithms, using the exercises and material in ReinforcementLearning: An Introduction [Sutton &amp; Barto]

Two implementations at present

* action-supervised: Runs the model, and if the model is successful, labels actions as successful, otherwise unsuccessful. The benchmark of success is gradually higher, leading to an improved model. 

* deep Q-learning: [This approach isn't working yet]. Deep Q-learning applied to cartpole. Probably overkill for a toy problem like this.  

# Useful Links:


* http://karpathy.github.io/2016/05/31/rl/

A tensorflow implementation of the CartPole problem using Deep-Q learning
* https://gist.github.com/arushir/04c58283d4fc00a4d6983dc92a3f1021
* https://www.nervanasys.com/demystifying-deep-reinforcement-learning/




# Installing Prerequistes#

```
mkvirtualenv RL
workon RL
pip install -r requirements.txt
```

# Running 
```
python cartpole.py
```


