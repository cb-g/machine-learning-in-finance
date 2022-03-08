# [machine-learning-in-finance](https://github.com/cb-g/machine-learning-in-finance)

Attempt to implement Python code examples from https://github.com/mfrdixon/ML_Finance_Codes in Julia. 

## [Chapter4-NNs](https://github.com/cb-g/machine-learning-in-finance/tree/main/Chapter4-NNs)

### [ML_in_Finance-Bayesian-Neural-Network.jl](https://github.com/cb-g/machine-learning-in-finance/blob/main/Chapter4-NNs/ML_in_Finance-Bayesian-Neural-Network.jl)

The original Python implementation is [ML_in_Finance-Bayesian-Neural-Network.ipynb](https://github.com/mfrdixon/ML_Finance_Codes/blob/master/Chapter4-NNs/ML_in_Finance-Bayesian-Neural-Network.ipynb).

The half-moon data creation is based on [scikit-learn's make_moons()](https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09b/sklearn/datasets/_samples_generator.py#L723). 

[MLJBase](https://github.com/JuliaAI/MLJBase.jl) offers a [make_moons function](https://github.com/JuliaAI/MLJBase.jl/blob/4a8f3f323f91ee6b6f5fb2b3268729b3101c003c/src/data/datasets_synthetic.jl#L256) for Julia. 

The neural network architecture is adapted from and closely follows [this Turing.jl tutorial here](https://turing.ml/dev/tutorials/03-bayesian-neural-network/#generic-bayesian-neural-networks) and [here](https://github.com/TuringLang/TuringTutorials/blob/master/notebook/03-bayesian-neural-network/03_bayesian-neural-network.ipynb) respectively.

![](https://github.com/cb-g/machine-learning-in-finance/blob/main/Chapter4-NNs/ML_in_Finance-Bayesian-Neural-Network_Plot_3.png?raw=true)

## [Chapter9-Reinforcement-Learning](https://github.com/cb-g/machine-learning-in-finance/tree/main/Chapter9-Reinforcement-Learning)

### [ML_in_Finance_FCW.jl](https://github.com/cb-g/machine-learning-in-finance/blob/main/Chapter9-Reinforcement-Learning/ML_in_Finance_FCW.jl)

The original Python implementation of the financial cliff walking problem is [ML_in_Finance_FCW.ipynb](https://github.com/mfrdixon/ML_Finance_Codes/blob/master/Chapter9-Reinforcement-Learning/ML_in_Finance_FCW.ipynb), which is based on [cliff_walking.py](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter06/cliff_walking.py), which in turn is a Julia-to-Python implementation from [Chapter06_Cliff_Walking.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningAnIntroduction.jl/blob/master/notebooks/Chapter06_Cliff_Walking.jl).
