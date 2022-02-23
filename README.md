# [machine-learning-in-finance](https://github.com/cb-g/machine-learning-in-finance)

Attempt to implement Python code examples from https://github.com/mfrdixon/ML_Finance_Codes in Julia. 

## [Chapter4-NNs](https://github.com/cb-g/machine-learning-in-finance/tree/main/Chapter4-NNs)

### [ML_in_Finance-Bayesian-Neural-Network.jl](https://github.com/cb-g/machine-learning-in-finance/blob/main/Chapter4-NNs/ML_in_Finance-Bayesian-Neural-Network.jl)

The original Python implementation is [ML_in_Finance-Bayesian-Neural-Network.ipynb](https://github.com/mfrdixon/ML_Finance_Codes/blob/master/Chapter4-NNs/ML_in_Finance-Bayesian-Neural-Network.ipynb).

The half-moon data creation is based on [scikit-learn's make_moons()](https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09b/sklearn/datasets/_samples_generator.py#L723). 
[MLJBase offers](https://github.com/JuliaAI/MLJBase.jl) a [make_moons function](https://github.com/JuliaAI/MLJBase.jl/blob/4a8f3f323f91ee6b6f5fb2b3268729b3101c003c/src/data/datasets_synthetic.jl#L256) for Julia. 

The neural network architecture is adapted from and closely follows [this Turing.jl tutorial here](https://turing.ml/dev/tutorials/03-bayesian-neural-network/#generic-bayesian-neural-networks) and [here](https://github.com/TuringLang/TuringTutorials/blob/master/notebook/03-bayesian-neural-network/03_bayesian-neural-network.ipynb) respectively.

![](https://github.com/cb-g/machine-learning-in-finance/blob/main/Chapter4-NNs/ML_in_Finance-Bayesian-Neural-Network_Plot_3.png?raw=true)
