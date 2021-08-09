# Morphology Computation shapes Integrated Information in Embodied Agents
## Introduction
This repository contains the code and sampled data of the experiments described in the paper "How Morphological Computation shapes Integrated Information in Embodied Agents" https://arxiv.org/pdf/2108.00904.pdf (Preprint). 

In this paper we analyze the information flow among the world, an agent's body and its environment. In this small toy example the agents consist of a body with two sensors, they are moving in a racetrack and their goal is to not touch the walls within the next two steps. The implementation of the racetrack and the design of the agents is due to Nathaniel Virgo.
We do not use a learning algorithm, but calculate the optimal behavior of the agent using the concept "Planning as Inference". In order to do that we need to sample the environment. 

## The Sampling

In order to get the empirical distribution of how the agent perceives its environment, we sample the sensory and motor values of agents performing two random movements. The code of the sampling can be found in the "sampleWorldandGoal.py" file. The random movements can be seen in the video below.  

https://user-images.githubusercontent.com/21078779/128511657-26f15662-3abc-4723-be36-f4d489aab8f5.mp4

The sampled data for the sensor length 0.5 to 2.75 can be found in the "SampledData" foulder. The agents in the video above have a sensor lenth of 1.5. 

## Planning as Infernce

Following the concept of "Planning as Inference" we optimize the likelihood of success by projecting iteratively to a goal manifold and an agent manifold, as described in the appendix of https://arxiv.org/pdf/2108.00904.pdf. These projections are implemented in the file "emAlgorInference.py" in the methods 
"conditioning" (e-projection) and "factorizing" (m-projection). The method "emAlg" calls for every sensor length the method "emAlgit" in which the iteration is implemented. Then the different measures are calculated and written in form of a vector to a file specified in the "main.py" function. This is done 3 times for one random input distribution, namely for the fully coupled agents, the controller driven agents and the reactive control agents.

## Different Measures of the Information Flow

The method "calculateMeasures" in "emAlgorInference.py" has 26 output values corresponding to 13 different measures, each evaluated for two timesteps. The measures described in the paper are in the following positions: Morphological computation (9 + 22), Integrated Information (2 + 15), Reactive Control  (1 + 14), Action Effect (10 + 23), Sensory Information (7 + 20), Control (6 + 19), Total Information Flow (5 + 18). The value at the 26th index is the probability of achieving the goal.  

## Plotting

The results from the 100 input distributions described in the paper are in the file "plottingdata.py". The plots in the paper were generated with the code in "plot.py".


