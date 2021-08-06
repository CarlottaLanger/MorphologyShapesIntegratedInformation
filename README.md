# MorphologyShapesIntegratedInformation
## Introduction
This repository contains the code and sampled data of the experiments described in the paper "How Morphological Computation shapes Integrated Information in Embodied Agents" https://arxiv.org/pdf/2108.00904.pdf (Preprint). 

In this paper we analyze the information flow among the world, an agent's body and its environment. In this small toy example the agents consist of a body with two sensors, they are moving in a racetrack and their goal is to not touch the walls within the next two steps. We do not use a learning algorithm, but calculate the optimal behavior of the agent using the concept "Planning as Inference".  

## The Sampling

In order to get the empirical distribution of how the agent perceives its environment, we sample the sensory and motor values of agents performing random tasks. The code of the sampling can be found in the "sampleWorldandGoal.py" file. The random movements can be seen in the video below.  

https://user-images.githubusercontent.com/21078779/128511657-26f15662-3abc-4723-be36-f4d489aab8f5.mp4

The sampled data for the sensor length 0.5 to 2.75 can be found in the "SampledData" foulder. 

## Planning as Infernce

Following the concept of "Planning as Inference" we optimize the likelihood of success by projecting iteratively to a goal manifold and an agent manifold, as described in the appendix of https://arxiv.org/pdf/2108.00904.pdf. These projections are implemented in the file "emAlgorInference.py" in the methods 
"conditioning" (e-projection) and the method "factorizing" (m-projection).

## Different Measures of the Information Flow

The method "calculateMeasures" in "emAlgorInference.py" has 26 output values. 

## Plotting


