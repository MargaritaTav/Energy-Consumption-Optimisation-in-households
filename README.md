# Energy Consumption Optimisation in households with a NILM-based Recommendation System

**Type:** Master's Thesis

**Author:** Margarita Tavkazakova


**1st Examiner:** Dr. Alona Zharova

**2nd Examiner:** Prof. Dr. Stefan Lessmann



## Table of Content

- [Summary](#summary)
- [Reproducing results](#Reproducing-results)
    - [Training code](#Training-code)
    - [Evaluation code](#Evaluation-code)
    - [Pretrained models](#Pretrained-models)
- [Results](#Results)
- [Project structure](-Project-structure)

## Summary

This research advances efforts to mitigate global warming by developing and evaluating a NILM-based multi-agent recommendation system designed to optimize household appliance energy usage and enhance renewable energy integration. The system provides precise energy monitoring and actionable recommendations to lower energy costs and reduce environmental impact. The NILM model developed in this theis is based on the research conducted by Massidda et al. (2020). The methodologies for training and testing are adapted from the existing repository (https://github.com/lmssdd/TPNILM), which provides a framework for implementing the techniques discussed in the research. The recommendation system uses methodologies of Zharova

The results showcase the system's ability to accurately identify appliances, deliver effective energy-saving recommendations, and significantly cut CO2 emissions, establishing a robust framework for promoting energy efficiency and renewable energy use, contributing  to global environmental sustainability efforts.
Futhermore, the project includes successful application of transfer learning for improved model adaptability. 

**Keywords**: Non-Intrusive Load Monitoring (NILM), Energy Efficiency, Renewable Energy Integration, Recommendation System, Transfer Learning, CO2 Emission Reduction



## Reproducing results


### Training code

-NILM Model

- Availability and Usage Agent

-Transfer learning

### Evaluation code

-Evaluate NILM

- Evaluate Availability and Usage Agent

- Evaluate transfer learning model

### Pretrained models

You can download the pretrained model here:

## Results
- NILM Model

  
- Availability and Usage Agent

  
- Transfer Learning Model 

## Project structure

(Here is an example from SMART_HOME_N_ENERGY, [Appliance Level Load Prediction](https://github.com/Humboldt-WI/dissertations/tree/main/SMART_HOME_N_ENERGY/Appliance%20Level%20Load%20Prediction) dissertation)

```bash
├── README.md
├── requirements.txt                                -- required libraries
├── data                                            -- stores csv file 
├── plots                                           -- stores image files
└── src
    ├── prepare_source_data.ipynb                   -- preprocesses data
    ├── data_preparation.ipynb                      -- preparing datasets
    ├── model_tuning.ipynb                          -- tuning functions
    └── run_experiment.ipynb                        -- run experiments 
    └── plots                                       -- plotting functions                 
```
