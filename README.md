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

This research advances efforts to mitigate global warming by developing and evaluating a NILM-based multi-agent recommendation system designed to optimize household appliance energy usage and enhance renewable energy integration. The system provides precise energy monitoring and actionable recommendations to lower energy costs and reduce environmental impact. The NILM model developed in this theis is based on the research conducted by Massidda et al. (2020). The methodologies for training and testing are adapted from the existing repository (https://github.com/lmssdd/TPNILM), which provides a framework for implementing the techniques discussed in the research. Additionally, the study adapts the research by Zharova et al. (https://github.com/AlonaZharova/Explainable_multi-agent_RecSys.git)

The results showcase the system's ability to accurately identify appliances, deliver effective energy-saving recommendations, and significantly cut CO2 emissions, establishing a robust framework for promoting energy efficiency and renewable energy use, contributing  to global environmental sustainability efforts.
Futhermore, the project includes successful application of transfer learning for improved model adaptability. 

**Keywords**: Non-Intrusive Load Monitoring (NILM), Energy Efficiency, Renewable Energy Integration, Recommendation System, Transfer Learning, CO2 Emission Reduction



## Run experiments

Use nilm.ipynb to train and evaluate models for NILM tasks:

- Fit a NILM model for your household dataset.
- Evaluate model performance on the test dataset and save results, including metrics and predicted consumption for each device. Alternatively you can use NILM_Agent from the agents.py file
- Save pretrained weights, making it possible to resume training or use the model for transfer learning or evaluation.

Use transfer_model.ipynb:

- Transfer pretrained NILM models to new datasets.
- Fine-tune these models for target datasets and save the results.
- Ensure the source-trained models are saved as checkpoints to enable transfer learning.

Use grid_search.ipynb:

- Perform grid search experiments to tune hyperparameters for models.
- Compare different configurations of models (e.g., Random Forest, XGBoost) and save the best-performing parameters for further experiments.

Use co2_reductions_results.ipynb:

- Analyze the impact of model-based energy reductions on CO2 emissions.
- Calculate reductions in energy consumption and emissions across scenarios.


Use recommendation_output.ipynb:
- Generate personalized recommendations using the models trained in nilm.ipynb and transfer_model.ipynb.
- Evaluate recommendations for specific datasets and save outputs as CSVs for further review.




## Results
- NILM Model

![alt text](https://github.com/MargaritaTav/Energy-Consumption-Optimisation-in-households/blob/main/images/image.png?raw=true)

- Availability and Usage Agent
![alt text](https://github.com/MargaritaTav/Energy-Consumption-Optimisation-in-households/blob/main/images/image-1.png?raw=true)
  
- Transfer Learning Model 
![alt text](https://github.com/MargaritaTav/Energy-Consumption-Optimisation-in-households/blob/main/images/image-2.png?raw=true)
## Project structure


```bash
├── README.md
├── images                                          -- includes images for the README
├── data                                            -- stores data generated by NILM                                       
└── src
    ├── agents.ipynb                                -- includes all necessary agents for the recommendation system 
    ├── helper_functions.ipynb                      -- stores helper functions
    ├── nilm.ipynb                                  -- runs nilm process
    └── recommendation_output.ipynb                 -- generates recommendations
    └── grid_search.ipynb                           -- performs grid search 
    └── co2_reductions_results.ipynb                -- calculates co2 emissions                
```
