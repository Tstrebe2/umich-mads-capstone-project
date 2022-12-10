# Classification of Chest X-ray images via Convolutional Neural Networks
> Summary of project [_here_](https://www.example.com). <!-- If you have the project hosted somewhere, include the link here. -->

## Table of Contents
* [General Info](#general-information)
* [Features](#features)
* [Screenshots](#screenshots)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)
<!-- * [License](#license) -->


## General Information
- Authors are Austin Geary, Tim Strebel, and Will Dougall.
- This project was conducted as the Captsone Project for the Masters in Applied Data Science (MADS) program in the School of Information at University of Michigan.
- We attempted to create a CNN model that would perform at a level sufficient for Radiology departments in clinical settings to use to determine the presence or absence of pneumonia in patient lungs.
- The reason for this undertaking is that skilled radiologists are in short supply and are often over-worked. AI can relieve the workload of human radiologists as well as accelerate the process of providing test results to physicians.
<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Features
- Modular collection of scripts for model training. Can use DenseNet121, ResNet18, and AlexNet as well as two different datasets called RSNA and CX14.
- Notebooks for producing visualizations of patient metadata, as well as model evaluation. A technique called grad-cam was used to visually highlight areas of activation by the model for positive cases.
- Utilized PyTorch lightning to abstract away parallelization and model training loops to streamline code.


## Screenshots
![Precision Recall Curve for best DenseNet121 model](https://raw.githubusercontent.com/Tstrebe2/umich-mads-capstone-project/main/figures/rsna-auprcs.png)
<!-- Show the Precision Recall curve for our best model. -->

![ROC Curve for best DenseNet121 model](https://raw.githubusercontent.com/Tstrebe2/umich-mads-capstone-project/main/figures/rsna-aurocs.png)
<!-- Show the ROC curve for our best model. -->

## Setup
Environment requirements are listed in a file called requirements.txt located in the root of the repository.
Datasources are listed in a file called img-data-source-readme.txt located in the data folder.

## Usage
For training the models, open up a command line, navigate to the src directory containing train.py, and then execute the script and pass in arguments found in the args.py file.

Example:
`python3 train.py --model densenet --epochs 5 --targets_path ../../data/rsna-targets.csv --image_dir ../../data/chest-xray-14/images --freeze_features All --init_learning_rate 3e-3`

A similar workflow should be followed for testing the model, but instead of executing train.py, execute test.py. Both train.py and test.py use the same set of args.

For running notebooks, simply launch a Jupyter Notebook server session and navigate to the notebooks directory.

## Project Status
Project is: _no longer being worked on_. The reason for this is our semester came to a close, but there are many other paths we could take to build upon the work done thus far.


## Acknowledgements
- This project was based on [this Kaggle competition](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/overview).
- Many thanks to the MADS Staff at the University of Michigan 
- Many thanks to Dr. Amilcare Gentili and Michael J. Kim from the VA Healthcare system for agreeing to be interviewed for our project.


## Contact
Created by @Tstrebe2, @austingeary and @Zbandit98


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->
