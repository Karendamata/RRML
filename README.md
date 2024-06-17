# Reliability and Resilience of a Machine Learning Model in Adversarial Scenarios

## About The Project


This project demonstrates how to collect relevant data during the training and testing of machine learning (ML) models to assess the reliability and resilience of these ML models subject to adversarial attacks. For illustration, a convolutional neural network (CNN) model is implemented to classify the images in the CIFAR-10 data set. The CNN is then tested on a poisoned dataset containing real and adversarial images generated with two adversarial attacks, including the Fast Gradient Sign Method (FGSM) and the Projected Gradient Descent (PGD), and varying the amount of noise injected in the images. Adaptive adversarial training is also performed to improve the system's performance. Data related to defect discovery and performance throughout the lifecycle of the CNN model is collected and made available in two spreadsheets for analysis and application of the reliability and resilience models. 

## Getting Started

This repository includes the code implemented to collect data related to defect discovery and performance throughout the lifecycle of an ML model and two example spreadsheets to promote reproducibility.

### Prerequisites

On your terminal, install the packages in the requirements.txt by running:

    pip install -r /path/to/requirements.txt

### Installation and Usage

Change your directory to the Code folder and run:

    python main.py
    
Once the code has finished running, two spreadsheets containing the data collected will be created in the Data folder. The file ending in FC-Results.csv contains the failure count data and several other Features, while the file ending in FT-Results.csv contains the failure time data.  

If one is interested in exploring different ranges of noise injected in the images, feel free to modify the 'initial' and 'final' variables. This flexibility allows you to tailor the code to your specific needs.


## Example Spreadsheets

In both data sets, each file contains four sheets representing four testing scenarios, indicating the noise range injected in the adversarial attacks simulated.    
_________
 - Failure time data for the application of reliability models without covariates:

        FailureTimeDatasets.xlsx 
_________
 - Failure Count data, including covariates, for the application of reliability and resilience models with covariates:

        FailureCountDatasets.xlsx 

### Features references

| Acronym           | Description             |
| ----------------- | ----------------------- |
| FN    | Failure number | 
| FT	| Failure time by image | 
| T	    | Iteration | 
| FC    | 	Failure count | 
| Alpha	| Learning rate used for training | 
| Epsilon   | 	Amount of noise injected in images | 
| FGSM_perc | 	Percentage of adversarial attacks generated using FGSM | 
| PGD_perc  | 	Percentage of adversarial attacks generated using PGD | 
| F1_Score  | 	F1 Score of the model considering testing data set | 
| Test_Accuracy | 	Accuracy of the model considering testing data set | 
| Test_Loss | 	Loss of the model considering testing data set | 
| Train_Accuracy    | 	Accuracy of the model considering training data set | 
| Train_Loss    | 	Loss of the model considering training data set | 
| Val_Accuracy  | 	Accuracy of the model considering validation data set | 
| Val_Loss  | 	Loss of the model considering validation data set | 
| Memory    | 	Amount of memory used | 
	
## License 

Distributed under the MIT License.

## Contact

Karen da Mata - [Linkedin](https://www.linkedin.com/in/karendamata) - kalvesdamata@umassd.edu

Lance Fiondella - [Linkedin](https://www.linkedin.com/in/lance-fiondella-0b04695/) - [Fiondella Lab](https://https://lfiondella.sites.umassd.edu) - lfiondella@umassd.edu