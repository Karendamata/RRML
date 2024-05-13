
# Reliability and Resilience of a Machine Learning Model

This GitHub includes two spreadsheets containing data related to defect discovery and performance throughout the lifecycle of a Machine Learning model to promote reproducibility.

In both data sets, each file contains four sheets representing four testing scenarios, which indicate the range of noise injected in the adversarial attacks simulated.  
_________
 - Failure time data for the application of reliability models without covariates:

        FailureTimeDatasets.xlsx 
_________
 - Failure Count data, including covariates, for the application of reliability and resilience models with covariates:

        FailureCountDatasets.xlsx 

## Features references

| Acronym           | Description             |
| ----------------- | ----------------------- |
| FN    | Failure number| 
| FT	| Failure time ( by image)| 
| T	    | Iteration| 
| FC    | 	Failure count| 
| Alpha	| Learning rate used for training| 
| Epsilon   | 	Amount of noise injected in images| 
| FGSM_perc | 	Percentage of adversarial attacks generated using FGSM| 
| PGD_perc  | 	Percentage of adversarial attacks generated using PGD| 
| F1_Score  | 	F1 Score of the model considering testing data set| 
| Test_Accuracy | 	Accuracy of the model considering testing data set| 
| Test_Loss | 	Loss of the model considering testing data set| 
| Train_Accuracy    | 	Accuracy of the model considering training data set| 
| Train_Loss    | 	Loss of the model considering training data set| 
| Val_Accuracy  | 	Accuracy of the model considering validation data set| 
| Val_Loss  | 	Loss of the model considering validation data set| 
| Memory    | 	Amount of memory used| 
	
