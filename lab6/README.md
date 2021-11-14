# Tradition Data Analysis Module

This lab focuses on traditional data analysis for a range of use cases.

Topics Covered Include:

* Data Relationship modeling based on caltrans incident information
* Risk Factor Identification report for loan defaults
* Job Salary distribution analysis
* Covid Trend Prediction using a Recurrent Neural Network

## Getting Started

All dependencies for this module can be installed via the supplied requirements.txt file in the root of this directory.

I recommend to use the provided docker/docker-compose file in order to get the tensorflow environment setup properly.

Once installed the ipynb's' can be run via jupyter

## Modules

### Risk Factor Identification.ipynb

Using the hmeq.csv, relevant factors that correlate to a loan defaulting are identified and a report is generated at Risk Factor.xlsx

### DataRelationship.ipynb

This module uses caltrans data and looks for a correlation between "distance traveled", "Number of Incidents","Delay Time", and "county". caltrans.csv is used

### Job Salary distribution Analysis

This module use traditional data cleaning methods to plot the job salary distributution outlined in Salaries.csv. As expected, the distribution is Gaussian.

### Covid Analysis.py

By using the first half of available covid data we are able to predict the trend of covid up to the current date. This is done using a 2 layer Recursive Neural Network. The overall trend is represented using the selu activation function, while the secondary layer used a sigmoid activation function. This layer proved to be very accurate.
