# LAB 4 - Curve Fitting
This project uses toolkits from numpy, scipy and sklearn to fit different different functions.
Those functions being polynomials, oscillating functions (sin) and multivariable functions.

## Installation
The file used is a simple ipynb file which can be ran using python w/ jupyter-notebook or with the jupyter-notebook package installed. graph.ipynb serves as the entrypoint for this project.

## Regression Methods
The following documentation outlines the different forms of regression used and the python packages used to do the regression.
### Polynomial Regression
Numpy's polyfit command is used to curve fit an arbitrary polynomoial created via desmos
### Ridge Regression
Sklearn's Ridge and polynomial curve fitting is also used to fit the polynomial created in the previous portion. 
### Linear Regression
Sklearn's linear regression package is also used to fit the original polynomial
### Damped Sine wave
Numpy's polyfit command is used to fit the first curve in a damped sin function. The polyfit package is then used again to fit the predicted curve/
### Arbitry tone fit with noise
An arbitray sine tone is converted to a csv and processed using pandas. The tone data has noise added which is then fit using numpy's polynomial regression. 
### Multivariable Regression
Using curve_fit from sklearn's optimize package, an arbitray multivariable function is predicted. The prediction results are varied based on the initial guess used.