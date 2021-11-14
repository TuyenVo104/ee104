# EE 104 Lab 5

## QuickStart

This section contains different ipynb notebooks that aim to solve different questions presented in the SJSU EE104 Class. In order to run the files Jupyter Notebook is required and can be the dependencies can be installed using `pipenv install` within the rooot of this repository. 

### Electron Force
The Electron Force.ipynb script finds the total amount of work needed to move an electron away from another electron for a distance up to 10 picometers

### HIC Numerical Integrattion
This script calculates the Head Injury Criterion Index for a crash test done in a specifed mercedes vehicle. The Crash test data is modeled with a trivial curvefit and the maximum impulse is used to find the HIC 

### RCL Circuit
An arbitray rlc circuit is modeled and the current of the circuit is found using scipy's ode solver.

### ER Modeling
This script models the impact of covid with a hospital. The hospital is modeled with two rooms: The Intesive Care Unit and Emergency care unit (ICU & ERU). The physical model of two cascaded containers is used to provide a base for this model. This section contains a csv with the total number of cases of covid as recorded by the CDC from the start of infection in the US. This data is used to model the input of the system.