## Nuclear Power Plant Event Identification using PCA and SFS 
This project demonstrates an event identification system for the Maanshan Nuclear Power Plant using simulation data generated from the MAAP5 software. The methodology combines **Principal Component Analysis (PCA)** for dimensionality reduction with **Sequential Forward Selection (SFS)** for sensor selection, followed by classification based on mean squared error (MSE). 

The work was originally done as part of an academic project.

## Data Description
The dataset was generated using MAAP5, configured with parameters from the Maanshan Nuclear Power Plant. It consists of time-series sensor readings from a variety of simulated accident scenarios and is divided into training and testing sets.

Each data sample corresponds to one simulated incident and includes:

- **1020 incidents** across **23 initiating event types**, such as:
  - Small, medium, and large Loss-of-Coolant Accidents (LOCA) over cold-legs and hot-legs
  - Steam Generator Tube Rupture (SGTR)
  - Main Steam Line Break

- **29 sensors** per incident, measuring quantities like:
  - Coolant volume, temperature, pressure of the steam generator
  - Flow rates through cold and hot valves

- **Time-series data** over 60 seconds for each sensor

The simulation data is not included in this repository due to licensing restrictions associated with the MAAP5 software.

## Code Description

The identification system processes sensor data in two stages: training and testing.

### 1. Feature Extraction Using PCA

To reduce the size of the data, the sensor readings of each event is transformed into a reduced-dimensional representation using PCA.

- Each sensor’s value in each event is transformed from 60-second time-series data to a k principal components
- The principal components capture dominant variance directions and are used as features for classification.

### 2. Classification

Classification is done using the k-th nearest neighbor algorithm, with k=1. In other words, we find the data point from training data that has the lowest mean squared error (MSE) when compared with the testing data, and set the predicted class as the event of the data point. Since we have reduced the dimensionality of the datasets, the comparing process is sped up significantly.

The prediction accuracy is 80.8%

### 3. Classification with SFS

To improve the classification accuracy, the Sequential Forward Selection (SFS) algorithm is used to choose the most relevant sensors

- SFS identifies the subset of sensors that contribute most to classification accuracy.
- At each stage, the sensor whose inclusion yields the highest increase in performance is added to the selected set.
- The process stops when no further improvement is observed.

In the end, 9 sensors chosen were chosen, resulting an accuracy of 91.7%.

Here's the flowchart of the SFS algorithm:

![flowchart](/sfs.png)
