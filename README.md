# Predicting Used Car Price
<p align="center">
  <img src="images/used_car_clipart.png"
  width="400"
  height="300"
  alt="Used car clip art">
</p>

# Project Overview
**Background:** A regression problem predicting the price of used cars from vehicle make, model, mileage, and other specifications to implement into a customer-facing app.

**Purpose:** Train a regression model that accurately and quickly predicts the market value of a new customer's car, minimizing RMSE.

**Techiniques:** One Hot Encoding, Pipelines, GridSearchCV, Linear Regression, Gradient Boosting with CatBoost and LightGBM.

# Installation and Setup

## Codes and Resources Used

  - <b>Editor Used</b>: Visual Studio Code
  - <b>Python Version</b>: 3.10.9

## Python Packages Used

  - <b>General Purpose</b>: ```math, numpy, re, time```  
  - <b>Data Manipulation</b>: ```pandas```  
  - <b>Data Visualization</b>: ```matplotlib, seaborn```  
  - <b>Machine Learning</b>: ```sklearn```  
  - <b>Gradient Boosting</b>: ```catboost, lightgbm```

# Data

## Source Data

<b>Features</b>
* *vehicle_type* - vehicle body type
* *registration_year* - vehicle registration year
* *gearbox* - gearbox type
* *power* - power (hp)
* *model_top125* 125 most frequent vehicle models
* *mileage* - mileage (km)
* *registration_month* - vehicle registration month
* *fuel_type* - fuel type
* *brand_top20* - 20 most frequent vehicle brands
* *not_repaired* - vehicle repaired or not 
* *postal_code_trunc1000* - postal code truncated to 1000s place

<b>Targets</b>
 * *price* - used car price (euros)
 
## Data Acquisition

The data were provided by TripleTen's Data Science bootcamp. The full dataset is loaded into the notebook but is proprietary information and cannot be shared online.

## Data Preprocessing

Variables missing data were all missing less than 15% of observations. Categorical missing values were filled with 'unknown' and quantitative missing values were imputed with medians. Duplicates were cleaned from the dataset.

# Code Structure
```
  ├── LICENSE
  ├── README.md          
  │
  ├── images
  │   └── used_car_clipart.png    
  │
  └── notebooks  
      └── car_price_analysis.ipynb  
```

# Results and Evaluation
 
<p align="left">
  <img src="/images/pairplot.png"
  width="600"
  height="600"
  alt="sns pair plot of variables colored by receiving benefits">
</p>

There is clustering in insurance benefits by age.

<p align="left">
  <img src="/images/correlation_matrix.png" 
  width="500"
  height="300"
  alt="Results of binary classification model tuning">
</p>

The best binary classifier is a logistic regression with threshold optimized to 0.43. This model achieved an F1, ROC AUC, accuracy, precision, and recall of 1.0 on the cross-validated training data.

<p align="left">
  <img src="/images/train_results.png"
  width="310"
  height="90"
  alt="Test results of logistic regression with threshold = 0.43">
</p>

On the test set, the logistic regression with threshold = 0.43 again achieved scores of 1.0 across the board. This model is a perfect predictor of whether a person will use insurance benefits or not.

<p align="left">
  <img src="/images/test_results.png"
  width="690"
  height="250"
  alt="Results of multi class classification model tuning">
</p>

The random forest model with SMOTEENN balanced and weighted classes achieved the best multi-class classification results, scoring a macro-averaged F1 score of 0.9894.

<p align="left">
  <img src="/images/important_features.png"
  width="510"
  height="100"
  alt="Test results of random forest multi class classification">
</p>

The random forest model with SMOTEENN and class weighting achieved similar results on the test set, achieving an F1 macro of 0.9764. This is a well fitting model and can be trusted to predict how many benefits a customer will use.

# Conclusions and Business Application

**Conclusions:** LightGBM GBDT achieved the best model fit (RMSE test = 1663.83). Predictions from this model will offer customers the predicted value of their car within $1,663.83 on average. The most important features were predicting price were power, registration year, postal code, and mileage.  

**Business Application:** Rusty Bargain will be able to implement this model in their app and be confident that customers will receive accurate predictions in about 1 second. 

**Future Research:** With additional time, more hyperparameters and trees/iterations could be performed to improve model accuracy. Additionally, further data cleaning may improve the accuracy of the results.
