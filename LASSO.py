# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LassoLarsCV
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

if __name__=="__main__":

    """
    Target:
    H1DA10 hours of video game play in a week
    Features:
    H1DA11 hours of listen to radio per week
    H1DA9 hours of video watch in a week
    H1DA8 hours of tv watch in a week
    H1GI6A White
    H1GI6B Black or African American
    H1GI6C Am Indian or NAtive Am
    H1GI6D Asian or Pacific Islander
    H1GI6E Other race
    H1GH1 Health
    H1GH29 Trying to loose or gain weight
    H1GH51 hours of sleep
    H1GH52 get enough sleep
    H1GH60 weight
    H1ED3 skipped grade
    H1ED5 repeat grade
    H1PL1 physical condition
    H1SE4 perception of intelligence
    H1WP10 perception of mother caring
    H1PF7 never argue with anyone
    H1PF15 upset by difficult problems
    H1PF16 go with gut feeling
    H1PF35 feel socially accepted
    H1DS5 get in serious fight
    H1NB1 know most people in neighborhood
    H1RE4 importance of religion
    H1EE1 want to go to college

    """
    names={
           "H1DA11":"hours of listen to radio per week",
           "H1DA9":"hours of video watch in a week",
           "H1DA8":"hours of tv watch in a week",
           "H1GI6A":"White",
           "H1GI6B":"Black or African American",
           "H1GI6C":"Am Indian or NAtive Am",
           "H1GI6D":"Asian or Pacific Islander",
           "H1GI6E":"Other race",
           "H1GH1":"Health",
           "H1GH29":"Trying to loose or gain weight",
           "H1GH51":"hours of sleep",
           "H1GH52":"get enough sleep",
           "H1GH60":"weight",
           "H1ED3":"skipped grade",
           "H1ED5":"repeat grade",
           "H1PL1":"physical condition",
           "H1SE4":"perception of intelligence",
           "H1WP10":"perception of mother caring",
           "H1PF7":"never argue with anyone",
           "H1PF15":"upset by difficult problems",
           "H1PF16":"go with gut feeling",
           "H1PF35":"feel socially accepted",
           "H1DS5":"get in serious fight",
           "H1NB1":"know most people in neighborhood",
           "H1RE4":"importance of religion",
           "H1EE1":"want to go to college"
          }
          
    dataset="ADDHEALTH"
    data=pd.read_csv("addhealth_pds.csv", low_memory=False)
    NAs={"H1DA11":[996, 998], "H1DA9":[996,998] , "H1DA8":[996,998], "H1GI6A":[6,8], "H1GI6B":[6,8], "H1GI6C":[6,8], "H1GI6D":[6,8], "H1GI6E":[6,8],
         "H1GH1":[6,8], "H1GH29":[6,8], "H1GH51":[96, 98], "H1GH52":[6, 8], "H1GH60":[996,998,999], "H1ED3":[6,8],
    "H1ED5":[6,8], "H1PL1":[6,8], "H1SE4":[96,98], "H1WP10":[6,8], "H1PF7":[6,8], "H1PF15":[6,8], "H1PF16":[6,8],
    "H1PF35":[6,8], "H1DS5":[6,8,9], "H1NB1":[6,8,9], "H1RE4":[6,8], "H1EE1":[6,8]}
    t_NA={"H1DA10":[996, 998]}
    feats=list(NAs.keys())
    targets=list(t_NA.keys())
    subset=data[feats+targets]
    #target clean 
    subset[targets[0]]=pd.to_numeric(subset[targets[0]], errors="coerce")
    subset[targets[0]]=subset[targets[0]].replace(t_NA[targets[0]], np.nan)
    for col in feats:
        subset[col]=pd.to_numeric(subset[col], errors="coerce")
        subset[col]=subset[col].replace(NAs[col], np.nan)
    subset=subset.dropna()
    print("Dataset size:", len(subset))
    target=subset.H1DA10
    features=subset[feats]
    #standardize features with mean=0 and sd=1
    for col in feats:
        features[col]=preprocessing.scale(features[col].astype('float64'))
    pred_train, pred_test, tar_train, tar_test = train_test_split(features, target, test_size=.3, random_state=10)
    #lasso regression model
    model=LassoLarsCV(cv=10, precompute=False).fit(pred_train,tar_train)
    #variable names and regression coefficients
    mapping=dict(zip(features.columns, model.coef_))
    mapping_names={names[k]:v for k,v in mapping.items()}
    #pretty printing
    sorted_mapping=sorted(tuple(mapping_names.items()), key=lambda tup:tup[1])
    print("\nCoefficients:")
    for k,v in sorted_mapping:
        print("{}: {}".format(k,v)) 
    # plot coefficient progression
    m_log_alphas = -np.log10(model.alphas_)
    ax = plt.gca()
    plt.plot(m_log_alphas, model.coef_path_.T)
    plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k', label='alpha CV')
    plt.ylabel('Regression Coefficients')
    plt.xlabel('-log(alpha)')
    plt.title('Regression Coefficients Progression for Lasso Paths')
    plt.show()
    
    # plot mean square error for each fold
    m_log_alphascv = -np.log10(model.cv_alphas_)
    plt.figure()
    plt.plot(m_log_alphascv, model.cv_mse_path_, ':')
    plt.plot(m_log_alphascv, model.cv_mse_path_.mean(axis=-1), 'k', label='Average across the folds', linewidth=2)
    plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k', label='alpha CV')
    plt.legend()
    plt.xlabel('-log(alpha)')
    plt.ylabel('Mean squared error')
    plt.title('Mean squared error on each fold')
    plt.show()
    
    # MSE from training and test data    
    train_error = mean_squared_error(tar_train, model.predict(pred_train))
    test_error = mean_squared_error(tar_test, model.predict(pred_test))
    print('Training data MSE: ', train_error)
    print('Test data MSE: ', test_error)
    
    # R-square from training and test data
    rsquared_train=model.score(pred_train,tar_train)
    rsquared_test=model.score(pred_test,tar_test)
    print('Training data R-square: ', rsquared_train)
    print('Test data R-square: ',rsquared_test)


