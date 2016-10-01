# -*- coding: utf-8 -*-
import pandas as pd
#import numpy as np
import statsmodels.formula.api as smf
#import statsmodels.stats.multicomp as multi


def print_frequency(data, col, format="count"):
    norm=True if format=="percent" else False
    print(format,'-')
    print(data[col].value_counts(sort=False, normalize=norm))


if __name__=="__main__":
    dataset="ADDHEALTH"
    data=pd.read_csv("addhealth_pds.csv", low_memory=False)

    #Set PANDAS to show all columns in DataFrame
    pd.set_option('display.max_columns', None)
    #Set PANDAS to show all rows in DataFrame
    pd.set_option('display.max_rows', None)

    print("\nExplanatory var- Feeling really sick in the past year (H1GH10)")
    print("Response var- Weight (H1GH60)\n")

    data["H1GH10"]=data["H1GH10"].convert_objects(convert_numeric=True)
    data["H1GH60"]=data["H1GH60"].convert_objects(convert_numeric=True)
    #Subset of respondents who gave a response
    subset=data[~((data["H1GH60"].isin([996, 998, 999])) | (data["H1GH10"].isin([6, 8])))] #valid weights/feeling sick responses
    subset2=subset.copy()

    model=smf.ols(formula='H1GH60 ~ C(H1GH10)', data=subset2)
    results=model.fit()
    print(results.summary())
    subset3=subset2[["H1GH10", "H1GH60"]]
    print("\nMean by category:")
    print(subset3.groupby("H1GH10").mean()) #mean by category
    print("\nStandard deviation by category:")
    print(subset3.groupby("H1GH10").std()) #std by category
