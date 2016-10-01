# -*- coding: utf-8 -*-
#python 3.5
import pandas as pd
import numpy as np

def print_frequency(data, col, format="count"):
    norm=True if format=="percent" else False
    print(format,'-')
    print(data[col].value_counts(sort=False, normalize=norm))


if __name__=="__main__":
    dataset="ADDHEALTH"
    data=pd.read_csv("addhealth_pds.csv", low_memory=False)
    # bug fix for display formats to avoid run time errors
    pandas.set_option('display.float_format', lambda x:'%f'%x)
    #Subset of respondents who gave a response
    subset=data[~((data["H1RE4"]==6) | (data["H1RE4"]==8))]#refused and dont know responses for religion importance
    subset=subset[~((subset["H1PF15"]==6) | (subset["H1PF15"]==8))]#refused and dont know responses for optimism
    subset2=subset.copy()
    #split the data into two groups- with and without religion
    #without religion might be useful later
    dont_have_religion=subset2[subset2["H1RE1"].isin([0, 96, 98, 99])]
    have_religion=subset2[~subset2["H1RE1"].isin([0, 96, 98, 99])]
    data_parts=[have_religion, dont_have_religion]

    cat_to_rating={i:6-i for i in range(1,6)}

    #for second analysis
    have_religion2=have_religion.copy()
    dont_have_religion2=dont_have_religion.copy()


    '''
    Factors for self esteem:

    * good qualities=h1pf30
    * lot to be proud of=h1pf32
    * like the way you are=h1pf33
    * socially accepted=h1pf35
    * feel loved and wanted=h1pf36

    '''

    self_esteem=["H1PF30", "H1PF32", "H1PF33", "H1PF35", "H1PF36"]
    for col in self_esteem:
        have_religion2=have_religion2[~((have_religion2[col]==6) | (have_religion2[col]==8))]
        dont_have_religion2=dont_have_religion2[~((dont_have_religion2[col]==6) | (dont_have_religion2[col]==8))]
    data_parts2=[have_religion2, dont_have_religion2]

    for index,part in enumerate(data_parts2):
        for col in self_esteem:
            part[col]=part[col].map(cat_to_rating)
            print_frequency(part, col, "count")
    #new variable for self-esteem as sum of above five variables
    for part in data_parts2:
        part["SELF_ESTEEM"]=((part["H1PF30"]+ part["H1PF32"]+ part["H1PF33"]+ part["H1PF35"]+ part["H1PF36"])/5.0).round()

    cols=["H1RE4", "H1PF15", "H1PF33"]
    #convert the optimism and like-as-i-am variable into ratings which is more intuitive
    for part in data_parts:
        for col in cols[1:]:
            part[col]=part[col].map(cat_to_rating)

    verbose=["\nVariable %d (%s): Importance of Religion",
             "\nVariable %d (%s): Respondents getting upset by difficult problems",
             "\nVariable %d (%s): Like the way you are"]
    print("\nFrequency distribution of for %s dataset:" %dataset)
    for index,col in enumerate(cols):
        print(verbose[index] %(index+1, col))
        print_frequency(have_religion, col, "count")
        print_frequency(have_religion, col, "percent")

    print("\nAdditional secondary variable SELF_ESTEEM:\n")
    for entry in ["SELF_ESTEEM"]:
        print_frequency(have_religion2, entry, "count")
        print_frequency(have_religion2, entry, "percent")


    #sanity check
    for index,val in enumerate(have_religion2["SELF_ESTEEM"]):
        if val>25:
            print("Error at index: ",index)
            print(have_religion2[["SELF_ESTEEM"]+self_esteem].iloc[index])
