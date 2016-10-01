# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn
import statsmodels.formula.api as smf


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


    #Subset of respondents who gave a response
    subset=data[~((data["H1RE4"]==6)|(data["H1RE4"]==8))]#refused and dont know responses for religion importance
    subset=subset[~((subset["H1PF15"]==6)|(subset["H1PF15"]==8))]#refused and dont know responses for optimism
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
        have_religion2=have_religion2[~((have_religion2[col]==6)|(have_religion2[col]==8))]
        dont_have_religion2=dont_have_religion2[~((dont_have_religion2[col]==6)|(dont_have_religion2[col]==8))]
    data_parts2=[have_religion2, dont_have_religion2]


    for index,part in enumerate(data_parts2):
        for col in self_esteem:
            part[col]=part[col].map(cat_to_rating)

    #new variable for self-esteem as sum of above five variables
    for part in data_parts2:
        part["SELF_ESTEEM"]=((part["H1PF30"]+ part["H1PF32"]+ part["H1PF33"]+ part["H1PF35"]+ part["H1PF36"])/5.0).round()


    cols=["H1RE4", "H1PF15", "H1PF33"]
    #convert the being-upset and like-as-i-am variable into ratings which is more intuitive
    for part in data_parts:
        for col in cols[1:]:
            part[col]=part[col].map(cat_to_rating)

    def cat_to_bin(row):
        return 1 if row["H1RE4"]<=2 else 0
    have_religion["H1RE4"] = pd.to_numeric(have_religion["H1RE4"], errors="coerce")
    have_religion["RELIGION_BINARY"]=have_religion.apply(lambda row:cat_to_bin(row), axis=1)
    have_religion["H1PF15"] = pd.to_numeric(have_religion["H1PF15"], errors="coerce")

    print ("OLS regression model for the association between Importance of religion and Getting Upset by difficulties")
    reg1=smf.ols("H1PF15 ~ RELIGION_BINARY", data=have_religion).fit()
    print(reg1.summary())
    print("Frequency distribution of RELIGION_BINARY variable:")
    print_frequency(have_religion, "RELIGION_BINARY")
    print("Cross-table:")
    print (pd.crosstab(have_religion["RELIGION_BINARY"], have_religion["H1PF15"]))

    have_religion2["H1RE4"] = pd.to_numeric(have_religion2["H1RE4"], errors="coerce")
    have_religion2["RELIGION_BINARY"]=have_religion2.apply(lambda row:cat_to_bin(row), axis=1)
    have_religion2["SELF_ESTEEM"] = pd.to_numeric(have_religion2["SELF_ESTEEM"], errors="coerce")

    print ("OLS regression model for the association between Importance of religion and Self-esteem")
    reg2=smf.ols("SELF_ESTEEM ~ RELIGION_BINARY", data=have_religion2).fit()
    print(reg2.summary())
    print("Frequency distribution of RELIGION_BINARY variable:")
    print_frequency(have_religion2, "RELIGION_BINARY")
    print("Cross-table:")
    print (pd.crosstab(have_religion2["RELIGION_BINARY"], have_religion2["SELF_ESTEEM"]))
    pass
