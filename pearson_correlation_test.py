# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn
import scipy
import matplotlib.pyplot as plt

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
       
    print("""Pearson correlation between-
    Weight (H1GH60) and
    Number of times inovolved in a serious fight (H1FV13):\n""")
    
    data["H1GH60"] = pd.to_numeric(data["H1GH60"], errors="coerce")
    data["H1FV13"] = pd.to_numeric(data["H1FV13"], errors="coerce")
    sub=data[data["H1FV13"].isin(range(0,366))]
    sub2=sub[~(sub["H1GH60"].isin([996, 998, 999]))]
    data_clean=sub2.dropna()
    x=data_clean["H1GH60"]
    y=data_clean["H1FV13"]
    
    pc1=scipy.stats.pearsonr(data_clean["H1GH60"], data_clean["H1FV13"])
    #pc1=scipy.stats.pearsonr(x, y)    
    print("(co-eff: %f, p-value:%f)" %(pc1[0], pc1[1]))
    scat1 = seaborn.regplot(x="H1GH60", y="H1FV13", fit_reg=False, data=data_clean)
    plt.xlabel("Weight in pounds")
    plt.ylabel("Number of times involved in a serious fight in past year")
    plt.show()
    
    
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
    
    print("""\nPearson correlation between-
    Importance of Religion (H1RE4) and
    Being Upset by difficult problems (H1PF15):\n""")
    have_religion["H1RE4"]=have_religion["H1RE4"].astype("category")
    #For Pearson co-eff only
    have_religion["H1RE4"]=have_religion["H1RE4"].map({1:4,2:3,3:2,4:1})
    have_religion["H1PF15"] = pd.to_numeric(have_religion["H1PF15"], errors="coerce")
    have_religion["H1RE4"] = pd.to_numeric(have_religion["H1RE4"], errors="coerce")
    data_clean=have_religion.dropna()
    pc2=scipy.stats.pearsonr(data_clean["H1RE4"], data_clean["H1PF15"])
    print("(co-eff: %f, p-value:%f)" %(pc2[0], pc2[1]))

    print("""\nPearson correlation between-
    Importance of Religion (H1RE4) and
    Self-esteem (SELF_ESTEEM):\n""")
    have_religion2["H1RE4"]=have_religion2["H1RE4"].astype("category")
    #For Pearson co-eff only
    have_religion2["H1RE4"]=have_religion2["H1RE4"].map({1:4,2:3,3:2,4:1})       
    have_religion2["SELF_ESTEEM"] = pd.to_numeric(have_religion2["SELF_ESTEEM"], errors="coerce")
    have_religion2["H1RE4"] = pd.to_numeric(have_religion2["H1RE4"], errors="coerce")
    data_clean2=have_religion2.dropna()
    pc3=scipy.stats.pearsonr(data_clean2["H1RE4"], data_clean2["SELF_ESTEEM"])
    print("(co-eff: %f, p-value:%f)" %(pc3[0], pc3[1]))
