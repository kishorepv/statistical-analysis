# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy.stats
import seaborn
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
    #convert the optimism and like-as-i-am variable into ratings which is more intuitive
    for part in data_parts:
        for col in cols[1:]:
            part[col]=part[col].map(cat_to_rating)
            
    have_religion["H1RE4"]=have_religion["H1RE4"].astype("category")        
    have_religion["H1RE4"]=have_religion["H1RE4"].cat.rename_categories(["Highly Imp", "Fairly Imp", " Fairly Unimp", "Highly unimp"])
    have_religion["H1PF15"] = pd.to_numeric(have_religion["H1PF15"], errors="coerce")
    def to_two(row):
        return 1 if row["H1PF15"]>=3 else 0
    
    have_religion_strict = have_religion[have_religion["H1PF15"]!=3]
    have_religion=have_religion_strict.copy()    
    have_religion["UPSET_BINARY"]=have_religion.apply(lambda row:to_two(row), axis=1)
    have_religion["UPSET_BINARY"] = pd.to_numeric(have_religion["UPSET_BINARY"], errors="coerce")    
    print("\nContingency table of observed count:\n")
    ct1=pd.crosstab(have_religion["UPSET_BINARY"], have_religion["H1RE4"])
    ##print(ct1)
    
    ##print("\nColumn Percentages:\n")
    colsum=ct1.sum(axis=0)
    colpct=ct1/colsum
    print(colpct)
    

    #chi-square
    ##print('\nChi-square value, p value, Expected counts')
    cs1=scipy.stats.chi2_contingency(ct1)
    ##print(cs1)
    
    # set variable types 
    have_religion["H1RE4"] = have_religion["H1RE4"].astype("category")
    # graph percent of respondents upset by dificult within each relgion importance group 
    ##seaborn.factorplot(x="H1RE4", y="UPSET_BINARY", data=have_religion, kind="bar", ci=None)
    ##plt.xlabel('Importance of Religion')
    ##plt.ylabel('Proportion of people getting upset by difficult problems')
    ##plt.show()
    
    
    def self_esteem_to_two(row):
        return 1 if row["SELF_ESTEEM"]>=3 else 0

    have_religion2["H1RE4"]=have_religion2["H1RE4"].astype("category")
    #have_religion2["H1RE4"]=have_religion2["H1RE4"].cat.rename_categories(["Highly Imp", "Fairly Imp", " Fairly Unimp", "Highly unimp"])

    have_religion2["SELF_ESTEEM"] = pd.to_numeric(have_religion2["SELF_ESTEEM"], errors="coerce")
    have_religion2_strict=have_religion2[have_religion2["SELF_ESTEEM"]!=3]
    #print_frequency(have_religion2_strict, "SELF_ESTEEM", format="count")
    #input()
    have_religion2=have_religion2_strict.copy()
    have_religion2["SELF_ESTEEM_BINARY"]=have_religion2.apply(lambda row:self_esteem_to_two(row), axis=1)    
    
    print("\nContingency table of observed count:\n")
    ct2=pd.crosstab(have_religion2["SELF_ESTEEM_BINARY"], have_religion2["H1RE4"])
    print(ct2)
    
    print("\nColumn Percentages:\n")
    colsum2=ct2.sum(axis=0)
    colpct2=ct2/colsum2
    print(colpct2)
    

    #chi-square
    print('\nChi-square value, p value, Expected counts')
    cs2=scipy.stats.chi2_contingency(ct2)
    print(cs2)
    
    seaborn.factorplot(x="H1RE4", y="SELF_ESTEEM_BINARY", data=have_religion2, kind="bar", ci=None)
    plt.xlabel("Importance of Religion")
    plt.ylabel("Proportion of respondents having positive self-esteem")
    plt.show()
    moderator="H1GH1"
    moderator_mod="HEALTHY"
    have_religion2[moderator]=pd.to_numeric(have_religion2[moderator], errors="coerce")
    sub=have_religion2[~(have_religion2[moderator].isin([6,8]))]
    sub[moderator]=sub[moderator].map(cat_to_rating)
    #print_frequency(have_religion2, moderator)
    #print_frequency(sub, moderator)
    def to_two_health(row):
        return 1 if row[moderator]>=3 else 0
    #HERE WE DONT FILTER RESPONSE 3 AS IT IS NOT NEUTRAL
    sub[moderator_mod]=sub.apply(lambda row:to_two_health(row), axis=1)
    #print_frequency(sub, moderator_mod)
    part_yes=sub[sub[moderator_mod]==1]
    part_no=sub[sub[moderator_mod]==0]
    whole=[part_yes, part_no]
    var=["Healthy", "Unhealthy"]
    for index,part in enumerate(whole):
        print("\nChi-square test for %s:\n" %(var[index]))
        ct2=pd.crosstab(part["SELF_ESTEEM_BINARY"], part["H1RE4"])
        print(ct2)
    
        print("\nColumn Percentages:\n")
        colsum2=ct2.sum(axis=0)
        colpct2=ct2/colsum2
        print(colpct2)
    

        #chi-square
        print('\nChi-square value, p value, Expected counts')
        cs2=scipy.stats.chi2_contingency(ct2)
        print(cs2)
    
        seaborn.factorplot(x="H1RE4", y="SELF_ESTEEM_BINARY", data=part, kind="bar", ci=None)
        plt.xlabel("Importance of Religion for %s" %(var[index]))
        plt.ylabel("Proportion of respondents having positive self-esteem for %s" %(var[index]))
        plt.show()
