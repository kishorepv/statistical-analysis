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
    def cat_to_bin_upset(row):
        return 1 if row["H1PF15"]>=4 else 0
    have_religion["H1RE4"] = pd.to_numeric(have_religion["H1RE4"], errors="coerce")    
    have_religion["RELIGION_BINARY"]=have_religion.apply(lambda row:cat_to_bin(row), axis=1)
    have_religion["H1PF15"] = pd.to_numeric(have_religion["H1PF15"], errors="coerce")
    have_religionx=have_religion[have_religion["H1PF15"]!=3]
    have_religion=have_religionx.copy()
    have_religion["UPSET_BINARY"]=have_religion.apply(lambda row: cat_to_bin_upset(row), axis=1)
    #print_frequency(have_religion, "UPSET_BINARY")
    #print_frequency(have_religion, "H1PF15")

    # logistic regression
    print ("Logistic model for the association between Importance of religion and Getting Upset by difficulties")
    lreg1 = smf.logit(formula = 'UPSET_BINARY ~ RELIGION_BINARY', data = have_religion).fit()
    print(lreg1.summary())
    # odds ratios
    print("Odds Ratios:")
    print(np.exp(lreg1.params))
    #other variables
    #H1DA10- no of hours of video games every week
    #H1GH1- health
    new_cols=["H1DA10", "H1GH1"]
    all_cols=new_cols+["RELIGION_BINARY","UPSET_BINARY"]
    subset_multi= have_religion[all_cols]
    missings=[[996,998], [6,8]]
    for index,col in enumerate(new_cols):
        subset_multi[col]=pd.to_numeric(subset_multi[col], errors="coerce")
        subset_multi[col]=subset_multi[col].replace(missings[index], np.nan)    
    subset_multi=subset_multi.dropna()
    def to_two_health(row):
        return 1 if row["H1GH1"]<=3 else 0
    def to_two_vgames(row):
        return 1 if row["H1DA10"]!=0 else 0    
    #remapping health to 0/1
    subset_multi["HEALTH_BINARY"]=subset_multi.apply(lambda row:to_two_health(row), axis=1)
    subset_multi["VGAMES_BINARY"]=subset_multi.apply(lambda row:to_two_vgames(row), axis=1)
    reg_cols=["RELIGION_BINARY", "VGAMES_BINARY"]+["HEALTH_BINARY", "UPSET_BINARY"]
    
    #for col in reg_cols:
    #    print_frequency(subset_multi, col)
        
    print ("Logistic Regresssion model 1:")
    lreg12 = smf.logit(formula = "UPSET_BINARY ~ RELIGION_BINARY + VGAMES_BINARY + HEALTH_BINARY", data = subset_multi).fit()
    print(lreg12.summary())   
    # odds ratios
    print("Odds Ratios:")
    print(np.exp(lreg12.params))
    # odd ratios with 95% confidence intervals
    params = lreg12.params
    conf = lreg12.conf_int()
    conf['OR'] = params
    conf.columns = ['Lower CI', 'Upper CI', 'OR']
    print (np.exp(conf))


    #Analysys 2
    have_religion2["H1RE4"] = pd.to_numeric(have_religion2["H1RE4"], errors="coerce")    
    have_religion2["RELIGION_BINARY"]=have_religion2.apply(lambda row:cat_to_bin(row), axis=1)
    have_religion2["SELF_ESTEEM"] = pd.to_numeric(have_religion2["SELF_ESTEEM"], errors="coerce")
    def cat_to_bin_esteem(row):
        return 1 if row["SELF_ESTEEM"]>=4 else 0
    have_religionx=have_religion2[have_religion2["SELF_ESTEEM"]!=3]
    have_religion2=have_religionx.copy()
    have_religion2["SELF_ESTEEM_BINARY"]=have_religion2.apply(lambda row: cat_to_bin_esteem(row), axis=1)
    #print_frequency(have_religion, "UPSET_BINARY")
    #print_frequency(have_religion, "H1PF15")    
    
    
    
    print ("Logistic model for the association between Importance of religion and Self-esteem")
    lreg2 = smf.logit(formula = "SELF_ESTEEM_BINARY ~ RELIGION_BINARY", data=have_religion2).fit()
    print(lreg2.summary())
    # odds ratios
    print("Odds Ratios:")
    print(np.exp(lreg2.params))
    # odd ratios with 95% confidence intervals
    params = lreg2.params
    conf = lreg2.conf_int()
    conf['OR'] = params
    conf.columns = ['Lower CI', 'Upper CI', 'OR']
    print (np.exp(conf))


    #other variables
    #H1PL1 - physical limitation
    #H1PF5 - satisified with relation with mother
    #H1GH28 - perception of weight 
    #tobinaries - H1PF5 12,45
    #           H1GH28 3, 1245
    
    
    new_cols=["H1PL1", "H1PF5", "H1GH28"]
    all_cols=new_cols+["RELIGION_BINARY","SELF_ESTEEM_BINARY"]
    subset_multi= have_religion2[all_cols]
    missings=[[6,8], [3,6,7,8], [6, 8]]
    for index,col in enumerate(new_cols):
        subset_multi[col]=pd.to_numeric(subset_multi[col], errors="coerce")
        subset_multi[col]=subset_multi[col].replace(missings[index], np.nan)    
    subset_multi=subset_multi.dropna()
    def to_two_relation_mother(row):
        return 1 if row["H1PF5"]<=2 else 0
    def to_two_wt_percept(row):
        return 1 if row["H1GH28"]==3 else 0
    #remappings
    subset_multi["RELATION_BINARY"]=subset_multi.apply(lambda row:to_two_relation_mother(row), axis=1)
    subset_multi["WT_BINARY"]=subset_multi.apply(lambda row:to_two_wt_percept(row), axis=1)    
    subset_multi["LIMIT_BINARY"]=subset_multi["H1PL1"]
    reg_cols=["RELIGION_BINARY", "SELF_ESTEEM"]+["RELATION_BINARY", "WT_BINARY", "LIMIT_BINARY"]
    
    #for col in reg_cols:
    #   print_frequency(subset_multi, col)
    print("Test for confounding effects of WT_BINARY, RELATION_BINARY or LIMIT_BINARY:\n" )    
    for var in ["WT_BINARY", "RELATION_BINARY", "LIMIT_BINARY"]:
        lreg=smf.logit(formula = "SELF_ESTEEM_BINARY ~ RELIGION_BINARY + %s" %(var), data=subset_multi).fit()
        print(lreg.summary())

    
    
    print ("Logistic regression model 2:")
    lreg22=smf.logit(formula = "SELF_ESTEEM_BINARY ~ RELIGION_BINARY + WT_BINARY + RELATION_BINARY + LIMIT_BINARY", data=subset_multi).fit()
    print(lreg22.summary())
    # odds ratios
    print("Odds Ratios:")
    print(np.exp(lreg22.params))
    # odd ratios with 95% confidence intervals
    params = lreg22.params
    conf = lreg22.conf_int()
    conf['OR'] = params
    conf.columns = ['Lower CI', 'Upper CI', 'OR']
    print (np.exp(conf))
    
    
 