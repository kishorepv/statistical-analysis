# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
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


    #print(have_religion2[["SELF_ESTEEM"]+self_esteem].head(10))
    cols=["H1RE4", "H1PF15", "H1PF33"]
    #convert the being-upset and like-as-i-am variable into ratings which is more intuitive
    for part in data_parts:
        for col in cols[1:]:
            part[col]=part[col].map(cat_to_rating)

    print("PLOTS:\n")
    print("UNIVARIABLE Plots:\n")
    print_frequency(have_religion, "H1RE4", "count")
    have_religion["H1RE4"]=have_religion["H1RE4"].astype("category")
    have_religion["H1RE4"]=have_religion["H1RE4"].cat.rename_categories(["Highly Imp", "Fairly Imp", " Fairly Unimp", "Highly unimp"])
    seaborn.countplot(x="H1RE4", data=have_religion)
    plt.xlabel('Importance of Religion')
    plt.title('Importance of Religion in life, in ADDHEALTH Study')
    plt.show()
    print("Summary of Importance of Religion variable:\n")
    print(have_religion["H1RE4"].describe())

    seaborn.countplot(x="H1PF15", data=have_religion)
    plt.xlabel('Rating for being upset by difficult problems')
    plt.title('Respondents getting upset by difficult problems, in ADDHEALTH Study')
    plt.show()
    have_religion["H1PF15"]=have_religion["H1PF15"].astype("category")
    print("Summary of Getting upset by difficult problems variable:\n")
    print(have_religion["H1PF15"].describe())

    seaborn.countplot(x="SELF_ESTEEM", data=have_religion2)
    plt.xlabel('Self-esteem rating')
    plt.title('Respondents rating for self-esteem, in ADDHEALTH Study')
    plt.show()
    print("Summary of SELF_ESTEEM Variable:\n")
    have_religion2["SELF_ESTEEM"]=have_religion2["SELF_ESTEEM"].astype("category")
    print(have_religion2["SELF_ESTEEM"].describe())

    print("\nBIVARIABLE Plots:\n")
    

    def to_two(row):
        return 1 if row["H1PF15"]>=3 else 0

    have_religion["H1PF15"] = have_religion["H1PF15"].convert_objects(convert_numeric=True)
    have_religion_strict = have_religion[have_religion["H1PF15"]!=3]
    have_religion_strict["H1PF15"] = have_religion_strict["H1PF15"].convert_objects(convert_numeric=True)
    have_religion_strict["UPSET_BINARY"]=have_religion_strict.apply(lambda row:to_two(row), axis=1)

    have_religion_strict["UPSET_BINARY"] = have_religion_strict["UPSET_BINARY"].convert_objects(convert_numeric=True)
    seaborn.factorplot(x="H1RE4", y="UPSET_BINARY", data=have_religion_strict, kind="bar", ci=None)
    plt.title("Proportion getting upset by difficult problems v/s Importance of Religion in life")
    plt.xlabel("Religion")
    plt.ylabel("Proportion getting upset by difficult problems")
    print("\n PLOT 1\n")
    plt.show()
    print("\n")

    def self_esteem_to_two(row):
        return 1 if row["SELF_ESTEEM"]>=3 else 0

    have_religion2["H1RE4"]=have_religion2["H1RE4"].astype("category")
    have_religion2["H1RE4"]=have_religion2["H1RE4"].cat.rename_categories(["Highly Imp", "Fairly Imp", " Fairly Unimp", "Highly unimp"])
    have_religion2_strict=have_religion2[have_religion2["SELF_ESTEEM"]!=3]
    have_religion2_strict["SELF_ESTEEM"] = have_religion2_strict["SELF_ESTEEM"].convert_objects(convert_numeric=True)
    have_religion2_strict["SELF_ESTEEM_BINARY"]=have_religion2_strict.apply(lambda row:self_esteem_to_two(row), axis=1)
    seaborn.factorplot(x="H1RE4", y="SELF_ESTEEM_BINARY", data=have_religion2_strict, kind="bar", ci=None)
    plt.title("Proportion of positive self-esteem v/s Importance of Religion in life")    
    plt.xlabel("Religion")
    plt.ylabel("Proportion of positive self-esteem")
    print("\n PLOT 2\n")
    plt.show()
   