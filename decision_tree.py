# -*- coding: utf-8 -*-
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics

from sklearn import tree
from io import StringIO
from IPython.display import Image
import pydotplus

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
    #other variables
    #H1DA10- no of hours of video games every week
    #H1GH1- health
    #H1DA5- active sports
    #H1DA6- excercise
    #H1TS13- taught about stress in school
    #H1PL1- physical limitation
    #H1SE4- how intelligent
    new_cols=["H1DA10", "H1GH1","H1GI6A", "H1GI6B", "H1GI6C", "H1GI6D", "H1GI6E", "H1DA5", "H1DA6","H1TS13","H1PL1" ,"H1SE4"]
    all_cols=new_cols+["RELIGION_BINARY","UPSET_BINARY"]
    subset_multi= have_religion[all_cols]

    missings=[[996,998], [6,8], [6,8],[6,8],[6,8],[6,8],[6,8],   [6,8], [6,8], [6,8], [6,8], [96,98]]
    for index,col in enumerate(new_cols):
        subset_multi[col]=pd.to_numeric(subset_multi[col], errors="coerce")
        subset_multi[col]=subset_multi[col].replace(missings[index], np.nan)
    subset_multi2=subset_multi.dropna()
    subset_multi=subset_multi2.copy()
    def to_two_health(row):
        return 1 if row["H1GH1"]<=3 else 0
    #def to_two_vgames(row):
    #    return 1 if row["H1DA10"]!=0 else 0
    #remapping health to 0/1
    subset_multi["HEALTH_BINARY"]=subset_multi.apply(lambda row:to_two_health(row), axis=1)
    #subset_multi["VGAMES_BINARY"]=subset_multi.apply(lambda row:to_two_vgames(row), axis=1)
    reg_cols=["RELIGION_BINARY", "H1DA10"]+["HEALTH_BINARY", "UPSET_BINARY"]
    #classify
    predictors = subset_multi[["H1GH1","RELIGION_BINARY", "H1DA10", "H1GI6A", "H1GI6B", "H1GI6C", "H1GI6D", "H1GI6E", "H1DA5", "H1DA6","H1TS13","H1PL1" ,"H1SE4"]] #is not included to create a simpler tree
    targets = subset_multi.UPSET_BINARY
    pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, targets, test_size=.4)
    print(pred_train.shape, pred_test.shape)
    print(tar_train.shape, tar_test.shape)
    classifier=DecisionTreeClassifier()
    classifier=classifier.fit(pred_train, tar_train)
    predictions=classifier.predict(pred_test)
    cmatrix=sklearn.metrics.confusion_matrix(tar_test,predictions)
    accuracy=sklearn.metrics.accuracy_score(tar_test, predictions)
    print ("Binary Classification Tree 1:")
    print("Confusion Matrix:")
    print(cmatrix)
    print("Accuracy: ", accuracy)


    #display decision tree
    os.chdir("C:\TREES")
    out = StringIO()
    tree.export_graphviz(classifier, out_file=out)
    graph=pydotplus.graph_from_dot_data(out.getvalue())
    img=graph.create_png()
    with open("file1.png", 'wb') as f:
        f.write(img)


    #Analysis 2
    have_religion2["H1RE4"] = pd.to_numeric(have_religion2["H1RE4"], errors="coerce")
    have_religion2["RELIGION_BINARY"]=have_religion2.apply(lambda row:cat_to_bin(row), axis=1)
    have_religion2["SELF_ESTEEM"] = pd.to_numeric(have_religion2["SELF_ESTEEM"], errors="coerce")
    def cat_to_bin_esteem(row):
        return 1 if row["SELF_ESTEEM"]>=4 else 0
    have_religionx=have_religion2[have_religion2["SELF_ESTEEM"]!=3]
    have_religion2=have_religionx.copy()
    have_religion2["SELF_ESTEEM_BINARY"]=have_religion2.apply(lambda row: cat_to_bin_esteem(row), axis=1)
    #other variables
    #H1PL1 - physical limitation
    #H1PF5 - satisified with relation with mother
    #H1GH28 - perception of weight
    new_cols=["H1PL1", "H1PF5", "H1GH28","H1GI6A", "H1GI6B", "H1GI6C", "H1GI6D", "H1GI6E", "H1DA5", "H1DA6","H1TS13","H1PL1" ,"H1SE4"]
    all_cols=new_cols+["RELIGION_BINARY","SELF_ESTEEM_BINARY"]
    subset_multix= have_religion2[all_cols]
    subset_multi=subset_multix.copy()
    missings=[[6,8], [3,6,7,8], [6, 8],[6,8],[6,8],[6,8],[6,8],[6,8],   [6,8], [6,8], [6,8], [6,8], [96,98]]
    for index,col in enumerate(new_cols):
        subset_multi=subset_multi.apply(lambda x:pd.to_numeric(x), axis=0)
        #subset_multi[col]=pd.to_numeric(subset_multi[col], errors="coerce")
        subset_multi[col]=subset_multi[col].replace(missings[index], np.nan)
    subset_multi2=subset_multi.dropna()
    subset_multi=subset_multi2.copy()
    def to_two_relation_mother(row):
        return 1 if row["H1PF5"]<=2 else 0
    def to_two_wt_percept(row):
        return 1 if row["H1GH28"]==3 else 0
    #remappings
    subset_multi["RELATION_BINARY"]=subset_multi.apply(lambda row:to_two_relation_mother(row), axis=1)
    subset_multi["WT_BINARY"]=subset_multi.apply(lambda row:to_two_wt_percept(row), axis=1)
    #classify
    subset_multi=subset_multi.dropna()
    predictors = subset_multi[["H1GH28", "RELIGION_BINARY", "RELATION_BINARY", "H1PL1","H1GI6A", "H1GI6B", "H1GI6C", "H1GI6D", "H1GI6E", "H1DA5", "H1DA6","H1TS13","H1PL1" ,"H1SE4"]]
    targets = subset_multi.SELF_ESTEEM_BINARY
    pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, targets, test_size=.4)

    classifier2=DecisionTreeClassifier()
    classifier2=classifier2.fit(pred_train, tar_train)
    predictions2=classifier2.predict(pred_test)
    cmatrix2=sklearn.metrics.confusion_matrix(tar_test,predictions2)
    accuracy2=sklearn.metrics.accuracy_score(tar_test, predictions2)
    print ("Binary Classification Tree 2:")
    print("Confusion Matrix:")
    print(cmatrix2)
    print("Accuracy: ", accuracy2)

    #display decision tree
    out = StringIO()
    tree.export_graphviz(classifier2, out_file=out)
    graph2=pydotplus.graph_from_dot_data(out.getvalue())
    img2=graph2.create_png()
    with open("file2.png", 'wb') as f:
        f.write(img2)
