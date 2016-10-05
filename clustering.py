# -*- coding: utf-8 -*-
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi
 
if __name__=="__main__":
    """
    Features:
    H1DA10 hours of video game play in a week
    H1DA11 hours of listen to radio per week
    H1DA9 hours of video watch in a week
    H1DA8 hours of tv watch in a week
    H1GH51 hours of sleep
    H1GH60 weight
    H1SE4 perception of intelligence
    H1EE1 want to go to college
    H1ED16 pay attention in school
    H1ED15 get along with teachers
    H1ED17 get homework done
    H1ED18 getting along with other students
    H1ED3 skipped grade
    H1PF35 feel socially accepted
    Grades:
    H1ED11 english grade
    H1ED12 mathematics grade
    H1ED13 history/social
    H1ED14 science grade
    
    """
    dataset="ADDHEALTH"
    data=pd.read_csv("addhealth_pds.csv", low_memory=False)
    
    d={"H1DA10":[996, 998], "H1DA11":[996, 998], "H1DA9":[996,998] , "H1DA8":[996,998],
    "H1ED16":[6,7,8], "H1ED15":[6,7,8], "H1ED17":[6,7,8], "H1ED18":[6,7,8],
    "H1ED3":[6,8], "H1PF35":[6,8], "H1GH51":[96, 98],"H1GH60":[996,998,999], "H1SE4":[96,98],"H1EE1":[6,8], 
    "H1ED11":[5,6,96,97,98],  "H1ED12":[5,6,96,97,98], "H1ED13":[5,6,96,97,98], "H1ED14":[5,6,96,97,98]}            
    variables=list(d.keys())
    subset=data[variables]
    for col in variables:
        subset[col]=pd.to_numeric(subset[col], errors="coerce")
        subset[col]=subset[col].replace(d[col], np.nan)
    subset=subset.dropna()
    #grade (secondary variable)
    subset["GRADE_REV"]=(subset["H1ED11"]+subset["H1ED12"]+subset["H1ED13"]+subset["H1ED14"])/4.0
    def grade_to_score(row):
        return (4.0-row["GRADE_REV"])
    subset["GRADE"]=subset.apply(lambda row:grade_to_score(row), axis=1)
    print("\nDataset size:", len(subset))
    feat_names=["H1DA10","H1DA11","H1DA9","H1DA8","H1GH51","H1GH60","H1SE4","H1EE1","H1ED16","H1ED15","H1ED17","H1ED18","H1ED3","H1PF35"]
    features=subset[feat_names]
    #standardize features with mean=0 and sd=1
    feats=[]
    target=None
    for col in feat_names:
        features[col]=preprocessing.scale(features[col].astype('float64'))
    clus_train, clus_test = train_test_split(features, test_size=.3, random_state=12)
    clusters=range(1,10)
    meandist=[]
    for k in clusters:
        model=KMeans(n_clusters=k)
        model.fit(clus_train)
        clusassign=model.predict(clus_train)
        meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1)) 
    / clus_train.shape[0])
    
    
    
    #Plot average distance from observations from the cluster centroid 
    #to use the Elbow Method to identify number of clusters to choose
    plt.plot(clusters, meandist)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average distance')
    plt.title('Selecting k with the Elbow Method')
    plt.show()
    
    for c in [2,7]:
        print("\nInterpreting %d cluster solution:\n" %(c))
        model2=KMeans(n_clusters=c)
        model2.fit(clus_train)
        clusassign=model2.predict(clus_train)
        
        print("\nPlot %d clusters:" %(c))
        pca_2 = PCA(2)
        plot_columns = pca_2.fit_transform(clus_train)
        plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model2.labels_,)
        plt.xlabel("Canonical variable 1")
        plt.ylabel("Canonical variable 2")
        plt.title("Scatterplot of Canonical Variables for %d Clusters" %(c))
        plt.show()
        
        #merge cluster assignment with clustering variables
        clus_train.reset_index(level=0, inplace=True)
        cluslist=list(clus_train["index"])
        labels=list(model2.labels_)
        newlist=dict(zip(cluslist, labels))
        newclus=DataFrame.from_dict(newlist, orient="index")
        newclus.columns = ["cluster"]
        newclus.reset_index(level=0, inplace=True)
        merged_train=pd.merge(clus_train, newclus, on="index")
    
        print("\nCluster frequencies:")
        print(merged_train.cluster.value_counts())
        clustergrp = merged_train.groupby("cluster").mean()
        print ("\nClustering variable means by cluster:")
        print(clustergrp)
    
    
        print("\nValidation of clusters in training data by examining cluster differences in grades using ANOVA:") 
        gpa_data=subset["GRADE"]
        gpa_train, gpa_test = train_test_split(gpa_data, test_size=.3, random_state=12)
        gpa_train1=pd.DataFrame(gpa_train)
        gpa_train1.reset_index(level=0, inplace=True)
        merged_train_all=pd.merge(gpa_train1, merged_train, on="index")
        sub1 = merged_train_all[[ "GRADE", "cluster"]].dropna()
        gpamod = smf.ols(formula='GRADE ~ C(cluster)', data=sub1).fit()
        print (gpamod.summary())
        m1= sub1.groupby('cluster').mean()
        print("Means for Grade by cluster:")   
        print(m1)
        m2= sub1.groupby("cluster").std()
        print ("Standard deviations for Grade by cluster:")
        print(m2)
        """
        mc1 = multi.MultiComparison(sub1["GRADE"], sub1['cluster'])
        res1 = mc1.tukeyhsd()
        print(res1.summary())
        
        """