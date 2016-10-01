# -*- coding: utf-8 -*-
#python 3.5
import pandas as pd

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
    subset=data[~(data["H1RE4"].isin([6,7,8]))]
    cols=["H1RE4", "H1PF15", "H1PF33"]
    verbose=["\nVariable %d (%s): Importance of Religion",
             "\nVariable %d (%s): Respondents getting upset by difficult problems",
             "\nVariable %d (%s): Like the way you are"]
    print("\nFrequency distribution of for %s dataset:" %dataset)
    for index,col in enumerate(cols):
        print(verbose[index] %(index+1, col))
        print_frequency(subset, col, "count")
        print_frequency(subset, col, "percent")
