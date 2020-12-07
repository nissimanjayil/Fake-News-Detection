import os, glob
import pandas as pd


'''Merge Separate csvs to One'''
path = "C:/Users/Owner/Fake-News-Detection/Dataset"

all_files = glob.glob(os.path.join(path, "*.csv"))
df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
df_merged   = pd.concat(df_from_each_file, ignore_index=True)
df_merged.to_csv( "dataset.csv")

'''Export Random Sample from Data'''
n = 400
df = pd.read_csv("dataset.csv")
sample = df.sample(n)
sample.to_csv("sample.csv")