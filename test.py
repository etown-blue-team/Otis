import numpy as np
import pandas as pd
import dictionary_builder as db

df = pd.read_csv("FL_insurance_sample.csv")
print(df.head())
x = db.DictionaryBuilder()
keys = x.build(df)
df = x.map(df,keys)
print(df.head())