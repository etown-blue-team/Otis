import numpy as np
import pandas as pd
import dictionary_builder as db

df = pd.read_csv("FL.csv")
x = db.DictionaryBuilder()
keys = x.build(df)
df = x.map(df,keys)
print(str(df.head()))