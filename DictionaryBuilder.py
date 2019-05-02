import pandas as pd
import numpy as np

class DictionaryBuilder:
	def __init__(self, data):
		self.key_list = {}	# Contains column names as keys and their created dictionaries
		self.df = data

	def build(self):
		'''
		Builds a dictionary of the unique entries in each non-numerical column and an assigned integer
	
		df = c1  c2  c3
             a   b   c
             d   e   f

		>>>build(df)

		_key_list = {c1: {a:1,d:2},
				     c2: {b:1,e:2},
				     c3: {c:1,f:2}}
		'''
	
		for x in list(self.df):
			if self.df[x].dtype is np.dtype('O'):
				unique = self.df[x].unique()
				keys = {}
				i=0
				for y in unique:
					keys[y] = i
					i += 1
					self.key_list[x] = keys

	def map(self):
		'''
		Maps the dictionary to the dataframe, replacing any non-numerical entries with integers
		
		df = c1  c2  c3
		     a   b   c
			 d   e   f
		
		>>>map(df)

		df = c1  c2  c3
		     1   1   1
			 2   2   2
		'''

		for x in self.key_list:
			self.df[x] = self.df[x].map(self.key_list[x])
		return self.df

	def clean(self, na=np.nan):
		'''
		Looks for missing values and drops those rows.
		Paramaters:
			na: The symbol used in the dataset to indicate a missing value

		df = c1  c2  c3
		     a   b   c
			 d   ?   f

		>>>clean('?')

		df = c1  c2  c3
		     a   b   c
		'''

		self.df.replace({na:np.nan}).dropna(inplace=True)
		return self.df
