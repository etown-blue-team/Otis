import pandas as pd
import numpy as np

class DictionaryBuilder:
	_key_list = {}	# Contains column names as keys and their created dictionaries

	def build(self, df):
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
	
		for x in list(df):
			if df[x].dtype is np.dtype('O'):
				unique = df[x].unique()
				keys = {}
				i=0
				for y in unique:
					keys[y] = i
					i += 1
					self._key_list[x] = keys
		return self._key_list

	def map(self, df, key_list):
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

		for x in key_list:
			df[x] = df[x].map(key_list[x])
		return df