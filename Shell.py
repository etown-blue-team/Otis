import numpy as np
import pandas as pd

mdf = pd.DataFrame()	#Master dataframe accessible by all functions

def build(df):
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
	
	key_list = {}	# Contains column names as keys and their created dictionaries
	for x in list(df):
		if df[x].dtype is np.dtype('O'):
			unique = df[x].unique()
			keys = {}
			i=0
			for y in unique:
				keys[y] = i
				i += 1
				key_list[x] = keys
	return key_list

def map(df, key_list):
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
	global mdf
	mdf = df
	for x in key_list:
		mdf[x] = df[x].map(key_list[x])

def import_data(file=""):
	'''Asks for a dataset and checks if it needs to build a dictionary or not'''
	global mdf
	if mdf.empty:
		if file == "":
			file = input("Data set: ")
		df = pd.read_csv(file)
		for x in list(df):
			if df[x].dtype is np.dtype('O'):
				keys = build(df)
				df = map(df,keys)
				break
	else:
		ch = input("Data set already loaded. Overwrite? (y/n) ")
		if ch.lower() == 'y':
			mdf = pd.DataFrame()
			import_data(file)

def export_data(file=""):
	'''Asks for a file name and writes out the modified dataframe'''
	global mdf
	if mdf.empty:
		print("Empty dataframe. Use import to load a data set")
	else:
		if file == "":
			file = input("File name: ")
		mdf.to_csv(str(file))

def view_data():
	n = int(input("Number of rows: "))
	print(mdf.head(n))

#Shell starts here
cmd_list = {'import': [import_data],'export': [export_data],'view':[view_data]}	#Dictionary of possible commands and which function to call with those commands
while (True):
	cmd = input('Otis> ')
	if cmd.lower() == 'exit':
		break
	elif cmd.split()[0] in cmd_list:		#If the first argument is a valid command
		if len(cmd.split()) == 1:			#If the user did not supply any arguments
			if len(cmd_list[cmd]) > 1:		#If the function to call takes any arguments
				args = cmd_list[cmd][1:]	
				cmd_list[cmd][0](args[0])
			else:							#If the fuction to call takes no arguments
				cmd_list[cmd][0]()
		else:								#If the user did supply arguments
			args = cmd.split()
			cmd_list[args[0]][0](args[1])	#Can only accept one argument at the moment
	else:
		print("Illegal command")