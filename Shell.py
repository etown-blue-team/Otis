import numpy as np
import pandas as pd

mdf = pd.DataFrame()	#Master dataframe accessible by all functions

def import_data(file=""):
	'''Asks for a dataset and checks if it needs to build a dictionary or not'''
	import dictionary_builder as db
	global mdf
	if mdf.empty:
		if file == "":
			file = input("Data set: ")
		df = pd.read_csv(file)
		for x in list(df):
			if df[x].dtype is np.dtype('O'):
				b = db.DictionaryBuilder(df)
				b.build()
				mdf = b.map()
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

def view_data(n=0):
	if mdf.empty:
		print("Empty dataframe. Use import to load a data set")
	else:
		if n == 0:
			n = input("Number of rows: ")
		print(mdf.head(int(n)))

def list_commands():
	global cmd_list
	for x in cmd_list:
		print(x)

#Shell starts here
cmd_list = {'import': [import_data],'export': [export_data],'view':[view_data],'help':[list_commands]}	#Dictionary of possible commands and which function to call with those commands
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
