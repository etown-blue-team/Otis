import numpy as np
import pandas as pd
import DictionaryBuilder as db
from Network import *
from sklearn.model_selection import train_test_split

mdf = pd.DataFrame()	#Master dataframe accessible by all functions
outdata = pd.DataFrame()
indata = pd.DataFrame()

inRunData = pd.DataFrame()
outRunData = pd.DataFrame()

neural = Network
class Shell:
	def __init__(self,id):
		self.id = id


	def import_data(self,file=""):
		'''Asks for a dataset and checks if it needs to build a dictionary or not'''
		global mdf
		if mdf.empty:
			if file == "":
				file = input("Data set: ")
			mdf = pd.read_csv(file)
			for x in list(mdf):
				if mdf[x].dtype is np.dtype('O'):
					b = db.DictionaryBuilder(mdf)
					b.build()
					mdf = b.map()
					break
		else:
			ch = input("Data set already loaded. Overwrite? (y/n) ")
			if ch.lower() == 'y':
				mdf = pd.DataFrame()
				self.import_data(file)

	def export_data(self,file=""):
		'''Asks for a file name and writes out the modified dataframe'''
		global mdf
		if mdf.empty:
			print("Empty dataframe. Use import to load a data set")
		else:
			if file == "":
				file = input("File name: ")
			mdf.to_csv(str(file))

	def view_data(self,n=0):
		if mdf.empty:
			print("Empty dataframe. Use import to load a data set")
			return 0
		else:
			if n == 0:
				print(mdf)
				return 1
			else:
				print(mdf.head(int(n)))
				return 1

	def list_commands(self):
		global cmd_list
		print('''
		import [source]      : Imports data from source (URL or File)
		export [destination] : Exports data to a CSV
		clear                : Clears current data
		view [rows]          : View [rows] rows
		train                : Trains network based on imported data
		run  [data]          : Get output from network
		help                 : Show this message
		exit                 : Exit Otis
		''')

	def train(self,col = -1):
		if mdf.empty:
			print("Empty dataframe. Data is needed to train. Use import to import data")
		else:
			if col == -1:
				col = int(input("Which column is output data "))
			
			col = int(col)
			outdata = mdf.iloc[:,col-1:col]
			indata = mdf.drop(mdf.columns[col-1],axis=1)
			
			inTrain, inTest, outTrain, outTest = train_test_split(indata, outdata, test_size=0.3, random_state=101)

			#print(inTrain)
			#print(inTest)
			#print(outTrain)
			#print(outTest)

			neural = Network(inTrain.values,outTrain.values)
			Network.train(neural, 1000)
			Network.run(neural, inTest)
		
	def clear(self):
		#TODO: Clear Network Too
		mdf.drop(mdf.index, inplace=True)
		indata.drop(indata.index, inplace=True)
		outdata.drop(outdata.index,inplace=True)
		print("Dataframe Cleared")
		

	#Shell starts here
	def interact(self):
		cmd_list = {'import': [self.import_data],'export': [self.export_data],'view':[self.view_data],'help':[self.list_commands],'train':[self.train],'clear':[self.clear]}	#Dictionary of possible commands and which function to call with those commands
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

if __name__ == "__main__":
	s = Shell(1)
	Shell.interact(s)