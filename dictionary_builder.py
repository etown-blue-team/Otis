class DictionaryBuilder:
    _key_list = {}	# Dictionary containing column names and their created dictionaries
    
    def build(df):	# Builds a dictionary of the unique entries in each non-numerical column and an assigned integer
        for x in list(df):
            if df[x].dtype is np.dtype('O'):
                unique = df[x].unique()
                keys = {}
                i=0
                for y in unique:
                    keys[y] = i
                    i += 1
                _key_list[x] = keys
                
    def apply(df):	# Applies the dictionary to the dataframe, replacing any non-numerical entries with integers
	pass
    
    def get():		# Returns the master dictionary containing the created dictionaries
        return _key_list
