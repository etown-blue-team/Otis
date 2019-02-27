class DictionaryBuilder:
    _key_list = {}
    
    def build(df):
        for x in list(df):
            if df[x].dtype is np.dtype('O'):
                unique = df[x].unique()
                keys = {}
                i=0
                for y in unique:
                    keys[y] = i
                    i += 1
                _key_list[x] = keys
                
    def apply(df):
        pass
    
    def get():
        return _key_list