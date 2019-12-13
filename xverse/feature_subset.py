from sklearn.base import BaseEstimator, TransformerMixin

# custom Transformer to split feature dataset
class FeatureSubset(BaseEstimator, TransformerMixin): 
    
    """Select a subset of features from the dataframe.
    Parameters
    ----------
    
    feature_names: list
        a list of features to subset. 
    """
    #Class Constructor
    def __init__(self, feature_names):
        assert isinstance(feature_names, list), "Expects list input. " + str(type(feature_names)) + " provided."
        self.feature_names = feature_names
    
    #return self nothing else to do here
    def fit(self, X, y=None):
        return self    
    
    #Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        return X[self.feature_names]

# custom Transformer to split X and Y
class SplitXY(FeatureSubset):
    
    """Split features and labels.
    Parameters
    ----------
    feature_names: list
        a list with the target column name
    """

    def __init__(self, feature_names):
        assert len(feature_names) == 1, "Works with only one target column. Multiple target column names provided."
        assert isinstance(feature_names, list), "Expects list input. " + str(type(feature_names)) + " provided."
        self.feature_names = feature_names
    
    def transform(self, X, y=None):
        return X[X.columns.difference(self.feature_names)], X[self.feature_names[0]].values