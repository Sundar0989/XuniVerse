import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.stats.stats as stats
import pandas.core.algorithms as algos
#from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from ..transformer import MonotonicBinning

pd.options.mode.chained_assignment = None 

class WOE(BaseEstimator, TransformerMixin):
    
    """Weight of evidence transformation for categorical variables. For numeric variables, 
    monotonic operation is provided as default with this package. 
    
    Parameters
    ----------
    feature_names: 'all' or list (default='all')
        list of features to perform WOE transformation. 
        - 'all' (default): All categorical features in the dataset will be used
        - list of features: ['age', 'income',......]
    
    exclude_features: list (default=None)
        list of features to be excluded from WOE transformation.
        - Example - ['age', 'income', .......]
        
    woe_prefix: string (default=None)
        Variable prefix to be used for the column created by WOE transformer. The default value is set 'None'.  
        
    treat_missing: {'separate', 'mode', 'least_frequent'} (default='separate')
        This parameter setting is used to handle missing values in the dataset.
        'separate' - Missing values are treated as a own group (category)
        'mode' - Missing values are combined with the highest frequent item in the dataset
        'least_frequent' - Missing values are combined with the least frequent item in the dataset
    
    woe_bins: dict of dicts(default=None)
        This feature is added as part of future WOE transformations or scoring. If this value is set, 
        then WOE values provided for each of the features here will be used for transformation. 
        Applicable only in the transform method. 
        Dictionary structure - {'feature_name': float list}
        Example - {'education': {'primary' : 0.1, 'tertiary' : 0.5, 'secondary', 0.7}}
    
    monotonic_binning: bool (default=True)
        This parameter is used to perform monotonic binning on numeric variables. If set to False, 
        numeric variables would be ignored.
    
    mono_feature_names: 'all' or list (default='all')
        list of features to perform monotonic binning operation. 
        - 'all' (default): All features in the dataset will be used
        - list of features: ['age', 'income',......]
    
    mono_max_bins: int (default=20)
        Maximum number of bins that can be created for any given variable. The final number of bins 
        created will be less than or equal to this number.
        
    mono_force_bins: int (default=3)
        It forces the module to create bins for a variable, when it cannot find monotonic relationship 
        using "max_bins" option. The final number of bins created will be equal to the number specified.
        
    mono_cardinality_cutoff: int (default=5)
        Cutoff to determine if a variable is eligible for monotonic binning operation. Any variable 
        which has unique levels less than this number will be treated as character variables. 
        At this point no binning operation will be performed on the variable and it will return the 
        unique levels as bins for these variable.
    
    mono_prefix: string (default=None)
        Variable prefix to be used for the column created by monotonic binning. 
        
    mono_custom_binning: dict (default=None)
        Using this parameter, the user can perform custom binning on variables. This parameter is also 
        used to apply previously computed bins for each feature (Score new data).
        Dictionary structure - {'feature_name': float list}
        Example - {'age': [0., 1., 2., 3.]}
        
    """
    
    # Initialize the parameters for the function
    def __init__(self, feature_names='all', exclude_features=None, woe_prefix=None, 
                 treat_missing='separate', woe_bins=None, monotonic_binning=True, 
                 mono_feature_names='all', mono_max_bins=20, mono_force_bins=3, 
                 mono_cardinality_cutoff=5, mono_prefix=None, mono_custom_binning=None):
        
        self.feature_names = feature_names
        self.exclude_features = exclude_features
        self.woe_prefix = woe_prefix
        self.treat_missing = treat_missing
        self.woe_bins = woe_bins #only used for future transformations
        
        #these features below are for monotonic operations on numeric variables.
        #It uses MonotonicBinning class from binning package.
        self.monotonic_binning = monotonic_binning
        self.mono_feature_names = mono_feature_names
        self.mono_max_bins = mono_max_bins
        self.mono_force_bins = mono_force_bins
        self.mono_cardinality_cutoff = mono_cardinality_cutoff
        self.mono_prefix = mono_prefix
        self.mono_custom_binning = mono_custom_binning #only used for monotonic transformations
    
    # check input data type - Only Pandas Dataframe allowed
    def check_datatype(self, X):
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError("The input data must be pandas dataframe. But the input provided is " + str(type(X)))
        return self
        
    # the fit function for WOE transformer
    def fit(self, X, y):
        
        #if the function is used as part of pipeline, then try to unpack tuple values 
        #produced in the previous step. Added as a part of pipeline feature. 
        try:
            X, y = X
        except:
            pass
        
        #check datatype of X
        self.check_datatype(X)
        
        #The length of X and Y should be equal
        if X.shape[0] != y.shape[0]:
            raise ValueError("Mismatch in input lengths. Length of X is " + str(X.shape[0]) + " \
                            but length of y is " + str(y.shape[0]) + ".")
        
        # The label must be binary with values {0,1}
        unique = np.unique(y)
        if len(unique) != 2:
            raise ValueError("The target column y must be binary. But the target contains " + str(len(unique)) + \
                             " unique value(s).")

        #apply monotonic binning operation
        if self.monotonic_binning:
            self.mono_bin_clf = MonotonicBinning(feature_names=self.mono_feature_names, 
                                            max_bins=self.mono_max_bins, force_bins=self.mono_force_bins,
                                            cardinality_cutoff=self.mono_cardinality_cutoff, 
                                            prefix=self.mono_prefix, custom_binning=self.mono_custom_binning)
            X = self.mono_bin_clf.fit_transform(X, y)
            self.mono_custom_binning = self.mono_bin_clf.bins
        
        #identify the variables to tranform and assign the bin mapping dictionary
        self.woe_bins = {} #bin mapping
        
        if not self.mono_custom_binning:
            self.mono_custom_binning= {}
        else:
            for i in self.mono_custom_binning:
                X[i] = X[i].astype('object')
            
        numerical_features = list(X._get_numeric_data().columns)
        categorical_features = list(X.columns.difference(numerical_features))
        
        #Identifying the features to perform fit
        if self.feature_names == 'all':
            self.transform_features = categorical_features
        else:
            self.transform_features = list(set(self.feature_names))
        
        #Exclude variables provided in the exclusion list
        if self.exclude_features:
            self.transform_features = list(set(self.transform_features) - set(self.exclude_features))
        
        temp_X = X[self.transform_features] #subset data only on features to fit
        temp_X = temp_X.astype('object') #convert categorical columns to object columns
        temp_X = self.treat_missing_values(temp_X) #treat missing values function
       
        #apply the WOE train function on dataset
        temp_X.apply(lambda x: self.train(x, y), axis=0)
        
        #provide Information value for each variable as a separate dataset
        self.iv_df = pd.DataFrame({'Information_Value':self.woe_df.groupby('Variable_Name').Information_Value.max()})
        self.iv_df = self.iv_df.reset_index()
        self.iv_df = self.iv_df.sort_values('Information_Value', ascending=False)
        
        return self
    
    #treat missing values based on the 'treat_missing' option provided by user
    def treat_missing_values(self, X):
        """
        treat_missing: {'separate', 'mode', 'least_frequent'} (default='separate')
        This parameter setting is used to handle missing values in the dataset.
        'separate' - Missing values are treated as a own group (category)
        'mode' - Missing values are combined with the highest frequent item in the dataset
        'least_frequent' - Missing values are combined with the least frequent item in the dataset
        """
        
        if self.treat_missing == 'separate':
            X = X.fillna('NA')
        elif self.treat_missing == 'mode':
            X = X.fillna(X.mode().iloc[0])
        elif self.treat_missing == 'least_frequent':
            for i in X:
                X[i] = X[i].fillna(X[i].value_counts().index[-1])
        else:
            raise ValueError("Missing values could be treated with one of these three options - \
                            'separate', 'mode', 'least_frequent'. \
                            The provided option is - " + str(self.treat_missing))
        
        return X
    
    #WOE binning - The function is applied on each columns identified in the fit function. 
    #Here, the input X is a Pandas Series type.
    def train(self, X, y):
        
        # Assign values
        woe_mapping = {} #dictionary mapping for the current feature
        temp_woe = pd.DataFrame({},index=[])
        temp_df = pd.DataFrame({'X': X, "Y":y})        
        grouped_df = temp_df.groupby('X', as_index=True)

        #calculate stats for variable and store it in temp_woe
        target_sum = grouped_df.Y.sum()
        temp_woe['Count'] = grouped_df.Y.count()
        temp_woe['Category'] = target_sum.index
        temp_woe['Event'] = target_sum
        temp_woe['Non_Event'] = temp_woe['Count'] - temp_woe['Event']
        temp_woe['Event_Rate'] = temp_woe['Event']/temp_woe['Count']
        temp_woe['Non_Event_Rate'] = temp_woe['Non_Event']/temp_woe['Count']
        
        #calculate distributions and woe
        total_event = temp_woe['Event'].sum()
        total_non_event = temp_woe['Non_Event'].sum()
        temp_woe['Event_Distribution'] = temp_woe['Event']/total_event
        temp_woe['Non_Event_Distribution'] = temp_woe['Non_Event']/total_non_event
        temp_woe['WOE'] = np.log(temp_woe['Event_Distribution']/temp_woe['Non_Event_Distribution'])
        temp_woe['Information_Value'] = (temp_woe['Event_Distribution']- \
                                         temp_woe['Non_Event_Distribution'])*temp_woe['WOE']
        temp_woe['Variable_Name'] = X.name
        temp_woe = temp_woe[['Variable_Name', 'Category', 'Count', 'Event', 'Non_Event', \
                             'Event_Rate', 'Non_Event_Rate', 'Event_Distribution', 'Non_Event_Distribution', \
                             'WOE', 'Information_Value']]
        temp_woe = temp_woe.replace([np.inf, -np.inf], 0)
        temp_woe['Information_Value'] = temp_woe['Information_Value'].sum()
        temp_woe = temp_woe.reset_index(drop=True)
        woe_mapping[str(X.name)] = dict(zip(temp_woe['Category'], temp_woe['WOE']))
        
        #assign computed values to class variables
        try:
            self.woe_df = self.woe_df.append(temp_woe, ignore_index=True)
            self.woe_bins.update(woe_mapping)
        except:
            self.woe_df = temp_woe
            self.woe_bins = woe_mapping
       
        return self
        
    #Transform new data or existing data based on the fit identified or custom transformation provided by user
    def transform(self, X, y=None):
        
        #if the function is used as part of pipeline, then try to unpack tuple values 
        #produced in the previous step. Added as a part of pipeline feature. 
        try:
            X, y = X
        except:
            pass
        
        self.check_datatype(X) #check input datatype. 
        outX = X.copy(deep=True) 
        
        #identify the features on which the transformation should be performed
        try:
            if self.transform_features:
                transform_features = self.transform_features
        except:
            if self.woe_bins:
                transform_features = list(self.woe_bins.keys())
            else:
                raise ValueError("Estimator has to be fitted to make WOE transformations")
        
        #final list of features to be transformed
        transform_features = list(set(transform_features) & set(outX.columns)) 
        
        #raise error if the list is empty
        if not transform_features:
            raise ValueError("Empty list for WOE transformation. \
                            Estimator has to be fitted to make WOE transformations")
        
        #use the custom bins provided by user for numeric variables
        if self.mono_custom_binning:
            try:
                if self.mono_bin_clf:
                    pass
            except:    
                self.mono_bin_clf = MonotonicBinning(feature_names=self.mono_feature_names, 
                                            max_bins=self.mono_max_bins, force_bins=self.mono_force_bins,
                                            cardinality_cutoff=self.mono_cardinality_cutoff, 
                                            prefix=self.mono_prefix, custom_binning=self.mono_custom_binning)

            outX = self.mono_bin_clf.transform(outX)
        
        outX = outX.astype('object') #convert categorical columns to object columns
        outX = self.treat_missing_values(outX) #treat missing values function
        #iterate through the dataframe and apply the bins
        for i in transform_features:
            
            tempX = outX[i] #pandas Series
            original_column_name = str(i)
            
            #create the column name based on user provided prefix
            if self.woe_prefix:
                new_column_name = str(self.woe_prefix) + '_' + str(i)
            else:
                new_column_name = original_column_name
            
            #check if the bin mapping is present 
            #check_is_fitted(self, 'woe_bins')
            if not self.woe_bins:
                raise ValueError("woe_bins variable is not present. \
                                Estimator has to be fitted to apply transformations.")
            
            outX[new_column_name] = tempX.replace(self.woe_bins[original_column_name])
            
        #transformed dataframe 
        return outX
    
    #Method that describes what we need this transformer to do
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
