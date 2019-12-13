import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import numpy as np
import math
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from ..transformer import WOE
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from collections import defaultdict
#from sklearn.utils.validation import check_is_fitted 
from functools import reduce

pd.options.mode.chained_assignment = None

class VotingSelector(BaseEstimator, TransformerMixin):
    
    """Select the input features for a binary model prediction using voting technique. Apply multiple 
    feature selection techniques (Linear and Non linear) on the dataset and calculate the vote secured 
    by all input features for a given binary target. The final dataset contains, 
    
    1. Feature selection techniques applied on the dataset
    2. Whether the variable is selected in step 1 for each technique
    3. Final votes
    
    Parameters
    ----------
    
    feature_names: 'all' or list (default='all')
        list of features to perform WOE transformation. 
        'all' (default) - All categorical features in the dataset will be used
        list of features - ['age', 'income',......]
    
    exclude_features: list (default=None)
        list of features to be excluded from WOE transformation.
        - Example - ['age', 'income', .......]
    
    selection_techniques: 'all', 'quick' or list(default='all')
        List of selection techniques to be applied on the data. 
        Available techniques - Weight of evidence ('WOE'), Random Forest ('RF'), 
        Recursive Feature Elimination ('RFE'), Extra Trees Classifier ('ETC'), 
        Chi Square ('CS'), L1 feature selection ('L_ONE').
        
        'all' - Apply all selection techniques ['WOE', 'RF', 'RFE', 'ETC', 'CS', 'L_ONE']
        'quick' - ['WOE','RF','ETC']
        list - user provided list of feature selection techniques from available techniques 
    
    no_of_featues: 'auto', 'sqrt' or int(default='auto')
        Number of features to be selected by each selection technique.
        'auto' - len(features)/2
        'sqrt' - sqrt(len(features)) rounded to the lowest number
        int - user provided number in integer format
    
    handle_category= 'woe' or 'le' (default='woe')
        Handle category values transformation using Label encoder 
        or Weight of Evidence option. Takes care of missing values too. 
        It treats missing values as separate level.
        'woe' - use weight of evidence transformation
        'le' - use label encoder transformation
    
    numerical_missing_values= 'median', 'mean' or 0 (default='median')
        Handle numerical variable missing values.
        'median' - use median of the column
        'mean' - use mean of the column
        0 - use 0 to impute the missing values
    
    minimum_votes = int (default=0)
        Minimum number of votes needed to select a variable after feature selection. 
        Only used in the transform process. Default value is set to 0 to select all variables.
        
    """
    
    def __init__(self, feature_names='all', exclude_features=None, selection_techniques='all', no_of_features='auto',
                 handle_category='woe', numerical_missing_values='median', minimum_votes=0):
        
        self.feature_names = feature_names
        self.exclude_features = exclude_features
        self.selection_techniques = selection_techniques
        self.no_of_features = no_of_features
        self.handle_category = handle_category
        self.numerical_missing_values = numerical_missing_values
        self.minimum_votes = minimum_votes
    
    # check input data type - Only Pandas Dataframe allowed
    def check_datatype(self, X):
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError("The input data must be pandas dataframe. But the input provided is " + str(type(X)))
        return self
    
    # List of feature importance for each variables
    @staticmethod
    def create_dataset(X, algorithm_name, column_names):
        
        #algorithm_name - It refers to the selection technique used to select features
        #column_names - Actual names for each feature
        
        data = pd.DataFrame(X, columns=[algorithm_name], index=column_names)
        data.index.name = 'Variable_Name'
        data = data.reset_index()
        data.sort_values([algorithm_name],ascending=0)
        return data
    
    #function for Weight of evidence
    def woe_information_value(self, X, y):
    
        clf = WOE()
        clf.fit(X, y)
        
        return clf.transform(X), clf.woe_bins, clf.iv_df
    
    #function for random forest classifier
    def random_forest(self, X, y, name):
        
        clf = RandomForestClassifier()
        clf.fit(X, y)
        output = self.create_dataset(clf.feature_importances_, name, X.columns)
        return output
          
    #function for RFE using Logistic Regression
    def recursive_feature_elimination(self, X, y, name):
        
        clf = LogisticRegression()
        rfe = RFE(clf, self.no_of_features)
        rfe.fit(X, y)

        rfe_column_list = X.columns
        rfe_filter_list = rfe.get_support()
        rfe_coef_list = rfe.estimator_.coef_[0].T
        rfe_mapping_list = rfe_column_list[rfe_filter_list]
        rfe_dict_mapping = dict(zip(rfe_mapping_list, rfe_coef_list))
        for i in set(rfe_column_list) - set(rfe_mapping_list):
            rfe_dict_mapping[i] = 0
        
        output = self.create_dataset(rfe_dict_mapping.values(), name, rfe_dict_mapping.keys())
        return output
    
    #function for extra trees classifier
    def extra_trees(self, X, y, name):
        
        clf = ExtraTreesClassifier()
        clf.fit(X, y)
        output = self.create_dataset(clf.feature_importances_, name, X.columns)
        return output
    
    #function for chi square
    def chi_square(self, X, y, name):
        
        clf = SelectKBest(score_func=chi2, k=self.no_of_features)
        clf.fit(X, y)
        output = self.create_dataset(clf.scores_, name, X.columns)
        return output
    
    #execute L1_feature_selection using Linear SVC
    def l1_feature_selection(self, X, y, name):
        
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
        clf = SelectFromModel(lsvc, prefit=True)
        feature_importances = clf.estimator.coef_[0].T
        output = self.create_dataset(feature_importances, name, X.columns)
        return output
        
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
            raise ValueError("The target column y must be binary. But the target contains " + str(len(unique)) \
                             + " unique value(s).")
        
        #assign available techniques
        self.available_techniques = ['WOE', 'RF', 'RFE', 'ETC', 'CS', 'L_ONE']
        
        #Identifying the features to perform fit
        if self.feature_names == 'all':
            self.use_features = X.columns
        else:
            self.use_features = list(set(self.feature_names))
        
        #Exclude variables provided in the exclusion list
        if self.exclude_features:
            self.use_features = list(set(self.use_features) - set(self.exclude_features))
        
        #raise error if there is no features to select
        if not list(self.use_features):
            raise ValueError("No input feature list provided. Please provide a list of features or \
                            set 'all' for feature_names to start the feature selection process")
        
        #determine the number of features to select
        if self.no_of_features == 'auto':
            self.no_of_features = int(len(self.use_features) / 2)
        elif self.no_of_features == 'sqrt':
            self.no_of_features = int(math.sqrt(len(self.use_features)))
        elif isinstance(self.no_of_features, int):
            pass
        else:
            raise ValueError("The number of features to select option is  invalid. \
                            Please provide a valid option - 'auto', 'sqrt' or integer value.")
        
        #start training on the data
        temp_X = X[self.use_features]
        self.feature_importances_, self.feature_votes_ = self.train(temp_X, y)
        
        return self
    
    #return categorical and numerical variables in input data as separate list
    @staticmethod
    def find_variable_type(X):
        
        numerical_features = list(X._get_numeric_data().columns)
        categorical_features = list(X.columns.difference(numerical_features))
        return numerical_features, categorical_features
    
    #fill numeric missing values in input data X
    def fill_numeric_missing_values(self, X, num_miss_present):
        
        num_mapping = {}
        if self.numerical_missing_values == 'median':
            X = X.fillna(X.median())
            for i in num_miss_present:
                num_mapping[i] = X[i].median()
        elif self.numerical_missing_values == 'mean':
            X = X.fillna(X.mean())
            for i in num_miss_present:
                num_mapping[i] = X[i].mean()
        elif self.numerical_missing_values == 0:
            X = X.fillna(0)
            for i in num_miss_present:
                num_mapping[i] = 0
        else:
            raise ValueError("Error while handling imputation for numerical values. \
                            Accepted inputs - 'median', 'mean' or 0")
        
        self.mapping.update(num_mapping)
        return X
    
    #train on the data
    def train(self, X, y):
        
        #determine selection technique based on user provided options
        if self.selection_techniques == 'all':
            self.selection_techniques = ['WOE', 'RF', 'RFE', 'ETC', 'CS', 'L_ONE']
        elif self.selection_techniques == 'quick':
            self.selection_techniques = ['WOE','RF','ETC']
        else:
            if not isinstance(self.selection_techniques, list):
                raise ValueError("The selection techniques provided should be list. \
                                But the input provided is " + str(type(self.selection_techniques)))
            
            if not self.selection_techniques:
                raise ValueError("The selection technique list is empty. Please provide a list of techniques.")
        
        #determine the categorical and numerical features in data and fit them accordingly
        self.numerical_features, self.categorical_features = self.find_variable_type(X)
        
        #assign outputs
        dfs = []
        output_columns = []
        self.mapping={} #mapping for both woe and label encoder
        
        #handle categorical values with either 'woe' or 'le'
        if self.handle_category == 'woe':
            transformed_X, self.mapping, iv_df = self.woe_information_value(X, y) #woe transformed_X
        elif self.handle_category == 'le':
            transformed_X = X.copy(deep=True)
            mapping=defaultdict(LabelEncoder) #le mapping initialize
            le_fit = X[self.categorical_features].fillna("NA").apply(lambda x: mapping[x.name].fit_transform(x)) 
            #transform input data X based on label encoded data
            for i in list(mapping.keys()):
                transformed_X[i] = mapping[i].transform(transformed_X[i].fillna("NA"))
            self.mapping=mapping 
        else:
            raise ValueError("Error while handling categorical values. \
                            Accepted inputs - 'woe' for Weight of evidence and 'le' for Label Encoder.")
        
        #Take the encoded categorical features and the original numerical features from the dataset
        X = pd.concat([transformed_X[self.categorical_features], X[self.numerical_features]], axis=1)
        #self.outX = X (testing feature-now decommissioned)
        
        #check if numeric missing variable needs to be imputed
        num_miss_present = list(X.columns[np.where(X.isnull().sum() > 0)])
        if len(num_miss_present) > 0:
            X = self.fill_numeric_missing_values(X, num_miss_present)
            
        #run woe function
        if 'WOE' in self.selection_techniques:
            name = 'Information_Value'
            if self.handle_category == 'le':
                _, _, iv_df = self.woe_information_value(X, y)
            dfs.append(iv_df)
            output_columns.append(name)
        
        #run random forest function
        if 'RF' in self.selection_techniques:
            name = 'Random_Forest'
            rf_df = self.random_forest(X, y, name)
            dfs.append(rf_df)
            output_columns.append(name)
        
        #run recursive feature elimination function
        if 'RFE' in self.selection_techniques:
            name = 'Recursive_Feature_Elimination'
            rfe_df = self.recursive_feature_elimination(X, y, name)
            dfs.append(rfe_df)
            output_columns.append(name)
        
        #run extra trees classifier function
        if 'ETC' in self.selection_techniques:
            name = 'Extra_Trees'
            etc_df = self.extra_trees(X, y, name)
            dfs.append(etc_df)
            output_columns.append(name)
        
        #run chi square function
        if 'CS' in self.selection_techniques:
            name = 'Chi_Square'
            scaler = MinMaxScaler()
            out_X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            chisq_df = self.chi_square(out_X, y, name)
            dfs.append(chisq_df)
            output_columns.append(name)
        
        # run l1 feature selection function
        if 'L_ONE' in self.selection_techniques:
            name = 'L_One'
            l1_df = self.l1_feature_selection(X, y, name)
            dfs.append(l1_df)
            output_columns.append(name)
        
        if output_columns: #check if any feature selection technique is executed above
            final_results = reduce(lambda left, right: pd.merge(left, right, on='Variable_Name', how='outer'), dfs)
            score_table = pd.DataFrame({}, [])
            score_table['Variable_Name'] = final_results['Variable_Name']
            for i in output_columns:
                #select n largest based on absolute value
                top_n_value_index = final_results[i].abs().nlargest(self.no_of_features).index
                column_index = final_results['Variable_Name'].index
                score_table[i] = column_index.isin(top_n_value_index).astype(int)
            score_table['Votes'] = score_table.sum(axis=1)
            return final_results, score_table.sort_values('Votes',ascending=0)
        else:
            raise ValueError("Please provide a valid selection technique. \
                            Available selection techniques are - Weight of evidence ('WOE'), \
                            Random Forest ('RF'), Recursive Feature Elimination ('RFE'), \
                            Extra Trees Classifier ('ETC'), Chi Square ('CS') \
                            and L1 feature selection ('L_ONE')")
    
    #select the features provided by user and subset input data X
    def transform(self, X):
        
        #if the function is used as part of pipeline, then try to unpack tuple values 
        #produced in the previous step. Added as a part of pipeline feature. 
        try:
            X, y = X
        except:
            pass
        
        #check for the map fitting
        #check_is_fitted(self, 'mapping')
        if not self.mapping:
            raise ValueError("Mapping variable is not present. \
                             Estimator has to be fitted to apply transformations.")
        
        #if the user provided number is greater than the selection techniques 
        #employed then set to the maximum votes of a variable
        if len(self.selection_techniques) < self.minimum_votes:
            self.minimum_votes = max(self.feature_votes_['Votes'])
        
        #select features
        self.use_features = self.feature_votes_[self.feature_votes_['Votes'] >= self.minimum_votes]['Variable_Name']
        X = X[self.use_features] #select the features that passed the minimum votes
        
        #apply mappings
        if self.mapping:
            for k in X.columns:
                if isinstance(self.mapping[k], dict):
                    X[k] = X[k].fillna("NA").replace(self.mapping[k])
                elif isinstance(self.mapping[k], int) or isinstance(self.mapping[k], float):
                    X[k] = X[k].fillna(self.mapping[k])
                elif isinstance(self.mapping[k], LabelEncoder):
                    X[k] = self.mapping[k].transform(X[k].fillna("NA"))
                else:
                    raise ValueError(("Value provided for input variable mapping is not valid. \
                                     Occurred at variable name: %s and mapping: %s. Please check the mapping value") \
                                     % (str(k), str(v)))
        
        return X
