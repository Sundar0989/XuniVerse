import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
#%matplotlib inline
import matplotlib as mpl
mpl.use('Agg')

pd.options.mode.chained_assignment = None 

class BarCharts(BaseEstimator, TransformerMixin):
    
    """Bar charts showing relationship between X and y (target). Input data for the BarCharts module 
    is the output of WOE fit function (woe_df).
    
    Parameters
    ----------
    feature_names: 'all' or list (default='all')
        list of features to make bivariate charts. 
        For numerical features, WOE transformation will be performed which internally does monotonic binning.
        - 'all' (default): All features in the dataset will be used. 
        - list of features: ['age', 'income',......]
    
    exclude_features: list (default=None)
        list of features to be excluded from making charts.
        - Example - ['age', 'income', .......]
        
    plot_metric: 'count' or 'mean' (default='mean')
        Metric to be used while plotting the bivariate chart.
        'count' - Event counts in the particular bin
        'mean' - Mean event rate in the particular bin
        
    bar_type: 'horizontal' or 'vertical' (default='vertical')
        Type of bar chart.
        
    fig_size: figure size for each of the individual plots (default=(8,6))
    
    bar_color: CSS color style picker. Use it with the hashtag in the front. (default='#058caa')
        Bar color
        
    num_color: CSS color style picker. Use it with the hashtag in the front (default='#ed8549')
        Numbers color. It represents the numbers written on top of the bar.
        
    """
    
    # Initialize the parameters for the function
    def __init__(self, feature_names='all', exclude_features=None, plot_metric='mean', 
                 bar_type='vertical', fig_size=(8,6), bar_color='#058caa', num_color = '#ed8549'):
        
        self.feature_names = feature_names
        self.exclude_features = exclude_features
        self.plot_metric = plot_metric
        self.bar_type = bar_type
        self.fig_size = fig_size
        self.bar_color = bar_color
        self.num_color = num_color
    
    # check input data type - Only Pandas Dataframe allowed
    def check_datatype(self, X):
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError("The input data must be pandas dataframe. But the input provided is " + str(type(X)))
        return self
        
    # the plot function for barcharts on WOE data
    def plot(self, X):
        
        #check datatype of X
        self.check_datatype(X)
        
        #The length of X and Y should be equal
        if X.shape[0] == 0:
            raise ValueError("Input data is empty. Please provide a valid dataset")
        
        #the input features available in woe dataset
        self.available_features = list(pd.Series.unique(X['Variable_Name']))
        
        #Identifying the features to perform fit
        if self.feature_names == 'all':
            self.plot_features = self.available_features
        else:
            self.plot_features = list(set(self.available_features).intersection(set(self.feature_names)))
        
        #Exclude variables provided in the exclusion list
        if self.exclude_features:
            self.plot_features = list(set(self.plot_features) - set(self.exclude_features))
        
        if not self.plot_features:
            raise ValueError("No features to plot for the given data")
        
        temp_X = X.copy(deep=True)
        temp_X = temp_X[temp_X['Variable_Name'].isin(self.plot_features)] #subset data only on features to fit
        temp_X['Event_Rate'] = temp_X['Event_Rate'] * 100
        
        #plot for each variable
        self._plot(temp_X)
                
        return self
    
    #the plot function for individual variable
    #Here input X is a pandas Series variable
    def _plot(self, X):
            
        bar_color = self.bar_color
        num_color = self.num_color
        plot_metric = self.plot_metric
        plot_figsize = self.fig_size
        bar_type = self.bar_type

        if plot_metric == 'mean':
            plot_on = 'Event_Rate'
            percentage_label = '%'
            y_label = 'Mean target'
        elif plot_metric == 'count':
            plot_on = 'Event'
            percentage_label = ''
            y_label = 'Target count'
        else:
            raise ValueError("Plot metric option is invalid. Available options - 'mean' for Event rate \
                            or 'count' for Event count.")

        if bar_type in ['vertical', 'v']:
            bar_kind = 'bar'
        elif bar_type in ['horizontal', 'h']:
            bar_kind = 'barh'
        else:
            raise ValueError("Bar type provided is invalid. Available options - 'vertical' or 'v'; \
                            'horizontal' or 'h'.")
            
        grouped = X.groupby(['Variable_Name'])
        
        for key, group in grouped:
        
            ax = group.plot('Category', plot_on, kind=bar_kind, color=bar_color, linewidth=1.0, \
                            edgecolor=['black'], figsize=plot_figsize)
            ax.set_title(str(key) + " vs " + str('target'))
            ax.set_xlabel(key)
            ax.set_ylabel(y_label)
            rects = ax.patches
            for rect in rects:
                if bar_type in ['vertical', 'v']:
                    text_placer = [rect.get_x(), rect.get_height(), rect.get_width()]
                    text_x_pos = text_placer[0]+text_placer[2]/2.
                    text_y_pos = 1.01*text_placer[1]
                else:      
                    text_placer = [rect.get_y(), rect.get_width(), rect.get_height()]
                    text_x_pos = 1.1*text_placer[1]
                    text_y_pos = text_placer[0]+text_placer[2]/2.
                ax.text(text_x_pos, text_y_pos, str(round(text_placer[1],1)) + str(percentage_label), \
                        ha='center', va='bottom', color=num_color, fontweight='bold')
        
        plt.show()
