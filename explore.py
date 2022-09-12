import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# =================== GENERAL ======================

alf = 0.05 # Rest well, sweet prince. [alf = alpha]

def col_range(df):
    '''
    Takes in a data frame, returns the 'describe' of the data frame with a new entry 'range'.
    'Range' is the difference between the 'max' and 'min' columns.
    '''
    stats_df = df.describe().T
    stats_df['range'] = stats_df['max'] - stats_df['min']
    print(stats_df)


# =================== VIS EXPLORE ======================

def plot_variable_pairs(df, num = ['area', 'tax_value']):
    '''
    Takes in DF (Train Please,) and numerical column pair 
    [Preset Numerical Data is area, and tax_value, but a new list can be fed in.]
    Returns the plotted out variable pairs heatmap and numerical pairplot.
    '''
    df_corr = df.corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(df_corr, cmap='Purples', annot = True, mask= np.triu(df_corr), linewidth=.5)
    plt.title("Correlations between variables")
    plt.show()
    
    sns.pairplot(df[num].sample(1_000), corner=True, kind='reg', plot_kws={'line_kws':{'color':'red'}})
    plt.show()

def plot_categorical_and_continuous_vars(df, cat = ["bedrooms", "bathrooms"], target = 'tax_value'):
    '''
    Takes a Data Frame,
    plots a continuous varible [default target value of tax_value](please enter an int/float column) as y
    sorted by categorical variable [default bedrooms, bathrooms] (Year_built not included, there's over 100 years, not exactly categorical to the scale we want) as x.
    Returns swarm plot, violin plot, and cat plot for each.
    '''
    for col in cat:
        sns.swarmplot(data=df.sample(800), x=col, y=target, s=3)
        plt.show()

        sns.violinplot(data=df.sample(1_000), x=col, y=target, s=3)
        plt.show()
        
        sns.catplot(data=df.sample(500), x=col, y=target, s=2)
        plt.show()

def explore_cat(df, cat = ['bedrooms', 'bathrooms'], target = 'tax_value'):
    '''
    Takes in dataframe (Remember, explore TRAIN!), 
    categorial columns (default is bedrooms, and bathrooms),
    target column (tax_value default), 
    returns printed value counts for each category in each column
    '''
    for col in cat:
        print(f"-=== {col} ===-")
        print(' ')
        print("Value Counts:")
        print("-------------")
        print(df[col].value_counts())
        print(" ")
        print("Percent of Total:")
        print("-----------------")
        print(df[col].value_counts(normalize=True)*100)
        print(" ")
        o = pd.crosstab(df[col], df[target])
        chi2, p, dof, e = stats.chi2_contingency(o)
        result = p < alf
        if result == True:
            print(' >> P is less than Alpha.')
        else:
            print(" >> P is greater than Alpha.")
        print(" ")
        sns.countplot(x=col, data=df)
        plt.title(col + ' counts')
        plt.show()
    
        sns.barplot(data=df, x=col, y=target)
        rate = df[target].mean()
        plt.axhline(rate, label= 'average ' + target + ' rate')
        plt.legend()
        plt.title(target + ' rate by ' + col)
        plt.show()

        print(' ')
        print('======================================')
        print(' ')
    print("-=== tax_value ===-")
    print(' ')
    print("Tax Value Average:")
    print(f"$ {round(df.tax_value.sum()/len(df.tax_value), 2)}")
    print("-------------")


def explore_num(df, num = ['area', 'tax_value']):
    '''
    Takes in DataFrame (Remember, explore w/ Train!), 
    numerical columns to explore (default area, tax_value).
    '''
    for col in num:
        sns.histplot(data=df, x=col, hue='bathrooms')
        plt.show()  

    cols = [col for col in num if col not in ['tax_value']]
    for col in cols:
        sns.scatterplot(data=df, x=col, y='tax_value', hue='bathrooms')
        plt.show()

    
# ============== Feature Selection ====================
# Do not rely solely on these.

def select_kbest(X_train, y_train, k_features):
    '''
    Takes in X_train, y_train, and the number of features to select.
    Returns the names of the selected features using SelectKBest from sklearn. 
    '''
    kbest = SelectKBest(f_regression, k=k_features)
    kbest.fit(X_train, y_train)
    
    print(X_train.columns[kbest.get_support()].tolist())
    
    
def select_rfe(X_train, y_train, k_features):
    '''
    Takes in X_train, y_train, and the number of features to select.
    Returns the names of the selected features using Recursive Feature Elimination from sklearn. 
    '''
    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=k_features)
    rfe.fit(X_train, y_train)
    
    print(X_train.columns[rfe.support_].tolist())
    
    
def select_sfs(X_train, y_train, k_features):
    '''
    Takes in X_train, y_train, and the number of features to select.
    Returns the names of the selected features using Sequential Feature Selector from sklearn. 
    '''
    model = LinearRegression()
    sfs = SequentialFeatureSelector(model, n_features_to_select=k_features)
    sfs.fit(X_train, y_train)
    
    print(X_train.columns[sfs.support_].tolist())

def select_best(X_train, y_train, k_features):
    '''
    Takes in X_train, y_train, and the number of features to select.
    Runs select_kbest, select_rfe, and select sfs functions to find the different bests.
    Returns bests.
    '''
    print("KBest:")
    print(select_kbest(X_train, y_train, k_features))
    print(" ")
    print("RFE:")
    print(select_rfe(X_train, y_train, k_features))
    print(" ")
    print("SFS:")
    print(select_sfs(X_train, y_train, k_features))


# =============== Modeling ===================

def get_rmse(y_train, target):
    y_train['baseline_mean'] = y_train[target].mean()
    y_train['baseline_median'] = y_train[target].median()

    # scores:
    rmse_mean = mean_squared_error(y_train.tax_value,
                                y_train['baseline_mean'], squared=False)
    rmse_med = mean_squared_error(y_train.tax_value,
                                y_train['baseline_median'], squared=False)

    print("RMSE Mean:")
    print(rmse_mean)
    print("----------------")
    print("RMSE Median:")
    print(rmse_med)
    print("----------------")
    if rmse_mean < rmse_med:
        print(f"RMSE Mean is lower so we'll use that : {rmse_mean}")
    elif rmse_med < rmse_mean:
        print(f"RMSE Median is lower so we'll use that : {rmse_med}")