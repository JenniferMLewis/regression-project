from tkinter.tix import COLUMN
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.impute import SimpleImputer

from env import user, password, host


# ================   General   ==================

def get_url(db, user=user, password=password, host=host):
    '''
    Takes database name for input,
    returns url, using user, password, and host pulled from your .env file.
    PLEASE save it as a variable, and do NOT just print your credientials to your document.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


def remove_outliers(df, col_list, k = 1.5):
    ''' 
    Accepts a Dataframe, a column list of columns you want to affect, 
    and a k variable (defines how far above and below the quartiles you want to go)[Default 1.5].
    Removes outliers from a list of columns in a dataframe 
    and return that dataframe.

    '''
    for col in col_list:
        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        iqr = q3 - q1   # calculate interquartile range

        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound
        # return dataframe without outliers
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    return df


def hist_plot(df):
    '''
    Plots Histograms for columns in the input Data Frame, 
    all but 'date', 'fips' and, 'year_built' as they're categorical not actually numbers.
    '''
    plt.figure(figsize=(16, 3))

    cols = [col for col in df.columns if col not in ['fips', 'year_built', 'date']]

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1 <-- Good to note
        plot_number = i + 1 
        plt.subplot(1, len(cols), plot_number)
        plt.title(col)
        df[col].hist(bins=5)
        # We're looking for shape not actual details, so these two are set to 'off'
        plt.grid(False)
        plt.ticklabel_format(useOffset=False)
        # mitigate overlap: This is handy. Thank you.
        plt.tight_layout()

    plt.show()


def box_plot(df, cols = ['bedrooms', 'bathrooms', 'area', 'tax_value']
):
    ''' 
    Takes in a Data Frame, and list of columns [Default : bedrooms, bathrooms, area, and tax_value]
    Plots Boxplots of input columns.
    '''
    
    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):
        plot_number = i + 1 
        plt.subplot(1, len(cols), plot_number)
        plt.title(col)
        sns.boxplot(data=df[[col]])
        plt.grid(False)
        plt.tight_layout()

    plt.show()


# ================   MVP   =================


def get_zillow_mvp():
    '''
    Returns the zillow 2017 dataset, checks local disk for zillow_2017_mvp.csv, if present loads it,
    otherwise it pulls the bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, 
    and taxvaluedollarcnt columns for Single Family Residential Property types with a transaction date in 2017
    from the SQL.
    '''
    filename = 'zillow_2017_mvp.csv'

    if os.path.isfile(filename):
        df = pd.read_csv(filename)
        df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                      'bathroomcnt':'bathrooms', 
                      'calculatedfinishedsquarefeet':'area',
                      'taxvaluedollarcnt':'tax_value',
                      "transactiondate" : "date"})
        df = df[df.bedrooms!=0]
        df = df.drop(columns="Unnamed: 0")
        return df
    else:
        df = pd.read_sql('''SELECT bedroomcnt, 
        bathroomcnt, 
        calculatedfinishedsquarefeet, 
        taxvaluedollarcnt,
        transactiondate
        FROM properties_2017
        JOIN propertylandusetype
        USING(propertylandusetypeid)
        JOIN predictions_2017
        USING(parcelid)
        WHERE propertylandusedesc = "Single Family Residential";
        ''', get_url('zillow'))
        df.to_csv(filename)
        df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                      'bathroomcnt':'bathrooms', 
                      'calculatedfinishedsquarefeet':'area',
                      'taxvaluedollarcnt':'tax_value'})
        df = df[df.bedrooms!=0]
        return df


def prepare_zillow_mvp(df):
    ''' 
    Prepares Zillow data for exploration
    Removes Outliers, Shows Distributions of Numeric Data via Histograms and Box plots,
    Converts bedrooms datatype from String to Float, Splits Data into Train, Validate, Test.
    Returns the Train, Validate, and Text Dataframes.
    '''

    # removing outliers
    df = remove_outliers(df, ['bedrooms', 'bathrooms', 'area', 'tax_value'])
    
    # dropping the row without a 2017 date

    hist_plot(df)
    box_plot(df)
    df = df.drop(columns= "date")
    
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

    return train, validate, test 


def wrangle_zillow_mvp():
    '''
    Acquire and prepare data from Zillow database for explore,
    Uses get_zillow_mvp and prepare_zillow_mvp functions.
    Returns Cleaned, Outlier Removed, Train, Validate, and Test Data Frames.
    '''
    train, validate, test = prepare_zillow_mvp(get_zillow_mvp())
    
    return train, validate, test


# ================ Expanded ==================

def get_zillow():
    '''
    Returns the zillow 2017 dataset, checks local disk for zillow_2017.csv, if present loads it,
    otherwise it pulls the bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, 
    taxvaluedollarcnt, yearbuilt, taxamount and, fips columns from the SQL.
    Dropping all properties with 0 bedrooms (it's not really a house if you can't sleep in it.)
    '''
    filename = 'zillow_2017.csv'

    if os.path.isfile(filename):
        df = pd.read_csv(filename)
        df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                      'bathroomcnt':'bathrooms', 
                      'calculatedfinishedsquarefeet':'area',
                      'taxvaluedollarcnt':'tax_value', 
                      'yearbuilt':'year_built',
                      'taxamount':'tax_amount',
                      'transactiondate' : "date"})
        df = df[df.bedrooms!=0]
        return df
    else:
        df = pd.read_sql('''SELECT bedroomcnt, 
        bathroomcnt, 
        calculatedfinishedsquarefeet, 
        taxvaluedollarcnt, 
        yearbuilt, 
        taxamount, 
        fips 
        transactiondate
        FROM properties_2017
        JOIN propertylandusetype
        USING(propertylandusetypeid)
        JOIN predictions_2017
        USING(parcelid)
        WHERE propertylandusedesc = "Single Family Residential";
        ''', get_url('zillow'))
        df.to_csv(filename)
        df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                      'bathroomcnt':'bathrooms', 
                      'calculatedfinishedsquarefeet':'area',
                      'taxvaluedollarcnt':'tax_value', 
                      'yearbuilt':'year_built',
                      'taxamount':'tax_amount',
                      'transactiondate':'date'})
        df = df[df.bedrooms!="0"]
        return df


def prepare_zillow(df):
    ''' Prepare zillow data for exploration'''

    # removing outliers
    df = remove_outliers(df, ['bedrooms', 'bathrooms', 'area', 'tax_value', 'tax_amount'])
    
    # getting distributions for numeric data
    hist_plot(df)
    box_plot(df)
    
    # converting column datatypes
    df['fips'] = df.fips.astype(str)
    df['bedrooms'] = df.bedrooms.astype(float)
    
    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    # impute year built using median (If the teacher is willing to, then it -should- be okay?)
    imputer = SimpleImputer(strategy='median')

    imputer.fit(train[['year_built']])

    train[['year_built']] = imputer.transform(train[['year_built']])
    validate[['year_built']] = imputer.transform(validate[['year_built']])
    test[['year_built']] = imputer.transform(test[['year_built']])       

    train['year_built'] = train.year_built.astype(str)
    validate['year_built'] = validate.year_built.astype(str)
    test['year_built'] = test.year_built.astype(str)

    return train, validate, test 


def wrangle_zillow():
    '''Acquire and prepare data from Zillow database for explore using acquire and prepare functions above.'''
    train, validate, test = prepare_zillow(get_zillow())
    
    return train, validate, test


# ============ MISC =============

def df_split(train, validate, test, target="tax_value"):
    '''
    Takes in train, validate, and test df, as well as target (default: "tax_value")
    Splits them into X, y using target.
    Returns X, y of train, validate, and test.
    y sets returned as a proper DataFrame.
    '''
    X_train, y_train = train.drop(columns=target), train[target]
    X_validate, y_validate = validate.drop(columns=target), validate[target]
    X_test, y_test = test.drop(columns=target), test[target]
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)
    return X_train, y_train, X_validate, y_validate, X_test, y_test


def scale(df, columns_for_scaling = ['bedrooms', 'bathrooms', 'tax_value'], scaler = MinMaxScaler()):
    '''
    Takes in df, columns to be scaled (default: bedrooms, bathrooms, tax_value), 
    and scaler (default: MinMaxScaler(); others can be used ie: StandardScaler(), RobustScaler(), QuantileTransformer())
    returns a copy of the df, scaled.
    '''
    scaled_df = df.copy()
    scaled_df[columns_for_scaling] = scaler.fit_transform(df[columns_for_scaling])
    return scaled_df