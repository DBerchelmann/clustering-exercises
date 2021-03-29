import pandas as pd
import numpy as np
import os

# acquire
from env import host, user, password
from pydataset import data
from datetime import date 
from scipy import stats

# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

# Create helper function to get the necessary connection url.
def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

    
    

  
    
    


# Use the above helper function and a sql query in a single function.
def new_zillow_data():
    '''
    This function reads data from the Codeup db into a df.
    '''
    zillow_sql = "SELECT * \
                        FROM properties_2017 \
                        JOIN (SELECT parcelid, max(logerror) as logerror, max(transactiondate) as transactiondate \
                              FROM predictions_2017 group by parcelid) as pred_17 using(parcelid) \
                        LEFT JOIN airconditioningtype using(airconditioningtypeid) \
                        LEFT JOIN architecturalstyletype using(architecturalstyletypeid) \
                        LEFT JOIN buildingclasstype using(buildingclasstypeid) \
                        LEFT JOIN heatingorsystemtype using(heatingorsystemtypeid) \
                        LEFT JOIN storytype using(storytypeid) \
                        LEFT JOIN typeconstructiontype using(typeconstructiontypeid) \
                        WHERE year(transactiondate) = 2017;" \
    
    
    return pd.read_sql(zillow_sql, get_connection('zillow'))



def get_zillow_data(cached=False):
    '''
    This function reads in telco churn data from Codeup database and writes data to
    a csv file if cached == False or if cached == True reads in telco df from
    a csv file, returns df.
    '''
    if cached == False or os.path.isfile('zillow_df.csv') == False:
        
        # Read fresh data from db into a DataFrame.
        df = new_zillow_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('zillowcluster_df.csv')
        
    else:
        
        # If csv file exists or cached == True, read in data from csv.
        df = pd.read_csv('zillowcluster_df.csv', index_col=0)
        
    return df


def split_data():
    '''
    split our data,
    takes in a pandas dataframe
    returns: three pandas dataframes, train, test, and validate
    '''
    train_val, test = train_test_split(df, train_size=0.8, random_state=123)
    train, validate = train_test_split(train_val, train_size=0.7, random_state=123)
    
    
    return train, validate, test


def clean_zillow(df):
    '''This functions cleans our dataset using a variety of tools:
    
    '''
    
    return df


 # wrangle!
def wrangle_zillow():
    '''
    wrangle_zillow will read in our zillow data as a pandas dataframe,
    clean the data
    split the data
    return: train, validate, test sets of pandas dataframes from zillow
    '''
    df = clean_zillow(new_zillow_data())
    
    
    
    return df

    


''''''''''''''''''
'Helper Functions'
'                '
''''''''''''''''''


def missing_zero_values_table(df):
    
    '''This function will look at any data set and report back on zeros and nulls for every column while also giving percentages of total values
        and also the data types. The message prints out the shape of the data frame and also tells you how many columns have nulls '''
    
    
    zero_val = (df == 0.00).astype(int).sum(axis=0)
    null_count = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mz_table = pd.concat([zero_val, null_count, mis_val_percent], axis=1)
    mz_table = mz_table.rename(
    columns = {0 : 'Zero Values', 1 : 'null_count', 2 : '% of Total Values'})
    mz_table['Total Zeroes + Null Values'] = mz_table['Zero Values'] + mz_table['null_count']
    mz_table['% Total Zero + Null Values'] = 100 * mz_table['Total Zeroes + Null Values'] / len(df)
    mz_table['Data Type'] = df.dtypes
    mz_table = mz_table[
        mz_table.iloc[:,1] >= 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"      
            "There are " +  str((mz_table['null_count'] != 0).sum()) +
          " columns that have NULL values.")
#         mz_table.to_excel('D:/sampledata/missing_and_zero_values.xlsx', freeze_panes=(1,0), index = False)

    return mz_table

missing_zero_values_table(df)


def handle_missing_values(df, prop_required_row = 0.5, prop_required_col = 0.5):
    ''' function which takes in a dataframe, required notnull proportions of non-null rows and columns.
    drop the columns and rows columns based on threshold:'''
    
    #drop columns with nulls
    threshold = int(prop_required_col * len(df.index)) # Require that many non-NA values.
    df.dropna(axis = 1, thresh = threshold, inplace = True)
    
    #drop rows with nulls
    threshold = int(prop_required_row * len(df.columns)) # Require that many non-NA values.
    df.dropna(axis = 0, thresh = threshold, inplace = True)
    
    
    return df


def features_missing(df):
    
    '''This function creates a new dataframe that analyzes the total features(columns) missing for the rows in
    the data frame. It also give s a percentage'''
    
    # Locate rows with. missing features and convert into a series
    df2 = df.isnull().sum(axis =1).value_counts().sort_index(ascending=False)
    
    # convert into a dataframe
    df2 = pd.DataFrame(df2)
    
    # reset the index
    df2.reset_index(level=0, inplace=True)
    
    # rename the columns for readability
    df2.columns= ['total_features_missing', 'total_rows_affected'] 
    
    # create a column showing the percentage of features missing from a row
    df2['pct_features_missing']= round((df2.total_features_missing /df.shape[1]) * 100, 2)
    
    # reorder the columns for readability/scanning
    df2 = df2[['total_features_missing', 'pct_features_missing', 'total_rows_affected']]
    
    return df2