import pandas as pd
import numpy as np
import os

# acquire
from env import host, user, password
from pydataset import data
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")

os.path.isfile('mallcustomers_df.csv')


# Create helper function to get the necessary connection url.
def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

    
    

# Use the above helper function and a sql query in a single function.
def new_mall_data():
    '''
    This function reads data from the Codeup db into a df.
    '''
    mall_customers_sql = "SELECT * \
                  FROM customers;;" \
    
    
    return pd.read_sql(mall_customers_sql, get_connection('mall_customers'))

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

def features_missing(df):
    
    '''This function creates a new dataframe that analyzes the total features(columns) missing for the rows in
    the data frame. It also give s a percentage'''
    
    
    df2 = df.isnull().sum(axis =1).value_counts().sort_index(ascending=False)
    df2 = pd.DataFrame(df2)
    df2.reset_index(level=0, inplace=True)
    df2.columns= ['total_features_missing', 'total_rows_affected'] 
    df2['pct_features_missing']= round((df2.total_features_missing /df.shape[1]) * 100, 2)
    df2 = df2[['total_features_missing', 'pct_features_missing', 'total_rows_affected']]
    
    return df2

def split_data(df):
    '''
    split our data,
    takes in a pandas dataframe
    returns: three pandas dataframes, train, test, and validate
    '''
    train_val, test = train_test_split(df, train_size=0.8, random_state=123)
    train, validate = train_test_split(train_val, train_size=0.7, random_state=123)
    
    
    return train, validate, test

def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # create a mask of columns whether they are object type or not
    mask = np.array(df.dtypes == "object")

        
    # get a list of the column names that are objects (from the mask)
    object_cols = df.iloc[:, mask].columns.tolist()
    
    return object_cols

def get_numeric_X_cols(train, object_cols):
    '''
    takes in a dataframe and list of object column names
    and returns a list of all other columns names, the non-objects. 
    '''
    numeric_cols = [col for col in train.columns.values if col not in object_cols]
    
    return numeric_cols

def min_max_scale(train, validate, test, numeric_cols):
    '''
    this function takes in 3 dataframes with the same columns, 
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a min-max scaler to the first dataframe and transforms all
    3 dataframes using that scaler. 
    it returns 3 dataframes with the same column names and scaled values. 
    '''
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).
     
    scaler = MinMaxScaler(copy=True).fit(train)

    #scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train. 
    # 
    train_scaled_array = scaler.transform(train[numeric_cols])
    validate_scaled_array = scaler.transform(validate[numeric_cols])
    test_scaled_array = scaler.transform(test[numeric_cols])

    # convert arrays to dataframes
    train_scaled = pd.DataFrame(train_scaled_array, 
                                columns=numeric_cols).\
                                set_index([train.index.values])

    validate_scaled = pd.DataFrame(validate_scaled_array,
                                   columns=numeric_cols).\
                                   set_index([validate.index.values])

    test_scaled = pd.DataFrame(test_scaled_array,
                               columns=numeric_cols).\
                                set_index([test.index.values])

    
    return train_scaled, validate_scaled, test_scaled