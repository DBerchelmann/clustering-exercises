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
    
    
    
    
    
    # Create new column (age_of_home)
    today = pd.to_datetime('today')
    df['age_of_home'] = today.year - df['yearbuilt']
    
    # clean up longitude (this cleans latitude as well)
    longindex = df.loc[df['longitude'].isin(['NaN'])].index
    df.drop(longindex , inplace=True)
        
    # drop rows that have a property id less than 250 (not single unit)
    cleanpropindexes250 = df.loc[df['propertylandusetypeid'] <= 250].index
    df.drop(cleanpropindexes250 , inplace=True)
    
    # drop any rows that have unit count greater than one (39 rows)
    singlecountindexes = df.loc[df['unitcnt'] > 1].index
    df.drop(singlecountindexes , inplace=True)
    
    # turn null values in pool into value 0
    df['poolcnt'].fillna(0, inplace=True)
    
    # Create bathrooms per sqft
    df['bath_pers_qft'] = df['bathroomcnt'] / df['calculatedfinishedsquarefeet']
    
    indextaxvalue = df.loc[df['taxvaluedollarcnt'].isin(['NaN'])].index
    df.drop(indextaxvalue , inplace=True)
    
    # If "fireplaceflag" is "True" and "fireplacecnt" is "NaN", we will set "fireplacecnt" equal to the median value of "1".
    df.loc[(df['fireplaceflag'] == True) & (df['fireplacecnt'].isnull()), ['fireplacecnt']] = 1
    
    # If 'fireplacecnt' is "NaN", replace with "0"
    df.fireplacecnt.fillna(0,inplace = True)
    
    # If "fireplacecnt" is 1 or larger "fireplaceflag" is "NaN", we will set "fireplaceflag" to "True".
    df.loc[(df['fireplacecnt'] >= 1.0) & (df['fireplaceflag'].isnull()), ['fireplaceflag']] = True
    df.fireplaceflag.fillna(0,inplace = True)
    
    # Convert "True" to 1
    df.fireplaceflag.replace(to_replace = True, value = 1,inplace = True)
    
    # Remove NaNs from tax amount
    indextax = df.loc[df['taxamount'].isin(['NaN'])].index
    df.drop(indextax , inplace=True)
    
    # Remove decimal from latitude and longitude
    df['latitude'] = df['latitude'].astype(int)
    df['longitude'] = df['longitude'].astype(int)
    
    # Convert latitude and longitude to positonal data points using lambda funtion (i.e. putting a decimal in the correct place)
    df['latitude'] = df['latitude'].apply(lambda x: x / 10 ** (len((str(x))) - 2))
    df['longitude'] = df['longitude'].apply(lambda x: x / 10 ** (len((str(x))) - 4))
    
    
    # Remove null years (small amount)
    indexyear = df.loc[df['yearbuilt'].isin(['NaN'])].index
    df.drop(indexyear , inplace=True)
    
    # change columns to int
    df['fips'] = df['fips'].astype(int)
    df['yearbuilt'] = df['yearbuilt'].astype(int)
    
    # Fill in lot size square footage nulls with median
    df['lotsizesquarefeet'].fillna((df['lotsizesquarefeet'].median()), inplace=True)
    
    # Clean up square footage
    indexsquarefeet = df.loc[df['calculatedfinishedsquarefeet'].isnull()].index 
    df.drop(indexsquarefeet, inplace=True)
    
    # Clean up tax amount
    indextaxamount = df.loc[df['taxamount'].isnull()].index 
    df.drop(indextaxamount, inplace=True)

    # Fill Bedroom Nulls
    df['bedroomcnt'].replace(0,df['bedroomcnt'].median(axis=0),inplace=True)
    
    # Fill Bathroom Nulls
    df['bathroomcnt'].replace(0,df['bathroomcnt'].median(axis=0),inplace=True)
    
    # Heating System
    df["heatingorsystemdesc"].fillna("None", inplace = True)
    df['heatingorsystemtypeid'].fillna(1, inplace=True)
    
    # AC
    df["airconditioningdesc"].fillna("None", inplace = True)
    df["airconditioningtypeid"].fillna(5,inplace = True)

    # Drop columns
    dropcols = ['regionidzip', 'finishedsquarefeet12', 'propertyzoningdesc', 'buildingqualitytypeid', 'regionidzip', 'calculatedbathnbr', 
                'fullbathcnt', 'landtaxvaluedollarcnt', 'structuretaxvaluedollarcnt', 'censustractandblock', 'regionidcity', 'unitcnt',
                'rawcensustractandblock', 'Unnamed: 0', 'propertycountylandusecode', 'regionidcounty', 'assessmentyear', 'propertylandusetypeid', 'id']
    df.drop(dropcols, axis=1, inplace=True)
    
    # convert columns to object
    df['fips'] = df['fips'].astype(object)
    df['yearbuilt'] = df['yearbuilt'].astype(object)
    
     # Columns to look for outliers
    df = df[df.taxvaluedollarcnt < 3_000_000]
    df = df[df.calculatedfinishedsquarefeet < 8000]
    
    dummy_df = pd.get_dummies(df['fips'])
    dummy_df.columns = ['la_county', 'orange_county', 'ventura_county']
    df = pd.concat([df, dummy_df], axis=1)
    
   
      
    # convert column to date time
    df['transactiondate'] = pd.to_datetime(df['transactiondate'])
    
    # Set parcelid as the index
    df = df.set_index('parcelid')
    

 
    
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

def outlier_function(df, cols, k):
	#function to detect and handle oulier using IQR rule
    for col in df[cols]:
        q1 = df.annual_income.quantile(0.25)
        q3 = df.annual_income.quantile(0.75)
        iqr = q3 - q1
        upper_bound =  q3 + k * iqr
        lower_bound =  q1 - k * iqr     
        df = df[(df[col] < upper_bound) & (df[col] > lower_bound)]
    return df