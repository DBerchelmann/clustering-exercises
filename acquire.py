import pandas as pd
import numpy as np
import os

# acquire
from env import host, user, password
from pydataset import data


# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")

os.path.isfile('zillowcluster_df.csv')


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