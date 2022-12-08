import os
import pandas as pd
import numpy as np
from env import get_connection

"""
This is how you get rid of the Unnamed: 0 column:

#read_csv(filename, index_col=0)
#to_csv(filename, index=False)
"""

def get_titanic_data():
    filename = "titanic.csv"
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_file(filename)
        
        # Return the dataframe to the calling code
        return df
    
def get_iris_data():
    """
    This function reads the iris data from Codeup db into a df.
    """
    filename = "iris.csv"
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        query = """
        SELECT sepal_length, sepal_width, petal_length, petal_width, species_name, species_id FROM measurements
        LEFT JOIN species USING (species_id);
        """
        # read the SQL query into a dataframe
        df = pd.read_sql(query, get_connection('iris_db'))
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)
        
        # Return the dataframe to the calling code
        return df
    
def get_telco_data():
    """
    This function reads the telco_churn data from Codeup db into a df.
    """
    filename = "telco_churn.csv"
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        query = """
        SELECT * FROM customers 
        LEFT JOIN contract_types USING (contract_type_id)
        LEFT JOIN internet_service_types USING (internet_service_type_id)
        LEFT JOIN payment_types USING (payment_type_id);
        """
        df = pd.read_sql(query, get_connection('telco_churn'))
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)
        
        # Return the dataframe to the calling code
        return df