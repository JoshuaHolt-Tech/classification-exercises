import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def prep_telco(working_df):
    """
    This function takes an input from the titanic data set and returns a
    cleaner set of data.
    """
    
    cols_to_drop = ['Unnamed: 0', 'customer_id', 'payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id']
    
    working_df.drop(columns=cols_to_drop, inplace = True)
    working_df['total_charges'] = working_df['total_charges'].replace(' ', 0).astype('float')
    encode_list = ['churn', 'paperless_billing', 'partner' , 'dependents']
    
    for col in working_df.columns:
        if col in encode_list:
            working_df[col] = working_df[col].replace({'Yes':True,'No':False}).astype('int')
        
    dummy_df = (pd.get_dummies(working_df[['gender', 'partner', 'streaming_movies', 'streaming_tv' , 'tech_support','dependents', 'multiple_lines', 'online_backup', 'online_security', 'device_protection', 'payment_type', 'internet_service_type', 'contract_type']], drop_first=True))
    
    working_df = pd.concat([working_df, dummy_df], axis=1)
    
    return working_df



def prep_titanic(df):
    """
    This function takes a dataframe with the titantic data set
    and areturns a cleaner version.
    """
    
    cols_to_drop = ['Unnamed: 0', 'passenger_id', 'age', 'embarked', 'class', 'deck']
    df.drop(columns=cols_to_drop, inplace = True)
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    return df



def prep_iris(df):
    """ 
    This function takes a dataframe from the iris dataset 
    containing the sepal_length, sepal_width, petal_length, 
    petal_width, species_name, species_id columns and returns 
    a cleaner dataset.
    """
    #Assigns column to a list and drops species_id:
    cols_to_drop = ['Unnamed: 0', 'species_id'] #Never imported the measurements_id column
    df = df.drop(columns=cols_to_drop)
    
    #Shortens the species column name:
    df.rename(columns={'species_name':'species'}, inplace=True)
    
    #Assigns two dummy columns and attach them to the dataframe.
    dummy_df = pd.get_dummies(df[['species']], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    return df