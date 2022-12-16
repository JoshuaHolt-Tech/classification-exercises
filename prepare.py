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
    
    #Total_charges has null entries and they are replaced with 0.
    #The assumption is these are new accounts where payment has not been made yet.
    working_df['total_charges'] = working_df['total_charges'].replace(' ', 0).astype('float')
    
    #Changes string yes/no to int 0/1 for use in machine learning algorithm:
    encode_list = ['churn', 'paperless_billing', 'partner' , 'dependents']
    for col in working_df.columns:
        if col in encode_list:
            working_df[col] = working_df[col].replace({'Yes':True,'No':False}).astype('int')
     
    #Adds dummy columns for columns that have multiple categories.
    dummy_df = (pd.get_dummies(working_df[['gender', 'streaming_movies', 'streaming_tv' , 'tech_support', 'multiple_lines', 'online_backup', 'online_security', 'device_protection', 'payment_type', 'internet_service_type', 'contract_type']], drop_first=True))
    
    #Removes columns which provide little information or are duplicate:
    cols_to_drop = ['gender', 'phone_service', 'multiple_lines', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'contract_type', 'payment_type', 'payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id', 'internet_service_type']    
    working_df.drop(columns=cols_to_drop, inplace = True)
    #Adds the two DataFrames back together for output:
    working_df = pd.concat([working_df, dummy_df], axis=1)
    
    return working_df



def prep_titanic(df):
    """
    This function takes a dataframe with the titantic data set
    and areturns a cleaner version.
    """

    #Adds dummy columns for columns that have multiple categories.
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], drop_first=True)

    #Removes columns which provide little information or are duplicate:    
    cols_to_drop = ['passenger_id', 'age', 'embarked', 'class', 'deck', 'sex', 'embark_town']
    df.drop(columns=cols_to_drop, inplace = True)
    
    #Adds the two DataFrames back together for output:
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
    cols_to_drop = ['species_id'] #Never imported the measurements_id column
    df = df.drop(columns=cols_to_drop)
    
    #Shortens the species column name:
    df.rename(columns={'species_name':'species'}, inplace=True)
    
    #Assigns two dummy columns and attach them to the dataframe.
    dummy_df = pd.get_dummies(df[['species']], drop_first=True)

    #Adds the two DataFrames back together for output:    
    df = pd.concat([df, dummy_df], axis=1)
    return df

# 20% test, 80% train_validate
# then of the 80% train_validate: 30% validate, 70% train. 
def train_validate(df, stratify_col = None, random_seed=1969):
    """
    This function takes in a DataFrame and column name for the stratify argument (defualt is None).
    It will split the data into three parts for training, testing and validating.
    """
    #This is logic to set the stratify argument:
    stratify_arg = ''
    if stratify_col != None:
        stratify_arg = df[stratify_col]
    else:
        stratify_arg = None
    
    #This splits the DataFrame into 'train' and 'test':
    train, test = train_test_split(df, train_size=.7, stratify=stratify_arg, random_state = random_seed)
    
    #The length of the stratify column changed and needs to be adjusted:
    if stratify_col != None:
        stratify_arg = train[stratify_col]
        
    #This splits the larger 'train' DataFrame into a smaller 'train' and 'validate' DataFrames:
    train, validate = train_test_split(train, test_size=.5, stratify=stratify_arg, random_state = random_seed)
    return train, validate, test