import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from  sklearn import preprocessing


def handle_missing(df,
                   col_name,
                   fill_type='mode', #mode, median, mean, value, or fill
                   value=None, #what to fill with if fill_type = 'value'
                   missing_name=None, #Value to replace with np.nan if not null
                   indicator=True):
    '''
    Function for handling missing data in a column
    Fills missing based on fill_type
    Returns dataframe with filled column and
    another column indicating where missing
    ''' 
    df_col = df[col_name].copy()

    name = df_col.name

    #If data uses missing indicator, replace with np.nan
    if missing_name != None:
        df_col.loc[df_col==missing_name] = np.nan

    #Create new series indicating null
    if indicator == True:
        df_nan = df_col.copy()
        df_nan.loc[df_col.isnull()] = 1
        df_nan.loc[~df_col.isnull()] = 0
        df_nan.name = name + '_missing_ind'

    #Fill in null values based on fill_type
    if fill_type == 'median':
        df_col = df_col.fillna(df_col.median())
    elif fill_type == 'mean':
        df_col = df_col.fillna(df_col.mean())
    elif fill_type == 'mode':
        df_col = df_col.fillna(df_col.loc[~df_col.isnull()].mode()[0])
    elif fill_type == 'value':
        if value == None:
            print('Must choose value if fill_type is value')
            return df
        df_col = df_col.fillna(value)
    elif fill_type == 'fill':
        df_col = (df_col.fillna(method='ffill')
                        .fillna(method='bfill')
                        .fillna(0))

    if indicator == True:
        df_col = pd.concat([df_col,df_nan],axis=1)
    else:
        df_col = df_col.to_frame()    

    df = df.drop(col_name,axis=1)    
    df = df.merge(df_col,left_index=True,right_index=True)

    return df


def multicollinearity_check(df):
    '''
    Function for automatically removing multicollinearity
    Slow for large datasets
    Inspired by:
    stats.stackexchange.com/questions/155028/how-to-systematically-remove-collinear-variables-in-python
    '''

    #Separate out non-numeric variables 
    numeric_list = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df_num = df.select_dtypes(include=numeric_list)

    thresh=5.0
    variables = list(range(df_num.shape[1]))
    drop_list = []
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(df_num.iloc[:, variables].values, ix)
               for ix in range(df_num.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            drop_list += [df_num.iloc[:, variables].columns[maxloc]]
            del variables[maxloc]
            dropped = True

    print('Variables with VIF > 5:')
    print(drop_list)

    return drop_list


def correlation_graph(df,y,filename):
    '''
    Function for printing out PDFs of all 
    variables against the y variable
    '''

    #Separate out non-numeric variables 
    numeric_list = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df = df.select_dtypes(include=numeric_list)

    #y_name = y.name

    #df = pd.concat([df,y],axis=1)  


    for col in df.columns:
        print(col)

        df.plot(kind='scatter',x=col,y=y,alpha=0.1,figsize=(15,10))
        matplotlib.pyplot.savefig(filename+'/'+str(col))
        plt.close()


def variable_matrix(df,cols,filename):
    '''
    Function for creating scatter matrix
    comparing cols against each other.
    Should not run too many columns at a time.
    '''

    df = df[cols]

    #Separate out non-numeric variables 
    numeric_list = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df = df.select_dtypes(include=numeric_list)

    cols = list(df.columns)

    scatter_matrix(df[cols],figsize=(20,12))
    matplotlib.pyplot.savefig(filename)
    plt.close()


def dummy_create(df,cols):
    '''
    Function for generating dummy variables 
    for all passed categorical cols
    '''

    for i in cols:

        df_dummy = pd.get_dummies(df[i])
        #Remove last dummy column for regression
        df_dummy.drop(df_dummy.columns[len(df_dummy.columns)-1],axis=1,inplace=True)
        #Drop original column
        df.drop(i,axis=1,inplace=True)
        df = df.merge(df_dummy,left_index=True,right_index=True)

    return df


def standardize_scale(df,cols):
    '''
    Function for standardizing scale of variables.
    (value - mean) / variance
    Not for nueral networks
    '''

    for i in cols:

        vari = df[i].var()
        mean = df[i].mean()

        df[i] = (df[i] - mean) / vari

    return df


def train_test_split(df,y_col):
    '''
    Function for splitting the train data
    into three parts. 70/20/10
    The 10% to only be used for final verify.
    Also separates out y variable
    '''

    np.random.seed(20)
    random_indicies = np.random.permutation(len(df))

    test_ver_len = int(len(df) * 0.3)
    verify_len = int(len(df) * 0.1)

    train_indicies = random_indicies[test_ver_len:]
    test_indicies = random_indicies[verify_len:test_ver_len]
    verify_indicies = random_indicies[:verify_len]

    df_train = df.iloc[train_indicies].copy()
    df_test = df.iloc[test_indicies].copy()
    df_verify = df.iloc[verify_indicies].copy()

    df_train_y = df_train[y_col].copy()
    df_train = df_train.drop(y_col,axis=1)

    df_test_y = df_test[y_col].copy()
    df_test = df_test.drop(y_col,axis=1)

    df_verify_y = df_verify[y_col].copy()
    df_verify = df_verify.drop(y_col,axis=1)

    return df_train, df_test, df_verify, df_train_y, df_test_y, df_verify_y




