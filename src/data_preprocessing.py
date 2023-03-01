import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler


'''
TAKES:
@ df --> dataframe that needs reindexing
'''
def reindex (df):
    df.reset_index(inplace = True)
    df.drop(['index'],axis=1,inplace=True)
    return df

'''
TAKES:
@ df --> the dataframe that needs splitting
@ sample_size --> the size of data that the training set will contain
@ maj_percent --> the percentage of the majority
@ min_percent --> the percentage of the minority
RETURNS:
$ df_testing --> data frame containg the data for testing
$ df_training --> data frame containg the data for training
'''
def split_df(df, sample_size, maj_percent, min_percent):
    # Splitting into minority, majority
    df_minority = df[df['Pha'] == 1]
    df_majority = df[df['Pha'] == 0]
    df_majority = df_majority.sample(n = 8000)

    # Sampling minority and majority
    majority = (sample_size//100)*maj_percent
    minority = (sample_size//100)*min_percent
    df_majority_subset = df_majority.sample(n = majority)
    df_minority_subset = df_minority.sample(n = minority)
    # To create the training dataset
    df_training = pd.concat([df_minority_subset, df_majority_subset], axis = 0)
    df_testing = pd.concat([df_minority, df_majority], axis = 0)

    reindex(df_training)
    reindex(df_testing)
    
    return df_testing, df_training

"""_summary_
This function contains the code, used in the preprocessing notebook,
that changes the dataframe
"""
def process_data(df):
    df.drop(['name', 'extent', 'rot_per',
         'BV', 'IR', 'spec_T', 
         'G', 'GM', 'UB', 'spec_B', 
         'data_arc', 'albedo', 'diameter',
         'om', 'w', 'ad', 'per_y', 'n_obs_used',
         'per', 'ma'],axis=1,inplace=True)
    df = df[df.pha!= 'nan']
    df.dropna(inplace=True)
    df = reindex(df)
    condition_code = []
    for i in df.condition_code:
        condition_code.append(int(i))
    condition_code_df = pd.DataFrame(data = condition_code, columns = ['Condition_Code'])
    df.drop(['condition_code'],axis=1,inplace=True)
    df = pd.concat([df,condition_code_df],axis=1)
    for i in ['neo','class']:
        oh = OneHotEncoder()
        oh_df = pd.DataFrame(oh.fit_transform(df[[i]]).toarray())
        df.drop([i],axis=1,inplace=True)
        df = pd.concat([df,oh_df],axis=1)
    le = LabelEncoder()
    df['pha'] = le.fit_transform(df['pha'])
    pha = []
    for i in df.pha:
        pha.append(int(i))
    pha_df = pd.DataFrame(data = pha, columns = ['Pha'])
    df.drop(['pha'],axis=1,inplace=True)
    df = pd.concat([df,pha_df],axis=1)
    df.columns = ['a', 'e', 'i', 'q', 'H', 'moid', 'n',
              'Condition_Code', 'True', 'False', 0, 1, 2, 3,
              4, 5, 6, 7, 8, 9, 10, 11, 'Pha']
    df_testing,df_training = split_df(df, 2200, 60, 40)
    X_train = df_training.iloc[:, df_training.columns != 'Pha'].values
    y_train = df_training.iloc[:, df_training.columns == 'Pha'].values
    X_test = df_testing.iloc[:, df.columns != 'Pha'].values
    y_test = df_testing.iloc[:, df.columns == 'Pha'].values
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train, X_train_scaled, X_test, X_test_scaled, y_train, y_test
    
'''
TAKES:
@ model --> The model whose accuracy you want to print
OUT:
$ Prints out the accuracy of the training and testing models
'''
def print_accuracy(model,scaled_model,X_train, X_train_scaled, X_test, X_test_scaled, y_train, y_test):
    accuracy = model.score(X_train,y_train)
    print('Model Training Accuracy: {:.3f}'.format(accuracy))
    accuracy = model.score(X_test,y_test)
    print('Model Testing Accuracy: {:.3f}'.format(accuracy))
    print("---" * 8)
    accuracy = scaled_model.score(X_train_scaled,y_train)
    print('Scaled Model Training Accuracy: {:.3f}'.format(accuracy))
    accuracy = scaled_model.score(X_test_scaled,y_test)
    print('Scaled Model Testing Accuracy: {:.3f}'.format(accuracy))
    
'''
TAKES:
@ search -->  The hyperparameter tuning object that you want to visualise
OUT:
$ Prints out various information about the accuracy of the model at the different stages of the tunnig process
'''
    # Taken from
    # https://scikit-learn.org/0.24/auto_examples/model_selection/plot_grid_search_stats.html
def plot_tuning(search):
    results_df = pd.DataFrame(search.cv_results_)
    results_df = results_df.sort_values(by=['rank_test_score'])
    results_df = (
        results_df
        .set_index(results_df["params"].apply(
            lambda x: "_".join(str(val) for val in x.values()))
        )
        .rename_axis('kernel')
    )
    results_df[
        ['params', 'rank_test_score', 'mean_test_score', 'std_test_score']
    ]
    # create df of model scores ordered by perfomance
    model_scores = results_df.filter(regex=r'split\d*_test_score')
    print(model_scores)
    # plot 30 examples of dependency between cv fold and AUC scores
    fig, ax = plt.subplots()
    sns.lineplot(
        data=model_scores.transpose().iloc[:30],
        dashes=False, palette='Set1', marker='o', alpha=.5, ax=ax
    )
    ax.set_xlabel("CV test fold", size=12, labelpad=10)
    ax.set_ylabel("Model AUC", size=12)
    ax.tick_params(bottom=True, labelbottom=False)
    plt.legend(bbox_to_anchor=(1.1, 1.1), bbox_transform=ax.transAxes)
    plt.show()
    # End
    # https://scikit-learn.org/0.24/auto_examples/model_selection/plot_grid_search_stats.html  