# This is adopted from the awesome AutoVIML project
# https://github.com/AutoViML/Auto_ViML/blob/master/autoviml/Auto_ViML.py

import copy

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import SelectKBest
from collections import defaultdict

def return_dictionary_list(lst_of_tuples):
    """ Returns a dictionary of lists if you send in a list of Tuples"""
    orDict = defaultdict(list)
    # iterating over list of tuples
    for key, val in lst_of_tuples:
        orDict[key].append(val)
    return orDict

def left_subtract(l1,l2):
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst

def count_freq_in_list(lst):
    """
    This counts the frequency of items in a list but MAINTAINS the order of appearance of items.
    This order is very important when you are doing certain functions. Hence this function!
    """
    temp=np.unique(lst)
    result = []
    for i in temp:
        result.append((i,lst.count(i)))
    return result

# from https://github.com/AutoViML/Auto_ViML/blob/master/autoviml/Auto_ViML.py#L3801
def remove_variables_using_fast_correlation(df, numvars, modeltype, target,
                                corr_limit = 0.70,verbose=0):
    """
    #### THIS METHOD IS KNOWN AS THE SULA METHOD in HONOR OF my mother SULOCHANA SESHADRI #######
    S U L A : Simple Uncorrelated Linear Algorithm will get a set of uncorrelated vars easily.
    SULA is a highly efficient method that removes variables that are highly correlated but
    breaks the logjam that occurs when deciding which of the 2 correlated variables to remove.
    SULA uses the MIS (mutual_info_score) to decide to keep the higher scored variable.
    This method enables an extremely fast and highly effective method that keeps the variables
    you want while discarding lesser highr correlated variables in less than a minute, even on a laptop.
    You need to send in a dataframe with a list of numeric variables and define a threshold - that's all.
    This threshold will define the "high Correlation" mark - you can set it as anything over + or - 0.70.
    You can change this limit. If two variables have absolute correlation higher than limit, they
    will be red-lined, and using a series of knockout rounds, one of them will get knocked out:
    To decide which variables to keep in knockout rounds, mutual information score is used.
    MIS returns a ranked list of correlated variables: when we select one to keep, we knock out those
    that were correlated to it. This way we knock out correlated variables from each round.
    In the end, SULA gives us the least correlated variables that have the best mutual information score!
    ##############  YOU MUST INCLUDE THE ABOVE MESSAGE IF YOU COPY THIS CODE IN YOUR LIBRARY #####
    """
    df = copy.deepcopy(df)
    print('Removing highly correlated variables using SULA method among (%d) numeric variables' %len(numvars))
    correlation_dataframe = df[numvars].corr().astype(np.float16)
    a = correlation_dataframe.values
    col_index = correlation_dataframe.columns.tolist()
    index_triupper = list(zip(np.triu_indices_from(a,k=1)[0],np.triu_indices_from(a,k=1)[1]))
    high_corr_index_list = [x for x in np.argwhere(abs(a[np.triu_indices(len(a), k = 1)])>=corr_limit)]
    low_corr_index_list =  [x for x in np.argwhere(abs(a[np.triu_indices(len(a), k = 1)])<corr_limit)]
    tuple_list = [y for y in [index_triupper[x[0]] for x in high_corr_index_list]]
    correlated_pair = [(col_index[tuple[0]],col_index[tuple[1]]) for tuple in tuple_list]
    corr_pair_dict = dict(return_dictionary_list(correlated_pair))
    keys_in_dict = list(corr_pair_dict.keys())
    reverse_correlated_pair = [(y,x) for (x,y) in correlated_pair]
    reverse_corr_pair_dict = dict(return_dictionary_list(reverse_correlated_pair))
    for key, val in reverse_corr_pair_dict.items():
        if key in keys_in_dict:
            if len(key) > 1:
                corr_pair_dict[key] += val
        else:
            corr_pair_dict[key] = val
    flat_corr_pair_list = [item for sublist in correlated_pair for item in sublist]
    #### You can make it a dictionary or a tuple of lists. We have chosen the latter here to keep order intact.
    corr_pair_count_dict = count_freq_in_list(flat_corr_pair_list)
    corr_list = [k for (k,v) in corr_pair_count_dict]
    ###### This is for ordering the variables in the highest to lowest importance to target ###
    if len(corr_list) == 0:
        final_list = list(correlation_dataframe)
        print('    No numeric vars removed since none have high correlation with each other in this data...')
    else:
        max_feats = len(corr_list)
        if modeltype == 'Regression':
            sel_function = mutual_info_regression
            fs = SelectKBest(score_func=sel_function, k=max_feats)
        else:
            sel_function = mutual_info_classif
            fs = SelectKBest(score_func=sel_function, k=max_feats)
        fs.fit(df[corr_list].astype(np.float16), df[target])
        mutual_info = dict(zip(corr_list,fs.scores_))
        #### The first variable in list has the highest correlation to the target variable ###
        sorted_by_mutual_info =[key for (key,val) in sorted(mutual_info.items(), key=lambda kv: kv[1],reverse=True)]
        #####   Now we select the final list of correlated variables ###########
        selected_corr_list = []
        #### select each variable by the highest mutual info and see what vars are correlated to it
        for each_corr_name in sorted_by_mutual_info:
            ### add the selected var to the selected_corr_list
            selected_corr_list.append(each_corr_name)
            for each_remove in corr_pair_dict[each_corr_name]:
                #### Now remove each variable that is highly correlated to the selected variable
                if each_remove in sorted_by_mutual_info:
                    sorted_by_mutual_info.remove(each_remove)
        ##### Now we combine the uncorrelated list to the selected correlated list above
        rem_col_list = left_subtract(list(correlation_dataframe),corr_list)
        final_list = rem_col_list + selected_corr_list
        removed_cols = left_subtract(numvars, final_list)
        if len(removed_cols) > 0:
            print('    Removing (%d) highly correlated variables:' %(len(removed_cols)))
            if len(removed_cols) <= 30:
                print('    %s' %removed_cols)
            if len(final_list) <= 30:
                print('    Following (%d) vars selected: %s' %(len(final_list),final_list))
    return final_list