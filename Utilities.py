"""
This file contains utility functions to apply DCC model to generating dynamic scenarios
"""

import numpy as np
import pandas as pd
import datetime
import DCC_class

def partial_to_full(res, columns, index, partial_views, partial_index, overwrite=True, **kwargs):
    """
    Reformate the class method partial_to_full
    :param res: the fitted model class result
    :param columns: the columns name
    :param index: the index which indicates which conditional covariance matrix to use
    :param partial_views: the input in the conditional expectation
    :param partial_index: the indices where the input variables are located in the DataFrame
    :param overwrite: if True, it will contain the whole time series of estimated covariance matrices from which the
    conditional update is taking place. if False, will use the last covariance matrix which corresponds to today's
    prediction
    :return: DataFrame that contains both the input and predicted variable values
    """
    if partial_views.shape[0] != len(partial_index):
        raise ValueError("The size of partial views does not match partial index!")

    partial_views = pd.DataFrame(partial_views.T, columns=columns[partial_index])
    if overwrite:
        temp = res.conditional_cov()
        Historic_cov = temp["Conditional covariance"][index]
        Historic_corr = temp["Conditional correlation"][index]
        Q = temp['Q'][index]
        Overwrite = {"Correlation": Historic_corr, "Covariance": Historic_cov, "Q": Q}
        full_paths = res.partial_to_full(pd.DataFrame(partial_views.iloc[0]).T, overwrite_cov=Overwrite, **kwargs)
        if len(partial_views) > 1:
            for i in range(1, len(partial_views)):
                if index + i >= len(temp["Conditional covariance"]):
                    index_final = len(temp["Conditional covariance"]) - 1
                else:
                    index_final = index + i

                Historic_corr = temp["Conditional correlation"][index_final]
                Historic_cov = temp["Conditional covariance"][index_final]
                Q = temp['Q'][index_final]
                Overwrite = {"Correlation": Historic_corr, "Covariance": Historic_cov, "Q": Q}
                temp_paths = res.partial_to_full(pd.DataFrame(partial_views.iloc[0]).T, overwrite_cov=Overwrite,
                                                 **kwargs)
                full_paths = pd.concat([full_paths, temp_paths])
                full_paths = full_paths.reset_index(drop=True)
    else:
        full_paths = res.partial_to_full(partial_views, **kwargs)

    return full_paths

def find_index(res, columns, begin_date, end_date, index, partial_views, partial_index, criterion="Max", **kwargs):
    """
    This function finds the index that give extreme values given partial outcome, this is for risk management
    purposes, not for forecasting purposes
    :param res: the fitted model class result
    :param columns: the columns name
    :param begin_date: the beginning date for which the time series of estimated covariance matrices are used for
    :param end_date: the end date for which the time series of estimated covariance matrices are used for
    :params index: an array of dates
    :params partial_views: np array which contains the paths of input variables
    :params partial_index: where the input variables are located in the DataFrame
    return: dictionary contains the model result, the date for which the conditional mean of interest variables are
    largest in magnitude and the values
    """
    if index[0] < begin_date:
        start = index.get_loc(begin_date)
    else:
        start = 0

    if index[-1] > end_date:
        end = index.get_loc(end_date)
    else:
        end = len(index) - 1

    indices = np.arange(start, end + 1)
    input_partial_views = partial_views[:, 0].reshape(partial_views.shape[0], 1)
    temp = pd.DataFrame(columns=columns, index=indices)
    # Given the start and end date, find the maximum or minimum predicted values
    for i in range(np.size(indices)):
        temp.iloc[i] = partial_to_full(res, columns, indices[i], input_partial_views, partial_index)

    full_index = np.arange(len(temp.columns))
    rest_index = list(set(full_index) - set(partial_index))
    temp = temp[columns[rest_index]]
    to_compare = temp.mean(axis=1)
    if criterion == "Max":
        id = to_compare.idxmax()
    elif criterion == "Min":
        id = to_compare.idxmin()

    return {"output": pd.DataFrame(temp.loc[id]).T, 'index': pd.DataFrame([[id, index[id]]],
                                                                          columns=['Location', 'Date'])}

def get_predictions(data, partial_view, partial_name, start_params=None, index=None, criterion=None, start=None,
                    end=None, **kwargs):
    """
    Given the data, it fits a Dynamic Conditional Correlation model, then perform conditional mean prediction given the
    input variables. If start and end are specified, the search engine will use all the estimated covariance matrices
    during this period and select the most extreme case for stress testing purposes. Otherwise, it will use today's
    covariance matrix to predict the outcome of the next period
    :param data: DataFrame contains training data
    :param partial_view: [[]], np array which contains the values for input variables conditioned upon. Can take
    multiple arrays of length >= 1 for multiperiod forecast
    :param partial_name: the names of the input variables
    :param start_params: initial guess of parameters
    :param index: If given, it will just use the covariance matrix of this time point to predict
    :param criterion: whether it's min or max, depending on the sign
    :param start: start date from which the search engine takes place
    :param end:
    :param kwargs:
    :return: DataFrame of both input and predicted values
    """
    model = DCC_class.DCC(data, **kwargs)
    res = model.fit(start_params=start_params, **kwargs)

    partial_index = [data.columns.get_loc(col) for col in partial_name]
    Add = np.array(partial_view)
    columns = data.columns
    if index is not None:
        full_paths = {"output": partial_to_full(res, columns, index, Add, partial_index, **kwargs),
                      "index": pd.DataFrame([[index, data.index[index]]], columns=['Location', 'Date']),
                      "model": res}
    elif criterion is not None and start is not None and end is not None:
        full_paths = find_index(res, columns, start, end, data.index, Add, partial_index, criterion=criterion, **kwargs)
        full_paths['model'] = res
    else:
        index = len(data) - 1
        full_paths = {"output": partial_to_full(res, columns, index, Add, partial_index, **kwargs),
                      "index": pd.DataFrame([[index, data.index[index]]], columns=['Location', 'Date']),
                      'mode;': res}

    return full_paths

def infer_values(data, partial_view, partial_name, inferred_name, **kwargs):
    """
    Wrapper function to make predictions based on input variables
    :param data: DataFrame contains training data
    :param partial_view: [[]], np array which contains the values for input variables conditioned upon. Can take
    multiple arrays of length >= 1 for multiperiod forecast
    :param partial_name: the names of the input variables
    :param inferred_name: the names of the inferred variables
    :param kwargs: can specify start and end dates, or index, see above function for details
    :return:
    """
    if not set(inferred_name).issubset(data.columns):
        raise Exception("Must provide data for the variables that you want to infer")

    Add = np.array(partial_view)
    if set(inferred_name).issubset(partial_name):
        partial_view = pd.DataFrame(Add.T, columns=partial_name)
        result = {'output': partial_view, 'index': pd.DataFrame(columns=['Location', 'Date'])}
    else:
        test = pd.DataFrame(data[partial_name])
        temp = list(set(inferred_name) - set(partial_name))
        test = pd.concat([test, data[temp]], axis=1)
        test = test.dropna()
        result = get_predictions(test, partial_view, partial_name, **kwargs)

    return result

def SPX_UST(data, partial_view, partial_name, **kwargs):
    """
    An example function to use DCC to infer US treasury yield using S&P 500, which is typically not possible using
    linear regression since the correlation is close to 0. Users can define their own customer function which then calls
    the infer_values function to predict
    """
    temp_view = []
    temp = []
    cols = ['SPX', 'UST']
    if set(partial_name) & set(cols):
        temp = list(set(partial_name) & set(cols))
        temp_index = [partial_name.index(i) for i in temp]
        temp_view = [partial_view[i] for i in temp_index]

    inferred_names = list(set(cols) - set(temp))
    return infer_values(data, temp_view, temp, inferred_names, **kwargs)

