"""Label encode and One-Hot encode dataframes."""

from sklearn.preprocessing import LabelEncoder
from pandas import get_dummies
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import merge
import numpy as np


# Auto encodes any dataframe column of type category or object.
def dummy_encode(df):
    """
    Encode any dataframe column of type category or object.

    ---
    df: pandas dataframe
    """
    columnsToEncode = list(
        df.select_dtypes(include=['category', 'object']))

    df1 = df.copy()

    for feature in columnsToEncode:
        le = LabelEncoder()
        try:
            df1[feature] = le.fit_transform(df[feature].astype(str))
        except Exception as e:
            print(e)
            print('Error encoding ' + feature)
    return df1


#######

def le_encode_column(column):
    """Label-encode pandas DataFrame column or Series"""
    le = LabelEncoder()
    le_column = le.fit_transform(column)
    if isinstance(column, DataFrame):
        le_column = Series(le_column).to_frame(column.name)
        # equivalent to:
        # le_column = DataFrame(le_column, columns=[df_column.name])
    elif isinstance(column, Series):
        le_column = Series(le_column)
    else:
        raise TypeError(
            "'column' should be of type pandas.DataFrame/Series")

    return le_column


def encode_df_column(df_column):
    """Encode pandas dataframe column 'df_column'."""
    print("column name: ", df_column.name)

    try:
        enc_df_column = get_dummies(
            df_column, prefix=df_column.name, prefix_sep='_')
    except MemoryError as me:
        print(me)
        print("MemoryError! Column: " + df_column.name)
        print("Proceed to label-encoding")
        enc_df_column = le_encode_column(df_column)
    except KeyError as ke:
        print(ke)
        print("KeyError! Column: " + df_column.name)
    except ValueError as ve:
        print(ve)
        print("ValueError! Column: " + df_column.name)
    except Exception:
        print('Error encoding feature ' + df_column.name)

    # print("column head", enc_df_column.head(1))

    assert (len(enc_df_column) == len(df_column)), \
    "Ouch! Encoded column's different length than original's!"

    return enc_df_column


def get_date_features(df, freqs=None, to_period=False):
    """
    Get dates objects from dataframe.

    ---
    df: pandas Dataframe
    freqs: frequencies of datetime objects
    """
    new_df = DataFrame()

    if freqs is None:
        freqs = ['Year', 'Month', 'Day', 'hour', 'min']
    else:
        for f in freqs:
            if f not in ('Year', 'Month', 'Day', 'hour', 'min'):
                raise ValueError(
                    "'%s' is not a valid value. Valid values are:"
                    "['Year', 'Month', 'Day', 'hour', 'min']"
                    % f)

    # computationally expensive: lots of columns!
    
    if to_period:
        for feature in df.columns:
            if df[feature].dtype == 'datetime64[ns]':
                for f in freqs:
                    try:
                        new_df[f] = \
                        df[feature].dt.to_period(f[0] if f != 'min' else f)
                    except KeyError as ke:
                        print(ke)
                        print("KeyError! Column: " + feature)
                    except ValueError as ve:
                        print(ve)
                        print("ValueError! Column: " + feature)
                    except Exception as e:
                        raise e
            else:
                new_df[feature] = df[feature]
    elif not to_period:
        for feature in df.columns:
            if df[feature].dtype == 'datetime64[ns]':
                for f in freqs:
                    try:
                        if f == 'Year':
                            new_df[f] = df[feature].dt.year
                        elif f == 'Month':
                            new_df[f] = df[feature].dt.month
                        elif f == 'Day':
                            new_df[f] = df[feature].dt.day
                        elif f == 'hour':
                            new_df[f] = df[feature].dt.hour
                        else:
                            new_df[f] = df[feature].dt.minute
                    except KeyError as ke:
                        print(ke)
                        print("KeyError! Column: " + feature)
                    except ValueError as ve:
                        print(ve)
                        print("ValueError! Column: " + feature)
                    except Exception as e:
                        raise e
            else:
                new_df[feature] = df[feature]
    else:
        raise ValueError(
            "%s is not a valid value for arg. 'to_period'"
            "Valid values for are ['True', 'False']" % to_period)

    assert (len(new_df.index) == len(df.index)), \
    "Ouch, encoded dataframe's different length than original's!"

    return new_df


def get_dummies_or_label_encode(df, target=None, upper=100):
    """
    Label or One-Hot encode columns.

    ---
    df: pandas Dataframe
    """
    if target is None:
        new_df = DataFrame()
        cols = df.columns
    else:
        new_df = df[target].to_frame(name=target)
        cols = df.drop([target], axis=1).columns

    print("columns:\n", cols)
    print()

    # column_types = ('str', 'object', 'category', 'datetime64')

    columns_to_encode = list(
        df.select_dtypes(include=['category', 'object']))

    for feature in cols:
        col = df[feature]
        # if df[feature].dtype.name in column_types:
        if feature in columns_to_encode:
            try:
                if new_df.empty:
                    # print("new_df is empty")
                    if len(col.unique()) <= upper:
                        new_df = encode_df_column(col)
                    else:
                        new_df = le_encode_column(col)
                else:
                    new_df = merge(
                        new_df, 
                        encode_df_column(col) if len(col.unique()) <= upper \
                        else le_encode_column(col).to_frame(feature),
                        left_index=True, right_index=True)
            except KeyError as ke:
                print(ke)
                print("KeyError! Column: " + feature)
            except ValueError as ve:
                print(ve)
                print("ValueError! Column: " + feature)
            except Exception as e:
                raise e
        else:
            if new_df.empty:
                # print("new_df is empty")
                new_df = col.to_frame(feature)
            else:
                new_df = merge(
                    new_df, col.to_frame(feature),
                    left_index=True, right_index=True)

    assert (len(new_df.index) == len(df.index)), \
    "Ouch, encoded dataframe's different length than original's!"

    print("new_df's head:\n", new_df.head(1))

    return new_df

