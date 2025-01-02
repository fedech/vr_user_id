import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np

# Dictionary mapping participant IDs to numerical values
IDS = {'group1_order1_user0': 1, 'group1_order1_user1': 2, 'group1_order1_user10': 3, 'group1_order1_user11': 4, 'group1_order1_user12': 5, 
       'group1_order1_user13': 6, 'group1_order1_user14': 7, 'group1_order1_user2': 8, 'group1_order1_user3': 9, 'group1_order1_user4': 10,
       'group1_order1_user5': 11, 'group1_order1_user6': 12, 'group1_order1_user7': 13, 'group1_order1_user8': 14, 'group1_order1_user9': 15, 
       'group1_order2_user0': 16, 'group1_order2_user1': 17, 'group1_order2_user10': 18, 'group1_order2_user11': 19, 'group1_order2_user12': 20, 
       'group1_order2_user13': 21, 'group1_order2_user14': 22, 'group1_order2_user2': 23, 'group1_order2_user3': 24, 'group1_order2_user4': 25, 
       'group1_order2_user5': 26, 'group1_order2_user6': 27, 'group1_order2_user7': 28, 'group1_order2_user8': 29, 'group1_order2_user9': 30, 
       'group2_order1_user0': 31, 'group2_order1_user1': 32, 'group2_order1_user10': 33, 'group2_order1_user11': 34, 'group2_order1_user12': 35, 
       'group2_order1_user13': 36, 'group2_order1_user14': 37, 'group2_order1_user2': 38, 'group2_order1_user3': 39, 'group2_order1_user4': 40, 
       'group2_order1_user5': 41, 'group2_order1_user6': 42, 'group2_order1_user7': 43, 'group2_order1_user8': 44, 'group2_order1_user9': 45, 
       'group2_order2_user0': 46, 'group2_order2_user1': 47, 'group2_order2_user10': 48, 'group2_order2_user11': 49, 'group2_order2_user12': 50, 
       'group2_order2_user13': 51, 'group2_order2_user14': 52, 'group2_order2_user2': 53, 'group2_order2_user3': 54, 'group2_order2_user4': 55, 
       'group2_order2_user5': 56, 'group2_order2_user6': 57, 'group2_order2_user7': 58, 'group2_order2_user8': 59, 'group2_order2_user9': 60}

class Preprocessing:
    """
    A class to perform data preprocessing steps including fixing feature naming, handling missing values,
    feature scaling, label encoding, and matching columns between datasets.

    Attributes:
        None
    """

    def __init__(self):
        pass

    @staticmethod
    def pipeline(df_fast, df_slow):
        """
        Perform data preprocessing pipeline including feature naming fixing, dropping non-varying variables,
        scaling, encoding, and label preparation.

        Args:
            df_fast (DataFrame): DataFrame containing fast data.
            df_slow (DataFrame): DataFrame containing slow data.

        Returns:
            DataFrame, DataFrame: Processed DataFrames for fast and slow data.
        """
        df_fast = Preprocessing.fix_feature_naming(df_fast)
        df_slow = Preprocessing.fix_feature_naming(df_slow)

        df_fast = Preprocessing.drop_non_varying_variables(df_fast)
        df_slow = Preprocessing.drop_non_varying_variables(df_slow)

        df_fast = Preprocessing.match_columns(df_slow, df_fast)

        df_slow, scaler = Preprocessing.scaling(df_slow)
        df_fast, _ = Preprocessing.scaling(df_fast, scaler)

        df_fast = Preprocessing.encoding(df_fast)
        df_slow = Preprocessing.encoding(df_slow)

        df_fast = Preprocessing.match_columns(df_slow, df_fast)

        df_fast = Preprocessing.prepare_label(df_fast)
        df_slow = Preprocessing.prepare_label(df_slow)
        return df_fast, df_slow

    @staticmethod
    def fix_feature_naming(df):
        """
        Fix feature naming by removing leading and trailing spaces and renaming specific columns.

        Args:
            df (DataFrame): Dataframe to fix feature naming.

        Returns:
            DataFrame: Dataframe with fixed feature naming.
        """
        df.columns = df.columns.str.strip()
        if "ID_" in df.columns :
            df.rename(columns={"ID_": "ID", "time_interval_": "time_interval"}, inplace=True)
        return df

    @staticmethod
    def drop_non_varying_variables(df):
        """
        Drop non-varying variables in a dataframe.

        Args:
            df (DataFrame): DataFrame to find non-varying variables.

        Returns:
            DataFrame: DataFrame containing non-varying variables and their variability percentages.
        """
        non_varying_columns = []

        for column in df.columns:
            unique_count = df[column].nunique()
            if unique_count == 1:
                non_varying_columns.append(column)

        df.drop(columns=df[non_varying_columns], inplace=True)
        return df

    @staticmethod
    def missing_columns(dataframe):
        """
        Returns a DataFrame containing missing column names and percent of missing values.

        Args:
            dataframe (DataFrame): DataFrame to check for missing columns.

        Returns:
            DataFrame: DataFrame containing missing column names and their percentage of missing values.
        """
        missing_values = dataframe.isnull().sum().sort_values(ascending=False)
        missing_values_pct = 100 * missing_values / len(dataframe)
        concat_values = pd.concat([missing_values, missing_values / len(dataframe), missing_values_pct.round(1)],
                                  axis=1)
        concat_values.columns = ['Missing Count', 'Missing Count Ratio', 'Missing Count %']
        return concat_values[concat_values.iloc[:, 1] != 0]

    @staticmethod
    def scaling(df, scaler=None):
        """
        Perform feature scaling on numeric columns using MinMaxScaler.

        Args:
            df (DataFrame): DataFrame to perform feature scaling.
            scaler (MinMaxScaler): Scaler to fit or transform.

        Returns:
            DataFrame, MinMaxScaler: Scaled DataFrame and fitted scaler.
        """
        numeric_cols = df.select_dtypes(include=['number']).columns.difference(["time_interval"])
        if not scaler:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df[numeric_cols])
        else:
            scaled_data = scaler.transform(df[numeric_cols])
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols)
        for col in df.columns:
            if col not in numeric_cols:
                scaled_df[col] = df[col].values
        return scaled_df, scaler

    @staticmethod
    def encoding(df):
        """
        Perform label encoding on categorical columns.

        Args:
            df (DataFrame): DataFrame to perform label encoding.

        Returns:
            DataFrame: DataFrame with encoded categorical features.
        """
        le = LabelEncoder()
        for col in df.select_dtypes('object').columns.difference(["ID"]):
            df[col] = le.fit_transform(df[col].astype(str))
        return df

    @staticmethod
    def match_columns(training_set, testing_set):
        """
        Matches column count between training and testing sets.

        Args:
            training_set (DataFrame): DataFrame representing the training set.
            testing_set (DataFrame): DataFrame representing the testing set.

        Returns:
            DataFrame: Testing set with matched columns.
        """
        for column in training_set.columns:
            if column not in testing_set.columns:
                testing_set[column] = 0
        for column in testing_set.columns:
            if column not in training_set.columns:
                testing_set = testing_set.drop(column)
        return testing_set

    @staticmethod
    def prepare_label(df):
        """
        Prepare labels by converting participant IDs to numerical values.

        Args:
            df (DataFrame): DataFrame containing participant IDs.

        Returns:
            DataFrame: DataFrame with prepared labels.
        """
        df['ID'] = df['ID'].apply(lambda x: x.split('/')[4])
        df['ID'] = df['ID'].apply(lambda x: IDS[x] - 1)
        df = df.reindex(sorted(df.columns), axis=1)
        return df