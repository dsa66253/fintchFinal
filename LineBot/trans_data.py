from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
def normalize_input(input_dataframe):
    input = input_dataframe.head(1)
    input_dataframe = input_dataframe.reset_index()
    input_dataframe = input_dataframe.drop(0)
    input_dataframe = input_dataframe.set_index('CUST_NO')
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(input.values)
    scaled_features_df = pd.DataFrame(scaled_features, columns=input_dataframe.columns)
    final_df = input_dataframe.append(scaled_features_df)

    return final_df

def padding(np_array):
    zero_len = max(0, 100 - np_array.shape[0])
    if zero_len > 0:
        return np.append(
            np_array[:100],
            np.zeros((zero_len, np_array.shape[1])), axis=0)

    return np_array[:100]


def parse_data(input_df):
    final_df = normalize_input(input_df)
    data_raw = final_df.values
    input_data = padding(data_raw)

    return np.array(input_data)