import argparse
import os
import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
from pca_classifier.utils.configs import configs
import os
import pickle
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser(description='organizer')
parser.add_argument(
    '--input_txn_folder',
    default='initial_data/txn/',
    type=str,
    help='where txn locate'
)
parser.add_argument(
    '--input_cust_folder',
    default='initial_data/cust_info/',
    type=str,
    help='where cust_info locate'
)
parser.add_argument(
    '--input_stock_folder',
    default='initial_data/stock_info/',
    type=str,
    help='where stock_info locate'
)
parser.add_argument(
    '--output_data_folder',
    default='data/',
    type=str,
    help='where model data locate'
)

args = parser.parse_args()

class DataOrganizer(object):
    def __init__(self):
        pass
    def _folder_iterator(self, folder_path) -> pd.DataFrame:
        assert os.path.exists(folder_path)
        for (dirpath, dirnames, filenames) in os.walk(folder_path):
            for filename in filenames:
                print('read:',os.path.join(dirpath, filename))
                df = pd.read_csv(os.path.join(dirpath, filename), index_col = 0)
                yield df

    def read_file(self, file_path) -> pd.DataFrame:
        assert os.path.exists(file_path)
        result = None
        for df in self._folder_iterator(file_path):
            if result is None:
                result = df
            else:
                result = result.append(df)
        result = result[pd.notnull(result.index)].copy()
        return result.sort_index()
    
    def _normalize_drop_stock(self, stock_df):
        # scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 100))
        scaler = StandardScaler()
        noindex_df = stock_df.reset_index()
        triple_idx_df = noindex_df.set_index(['DATE_RANK', 'STOCK_NO'])
        triple_idx_df.drop(['CAPITAL_TYPE', 'ALPHA', 'BETA_21D', 'BETA_65D', 'BETA_250D'],1,inplace=True)
        triple_idx_df = triple_idx_df.dropna()
        scaled_features = scaler.fit_transform(triple_idx_df.values)
        scaled_features_df = pd.DataFrame(scaled_features, index=triple_idx_df.index, columns=triple_idx_df.columns)
        return scaled_features_df
    
    def _normalize_drop_txn(self, txn_df):
        # min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 100))
        scaler = StandardScaler()
        noindex_df = txn_df.reset_index().copy()
        triple_idx_df = noindex_df.set_index(['DATE_RANK', 'CUST_NO', 'STOCK_NO'])
        triple_idx_df.drop(['MARKET_TYPE_CODE', 'ROI'],1,inplace=True)
        triple_idx_df = triple_idx_df.dropna()
        triple_idx_df.BS_CODE = triple_idx_df.BS_CODE.transform(lambda x: 0 if x == 'B' else 1)
        triple_idx_df.COMMISION_TYPE_CODE = triple_idx_df.COMMISION_TYPE_CODE.transform(lambda x: configs.COMMISION_TYPE_CODE[x])
        scaled_features = scaler.fit_transform(triple_idx_df.values)
        scaled_features_df = pd.DataFrame(scaled_features, index=triple_idx_df.index, columns=triple_idx_df.columns)
        # scaled_features_df.BS_CODE = scaled_features_df.BS_CODE.transform(lambda x: 0 if x == 0 else 1)
        return scaled_features_df
    
    def _normalize_drop_cust(self, cust_df):
        # min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 100))
        scaler = StandardScaler()
        noindex_df = cust_df.reset_index().copy()
        noindex_df.BREACH_DATE_RANK = noindex_df.BREACH_DATE_RANK.transform(lambda x: 0 if np.isnan(x) else x)
        triple_idx_df = noindex_df.set_index(['CUST_NO'])
        triple_idx_df.drop(['SOURCE_CODE', 'BREACH_IND', 'BREACH_RANK', 'INVESTMENT_TXN_CODE', 'NONTXN_COUNT'],1,inplace=True)
        triple_idx_df = triple_idx_df.dropna()
        back_off = triple_idx_df.copy()
        scaled_features = scaler.fit_transform(triple_idx_df.values)
        scaled_features_df = pd.DataFrame(scaled_features, index=triple_idx_df.index, columns=triple_idx_df.columns)
        scaled_features_df.BREACH_DATE_RANK = back_off.BREACH_DATE_RANK
        return scaled_features_df

    def organize_data(self):
        """[read from csv and process df]
        """
        print('read txn')
        txn_df = self.read_file(args.input_txn_folder)
        txn_df = self._normalize_drop_txn(txn_df.copy())

        print('read cust')
        cust_df = self.read_file(args.input_cust_folder)
        cust_df = self._normalize_drop_cust(cust_df.copy())

        print('read stock')
        stock_df = self.read_file(args.input_stock_folder)
        stock_df = self._normalize_drop_stock(stock_df.copy())

        cust_txn_df = pd.merge(txn_df.reset_index(), cust_df.reset_index(), on=['CUST_NO'])
        result_df = pd.merge(cust_txn_df, stock_df.reset_index(), on=['DATE_RANK', 'STOCK_NO'])

        sorted_merge_df = result_df.sort_values(['CUST_NO','DATE_RANK'], ascending=[True, True])
        cust_idx_df = sorted_merge_df.set_index(['CUST_NO'])
        cust_idx_df.drop(['STOCK_NO'],1,inplace=True)

        breach = cust_idx_df[cust_idx_df.BREACH_DATE_RANK > 0].copy()
        no_breach = cust_idx_df[cust_idx_df.BREACH_DATE_RANK == 0].copy()
        breach_cust = cust_idx_df[cust_idx_df.BREACH_DATE_RANK > 0].index.drop_duplicates().copy()
        no_breach_cust = cust_idx_df[cust_idx_df.BREACH_DATE_RANK == 0].index.drop_duplicates().copy()

        print('drop unused data')
        breach.drop(['BREACH_DATE_RANK', 'DATE_RANK'],1,inplace=True)
        no_breach.drop(['BREACH_DATE_RANK', 'DATE_RANK'],1,inplace=True)
        print('organize data end')
        return breach, no_breach, pd.array(breach_cust), pd.array(no_breach_cust)

def check_all_equal(check_list):
    if check_list.count(check_list[0]) == len(check_list):
        return True
    return False

def padding(np_array):
    zero_len = max(0, configs.max_seq_len - np_array.shape[0])
    if zero_len > 0:
        return np.append(
                np_array[:configs.max_seq_len],
                np.zeros((zero_len, np_array.shape[1])), axis=0)

    return np_array[:configs.max_seq_len]

def write_file(pkl_path, x, y):
    print(f'writing {pkl_path} ...')
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'x': x,
            'y': y,
        }, f)

def parse_data(data_df, cust_list):
    data = []
    data_df1 = data_df.reset_index()
    data_df2 = data_df1.set_index('CUST_NO')
    for i, cust in tqdm(enumerate(cust_list)):
        cust_df = data_df2.loc[cust]
        if len(cust_df.shape) < 2:
            continue
        cust_data_raw = cust_df.values
        x = padding(cust_data_raw)
        data.append(x)
    return np.array(data)

def split_data(breach, no_breach):
    print('turn df to model data')
    breach_size = breach.shape[0]
    no_breach_size = int(np.round((1-configs.no_breach_delete_percentage/100)*no_breach.shape[0]))
    no_breach = no_breach[:no_breach_size]

    valid_breach_size = int(np.round(configs.valid_set_size_percentage/100*breach_size))
    test_breach_size = int(np.round(configs.test_set_size_percentage/100*breach_size))
    train_breach_size = breach_size - (valid_breach_size + test_breach_size)

    valid_no_breach_size = int(np.round(configs.valid_set_size_percentage/100*no_breach_size))
    test_no_breach_size = int(np.round(configs.test_set_size_percentage/100*no_breach_size))
    train_no_breach_size = no_breach_size - (valid_no_breach_size + test_no_breach_size)
    
    x_train = np.append(
            breach[:train_breach_size,:,:],
            no_breach[:train_no_breach_size,:,:], axis=0)
    y_train = np.append(
            np.ones((train_breach_size,), dtype=int),
            np.zeros((train_no_breach_size,), dtype=int), axis=0)

    x_valid = np.append(
            breach[train_breach_size:train_breach_size+valid_breach_size,:,:],
            no_breach[train_no_breach_size:train_no_breach_size+valid_no_breach_size,:,:], axis=0)
    y_valid = np.append(
            np.ones((valid_breach_size,), dtype=int),
            np.zeros((valid_no_breach_size,), dtype=int), axis=0)

    x_test = np.append(
            breach[train_breach_size+valid_breach_size:,:,:],
            no_breach[train_no_breach_size+valid_no_breach_size:,:,:], axis=0)
    y_test = np.append(
            np.ones((test_breach_size,), dtype=int),
            np.zeros((test_no_breach_size,), dtype=int), axis=0)
    
    
    print('============ rate of 0 ===========')
    print('train: {}, valid: {}, test: {}'.format(
        train_no_breach_size/(train_breach_size+train_no_breach_size),
        valid_no_breach_size/(valid_breach_size+valid_no_breach_size),
        test_no_breach_size/(test_breach_size+test_no_breach_size)
        ))
    
    return [x_train, y_train, x_valid, y_valid, x_test, y_test]



def main():
    data_organizer = DataOrganizer()
    breach_result_df, no_breach_result_df, breach_cust, no_breach_cust = data_organizer.organize_data()

    breach_data = parse_data(breach_result_df, breach_cust)
    no_breach_data = parse_data(no_breach_result_df, no_breach_cust)

    x_train, y_train, x_valid, y_valid, x_test, y_test = split_data(breach_data, no_breach_data)
    print(f'train: {x_train.shape[0]}\nvalid: {x_valid.shape[0]}\ntest: {x_test.shape[0]}')
    write_file(os.path.join(args.output_data_folder, 'train.pkl'), x_train, y_train)
    write_file(os.path.join(args.output_data_folder, 'valid.pkl'), x_valid, y_valid)
    write_file(os.path.join(args.output_data_folder, 'test.pkl'), x_test, y_test)
    print('end')
if __name__ == '__main__':
    main()