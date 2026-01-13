# coding: utf-8

"""
Data pre-processing
##########################
"""
from logging import getLogger
from collections import Counter
import os
import pandas as pd
import numpy as np
import torch
from src_new_item.utils.data_utils import (ImageResize, ImagePad, image_to_tensor, load_decompress_img_from_lmdb_value)
import lmdb


class RecDataset(object):
    def __init__(self, config, df=None):
        self.config = config
        self.logger = getLogger()

        # data path & files
        self.dataset_name = config['dataset']
        self.dataset_path = os.path.abspath(config['data_path']+self.dataset_name)

        # dataframe
        self.uid_field = self.config['USER_ID_FIELD']
        self.iid_field = self.config['ITEM_ID_FIELD']
        self.splitting_label = self.config['inter_splitting_label']

        if df is not None:
            self.df = df
            return
        # if all files exists
        check_file_list = [self.config['inter_file_name']]
        for i in check_file_list:
            print(self.dataset_path)
            print(i)
            file_path = os.path.join(self.dataset_path, i)
            if not os.path.isfile(file_path):
                raise ValueError('File {} not exist'.format(file_path))

        # load rating file from data path?
        self.load_inter_graph(config['inter_file_name'])
        self.item_num = int(max(self.df[self.iid_field].values)) + 1
        self.user_num = int(max(self.df[self.uid_field].values)) + 1

    def load_inter_graph(self, file_name):
        inter_file = os.path.join(self.dataset_path, file_name)
        cols = [self.uid_field, self.iid_field, self.splitting_label]
        self.df = pd.read_csv(inter_file, usecols=cols, sep=self.config['field_separator'])
        if not self.df.columns.isin(cols).all():
            raise ValueError('File {} lost some required columns.'.format(inter_file))

    # dataset.py  — RecDataset 内新增

    def get_user_sequences(self, max_len: int = 50):
        """
        返回 {user_id: [i1, i2, ..., ik]} 的字典，每个用户按“交互出现顺序”截取最近 max_len 个 item。
        如果你的 .inter 文件没有时间戳，就用 df 当前顺序近似时间顺序。
        """
        uid = self.uid_field
        iid = self.iid_field
        seqs = {}
        # 按出现顺序分组聚合
        for u, df_u in self.df.groupby(uid):
            items = df_u[iid].tolist()
            if not items:
                continue
            if len(items) > max_len:
                items = items[-max_len:]
            seqs[int(u)] = items
        return seqs

    def get_interaction_history(self):
        """
        返回列表 [(u, i, ts, rating), ...]。
        若无时间戳列或某些行为 NaN，则用行索引充当 ts；rating 对隐式反馈统一为 1.0。
        """
        uid, iid = self.uid_field, self.iid_field
        df = self.df.reset_index(drop=True)

        time_col = self._detect_time_col()
        ts_series = None
        if time_col and time_col in df.columns:
            ts_series = pd.to_numeric(df[time_col], errors="coerce")

        history = []
        for idx, row in df.iterrows():
            u = int(row[uid])
            i = int(row[iid])
            ts = float(ts_series.iloc[idx]) if ts_series is not None and not math.isnan(ts_series.iloc[idx]) else float(
                idx)
            rating = 1.0
            history.append((u, i, ts, rating))
        return history

    def split(self):
        dfs = []
        
        # For NewItem Test
        new_items = np.load("../data/" + self.dataset_name +'/new_items.npy')
        total_df = pd.read_csv(f"../data/{self.dataset_name}/{self.dataset_name}.inter", sep = '\t')
        df_0 = total_df[total_df.x_label == 0]
        df_2 = total_df[total_df.x_label == 2]
        new_df = pd.concat([df_2, df_0[df_0['itemID'].isin(new_items)].iloc[:, [0, 1]]])

        # splitting into training/validation/test
        for i in range(3):
            temp_df = self.df[self.df[self.splitting_label] == i].copy()
            temp_df.drop(self.splitting_label, inplace=True, axis=1)        # no use again
            dfs.append(temp_df)
        if self.config['filter_out_cod_start_users']:
            # filtering out new users in val/test sets
            train_u = set(dfs[0][self.uid_field].values)
            for i in [1, 2]:
                dropped_inter = pd.Series(True, index=dfs[i].index)
                dropped_inter ^= dfs[i][self.uid_field].isin(train_u)
                dfs[i].drop(dfs[i].index[dropped_inter], inplace=True)
            
            dropped_inter = pd.Series(True, index=new_df.index)
            dropped_inter ^= new_df[self.uid_field].isin(train_u)
            new_df.drop(new_df.index[dropped_inter], inplace=True)
            

        # wrap as RecDataset
        full_ds = [self.copy(_) for _ in dfs]

        new_df = self.copy(new_df)

        return full_ds, new_df

    def copy(self, new_df):
        """Given a new interaction feature, return a new :class:`Dataset` object,
                whose interaction feature is updated with ``new_df``, and all the other attributes the same.

                Args:
                    new_df (pandas.DataFrame): The new interaction feature need to be updated.

                Returns:
                    :class:`~Dataset`: the new :class:`~Dataset` object, whose interaction feature has been updated.
                """
        nxt = RecDataset(self.config, new_df)

        nxt.item_num = self.item_num
        nxt.user_num = self.user_num
        return nxt

    def get_user_num(self):
        return self.user_num

    def get_item_num(self):
        return self.item_num

    def shuffle(self):
        """Shuffle the interaction records inplace.
        """
        self.df = self.df.sample(frac=1, replace=False).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Series result
        return self.df.iloc[idx]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        info = [self.dataset_name]
        self.inter_num = len(self.df)
        uni_u = pd.unique(self.df[self.uid_field])
        uni_i = pd.unique(self.df[self.iid_field])
        tmp_user_num, tmp_item_num = 0, 0
        if self.uid_field:
            tmp_user_num = len(uni_u)
            avg_actions_of_users = self.inter_num/tmp_user_num
            info.extend(['The number of users: {}'.format(tmp_user_num),
                         'Average actions of users: {}'.format(avg_actions_of_users)])
        if self.iid_field:
            tmp_item_num = len(uni_i)
            avg_actions_of_items = self.inter_num/tmp_item_num
            info.extend(['The number of items: {}'.format(tmp_item_num),
                         'Average actions of items: {}'.format(avg_actions_of_items)])
        info.append('The number of inters: {}'.format(self.inter_num))
        if self.uid_field and self.iid_field:
            sparsity = 1 - self.inter_num / tmp_user_num / tmp_item_num
            info.append('The sparsity of the dataset: {}%'.format(sparsity * 100))
        return '\n'.join(info)
