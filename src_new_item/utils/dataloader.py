# coding: utf-8
"""
Wrap dataset into dataloader (patched)
- Adds user short-term sequence support via get_user_sequences()
- Binds get_user_sequences to dataset if it doesn't exist
- Caches per-user sequences and invalidates cache on shuffle/pretrain_setup
"""
import math
import torch
import random
import numpy as np
from logging import getLogger
from scipy.sparse import coo_matrix


class AbstractDataLoader(object):
    """:class:`AbstractDataLoader` is an abstract object which would return a batch of data which is loaded by
    :class:`~recbole.data.interaction.Interaction` when it is iterated.
    And it is also the ancestor of all other dataloader.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        dl_format (InputType, optional): The input type of dataloader. Defaults to
            :obj:`~recbole.utils.enum_type.InputType.POINTWISE`.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.

    Attributes:
        dataset (Dataset): The dataset of this dataloader.
        shuffle (bool): If ``True``, dataloader will shuffle before every epoch.
        real_time (bool): If ``True``, dataloader will do data pre-processing,
            such as neg-sampling and data-augmentation.
        pr (int): Pointer of dataloader.
        step (int): The increment of :attr:`pr` for each batch.
        batch_size (int): The max interaction number for all batch.
    """
    def __init__(self, config, dataset, additional_dataset=None,
                 batch_size=1, neg_sampling=False, shuffle=False):
        self.config = config
        self.logger = getLogger()
        self.dataset = dataset
        self.dataset_bk = self.dataset.copy(self.dataset.df)
        self.additional_dataset = additional_dataset
        self.batch_size = batch_size
        self.step = batch_size
        self.shuffle = shuffle
        self.neg_sampling = neg_sampling
        self.device = config['device']
        print(self.dataset)
        self.sparsity = 1 - self.dataset.inter_num / self.dataset.user_num / self.dataset.item_num
        self.pr = 0
        self.inter_pr = 0

    def pretrain_setup(self):
        """This function can be used to deal with some problems after essential args are initialized,
        such as the batch-size-adaptation when neg-sampling is needed, and so on. By default, it will do nothing.
        """
        pass

    def data_preprocess(self):
        """This function is used to do some data preprocess, such as pre-neg-sampling and pre-data-augmentation.
        By default, it will do nothing.
        """
        pass

    def __len__(self):
        return math.ceil(self.pr_end / self.step)

    def __iter__(self):
        if self.shuffle:
            self._shuffle()
        return self

    def __next__(self):
        if self.pr >= self.pr_end:
            self.pr = 0
            self.inter_pr = 0
            raise StopIteration()
        return self._next_batch_data()

    @property
    def pr_end(self):
        """This property marks the end of dataloader.pr which is used in :meth:`__next__()`."""
        raise NotImplementedError('Method [pr_end] should be implemented')

    def _shuffle(self):
        """Shuffle the order of data, and it will be called by :meth:`__iter__()` if self.shuffle is True.
        """
        raise NotImplementedError('Method [shuffle] should be implemented.')

    def _next_batch_data(self):
        """Assemble next batch of data in form of Interaction, and return these data.

        Returns:
            Interaction: The next batch of data.
        """
        raise NotImplementedError('Method [next_batch_data] should be implemented.')


class TrainDataLoader(AbstractDataLoader):
    """
    General dataloader with negative sampling.
    """
    def __init__(self, config, dataset, batch_size=1, shuffle=False):
        super().__init__(config, dataset, additional_dataset=None,
                         batch_size=batch_size, neg_sampling=True, shuffle=shuffle)

        # special for training dataloader
        self.history_items_per_u = dict()
        # full items in training.
        self.all_items = self.dataset.df[self.dataset.iid_field].unique().tolist()
        self.all_uids = self.dataset.df[self.dataset.uid_field].unique()
        self.all_items_set = set(self.all_items)
        self.all_users_set = set(self.all_uids)
        self.all_item_len = len(self.all_items)
        # if full sampling
        self.use_full_sampling = config['use_full_sampling']

        if config['use_neg_sampling']:
            if self.use_full_sampling:
                self.sample_func = self._get_full_uids_sample
            else:
                self.sample_func = self._get_neg_sample
        else:
            self.sample_func = self._get_non_neg_sample

        self._get_history_items_u()
        self.neighborhood_loss_required = config['use_neighborhood_loss']
        if self.neighborhood_loss_required:
            self.history_users_per_i = {}
            self._get_history_users_i()
            self.user_user_dict = self._get_my_neighbors(self.config['USER_ID_FIELD'])
            self.item_item_dict = self._get_my_neighbors(self.config['ITEM_ID_FIELD'])

        # New Item Setting
        self.new_items = config['new_items']
        if self.new_items:
            self.all_new_items_set = set(np.load(f"../data/{config['dataset']}/new_items.npy"))

        # === NEW: sequence cache & adapter binding ===
        self._user_seq_cache = None
        self._user_seq_cache_maxlen = None
        # If dataset has no get_user_sequences, bind ours for compatibility with models/trainer
        if not hasattr(self.dataset, 'get_user_sequences'):
            # bind method so model/trainer can call train_data.dataset.get_user_sequences(...)
            self.dataset.get_user_sequences = self.get_user_sequences  # type: ignore[attr-defined]
            getLogger().info('[TrainDataLoader] Bound get_user_sequences() to dataset for short-term features.')

    # === NEW: short-term sequence support ===
    def _build_user_seq_cache(self):
        """Build a per-user full interaction sequence cache (chronologically if possible)."""
        df = self.dataset.df
        uid_col = self.dataset.uid_field
        iid_col = self.dataset.iid_field
        # try to sort by timestamp if available
        ts_col = getattr(self.dataset, 'ts_id', None)
        if isinstance(ts_col, str) and (ts_col in df.columns):
            df_sorted = df.sort_values([uid_col, ts_col], kind='mergesort')
        else:
            # fall back to stable sort by uid to keep original order per user
            df_sorted = df.sort_values([uid_col], kind='mergesort')
        grp = df_sorted.groupby(uid_col)[iid_col]
        self._user_seq_cache = {int(u): list(items.values) for u, items in grp}
        self._user_seq_cache_maxlen = None

    def get_user_sequences(self, max_len=50, pad_id=0, users=None):
        """Return per-user recent item sequences for short-term preference models.

        Args:
            max_len (int): Maximum length of the recent history window.
            pad_id (int): Left-padding id (usually 0).
            users (array-like|None): Specific users; if None, use all users in training set.
        Returns:
            (seq_tensor, len_tensor):
                seq_tensor: LongTensor [num_users, max_len] on self.device
                len_tensor: LongTensor [num_users] actual lengths before padding
        """
        if self._user_seq_cache is None:
            self._build_user_seq_cache()
        if users is None:
            users = self.all_uids
        # ensure iterable of ints
        users = [int(u) for u in (users.tolist() if hasattr(users, 'tolist') else list(users))]
        n = len(users)
        seqs = np.full((n, max_len), pad_id, dtype=np.int64)
        lens = np.zeros(n, dtype=np.int64)
        for i, u in enumerate(users):
            hist = self._user_seq_cache.get(u, [])
            L = min(len(hist), max_len)
            if L > 0:
                seqs[i, -L:] = hist[-L:]
            lens[i] = L
        seq_tensor = torch.as_tensor(seqs, dtype=torch.long, device=self.device)
        len_tensor = torch.as_tensor(lens, dtype=torch.long, device=self.device)
        return seq_tensor, len_tensor

    def pretrain_setup(self):
        """
        Reset dataloader. Outputing the same positive & negative samples with each training.
        Also invalidate user sequence cache due to potential order changes.
        """
        if self.shuffle:
            self.dataset = self.dataset_bk.copy(self.dataset_bk.df)
        self.all_items.sort()
        if self.use_full_sampling:
            self.all_uids.sort()
        random.shuffle(self.all_items)
        # invalidate seq cache on reset
        self._user_seq_cache = None
        self._user_seq_cache_maxlen = None
        # self.dataset.sort_by_chronological()  # optional if available

    def inter_matrix(self, form='coo', value_field=None):
        if not self.dataset.uid_field or not self.dataset.iid_field:
            raise ValueError('dataset doesn\'t exist uid/iid, thus can not converted to sparse matrix')
        return self._create_sparse_matrix(self.dataset.df, self.dataset.uid_field,
                                          self.dataset.iid_field, form, value_field)

    def _create_sparse_matrix(self, df_feat, source_field, target_field, form='coo', value_field=None):
        src = df_feat[source_field].values
        tgt = df_feat[target_field].values
        if value_field is None:
            data = np.ones(len(df_feat))
        else:
            if value_field not in df_feat.columns:
                raise ValueError('value_field [{}] should be one of `df_feat`\'s features.'.format(value_field))
            data = df_feat[value_field].values
        mat = coo_matrix((data, (src, tgt)), shape=(self.dataset.user_num, self.dataset.item_num))

        if form == 'coo':
            return mat
        elif form == 'csr':
            return mat.tocsr()
        else:
            raise NotImplementedError('sparse matrix format [{}] has not been implemented.'.format(form))

    @property
    def pr_end(self):
        if self.use_full_sampling:
            return len(self.all_uids)
        return len(self.dataset)

    def _shuffle(self):
        self.dataset.shuffle()
        if self.use_full_sampling:
            np.random.shuffle(self.all_uids)
        # dataset order changed -> invalidate seq cache
        self._user_seq_cache = None
        self._user_seq_cache_maxlen = None

    def _next_batch_data(self):
        return self.sample_func()

    def _get_neg_sample(self):
        cur_data = self.dataset[self.pr: self.pr + self.step]
        self.pr += self.step
        user_tensor = torch.tensor(cur_data[self.config['USER_ID_FIELD']].values).type(torch.LongTensor).to(self.device)
        item_tensor = torch.tensor(cur_data[self.config['ITEM_ID_FIELD']].values).type(torch.LongTensor).to(self.device)
        batch_tensor = torch.cat((torch.unsqueeze(user_tensor, 0),
                                  torch.unsqueeze(item_tensor, 0)))
        u_ids = cur_data[self.config['USER_ID_FIELD']]
        neg_ids = self._sample_neg_ids(u_ids).to(self.device)
        if self.neighborhood_loss_required:
            i_ids = cur_data[self.config['ITEM_ID_FIELD']]
            pos_neighbors, neg_neighbors = self._get_neighborhood_samples(i_ids, self.config['ITEM_ID_FIELD'])
            pos_neighbors, neg_neighbors = pos_neighbors.to(self.device), neg_neighbors.to(self.device)
            batch_tensor = torch.cat((batch_tensor, neg_ids.unsqueeze(0),
                                      pos_neighbors.unsqueeze(0), neg_neighbors.unsqueeze(0)))
        else:
            batch_tensor = torch.cat((batch_tensor, neg_ids.unsqueeze(0)))
        return batch_tensor

    def _get_non_neg_sample(self):
        cur_data = self.dataset[self.pr: self.pr + self.step]
        self.pr += self.step
        user_tensor = torch.tensor(cur_data[self.config['USER_ID_FIELD']].values).type(torch.LongTensor).to(self.device)
        item_tensor = torch.tensor(cur_data[self.config['ITEM_ID_FIELD']].values).type(torch.LongTensor).to(self.device)
        batch_tensor = torch.cat((torch.unsqueeze(user_tensor, 0),
                                  torch.unsqueeze(item_tensor, 0)))
        return batch_tensor

    def _get_full_uids_sample(self):
        user_tensor = torch.tensor(self.all_uids[self.pr: self.pr + self.step]).type(torch.LongTensor).to(self.device)
        self.pr += self.step
        return user_tensor

    def _sample_neg_ids(self, u_ids):
        neg_ids = []
        if self.new_items:
            for u in u_ids:
                iid = self._random()
                while (iid in self.history_items_per_u[u]) or (iid in self.all_new_items_set):
                    iid = self._random()
                neg_ids.append(iid)
        else:
            for u in u_ids:
                iid = self._random()
                while iid in self.history_items_per_u[u]:
                    iid = self._random()
                neg_ids.append(iid)
        return torch.tensor(neg_ids).type(torch.LongTensor)

    def _get_my_neighbors(self, id_str):
        ret_dict = {}
        a2b_dict = self.history_items_per_u if id_str == self.config['USER_ID_FIELD'] else self.history_users_per_i
        b2a_dict = self.history_users_per_i if id_str == self.config['USER_ID_FIELD'] else self.history_items_per_u
        for i, j in a2b_dict.items():
            k = set()
            for m in j:
                k |= b2a_dict.get(m, set()).copy()
            k.discard(i)                        # remove myself
            ret_dict[i] = k
        return ret_dict

    def _get_neighborhood_samples(self, ids, id_str):
        a2a_dict = self.user_user_dict if id_str == self.config['USER_ID_FIELD'] else self.item_item_dict
        all_set = self.all_users_set if id_str == self.config['USER_ID_FIELD'] else self.all_items_set
        pos_ids, neg_ids = [], []
        for i in ids:
            pos_ids_my = a2a_dict[i]
            if len(pos_ids_my) <= 0 or len(pos_ids_my)/len(all_set) > 0.8:
                pos_ids.append(0)
                neg_ids.append(0)
                continue
            pos_id = random.sample(pos_ids_my, 1)[0]
            pos_ids.append(pos_id)
            neg_id = random.sample(all_set, 1)[0]
            while neg_id in pos_ids_my:
                neg_id = random.sample(all_set, 1)[0]
            neg_ids.append(neg_id)
        return torch.tensor(pos_ids).type(torch.LongTensor), torch.tensor(neg_ids).type(torch.LongTensor)

    def _random(self):
        rd_id = random.sample(self.all_items, 1)[0]
        return rd_id

    def _get_history_items_u(self):
        uid_field = self.dataset.uid_field
        iid_field = self.dataset.iid_field
        uid_freq = self.dataset.df.groupby(uid_field)[iid_field]
        for u, u_ls in uid_freq:
            self.history_items_per_u[u] = set(u_ls.values)
        return self.history_items_per_u

    def _get_history_users_i(self):
        uid_field = self.dataset.uid_field
        iid_field = self.dataset.iid_field
        iid_freq = self.dataset.df.groupby(iid_field)[uid_field]
        for i, u_ls in iid_freq:
            self.history_users_per_i[i] = set(u_ls.values)
        return self.history_users_per_i


class EvalDataLoader(AbstractDataLoader):
    """
        additional_dataset: training dataset in evaluation
    """
    def __init__(self, config, dataset, additional_dataset=None,
                 batch_size=1, shuffle=False):
        super().__init__(config, dataset, additional_dataset=additional_dataset,
                         batch_size=batch_size, neg_sampling=False, shuffle=shuffle)

        if additional_dataset is None:
            raise ValueError('Training datasets is nan')
        self.eval_items_per_u = []
        self.eval_len_list = []
        self.train_pos_len_list = []

        self.eval_u = self.dataset.df[self.dataset.uid_field].unique()
        self.pos_items_per_u = self._get_pos_items_per_u(self.eval_u).to(self.device)
        self._get_eval_items_per_u(self.eval_u)
        self.eval_u = torch.tensor(self.eval_u).type(torch.LongTensor).to(self.device)

    @property
    def pr_end(self):
        return self.eval_u.shape[0]

    def _shuffle(self):
        self.dataset.shuffle()

    def _next_batch_data(self):
        inter_cnt = sum(self.train_pos_len_list[self.pr: self.pr+self.step])
        batch_users = self.eval_u[self.pr: self.pr + self.step]
        batch_mask_matrix = self.pos_items_per_u[:, self.inter_pr: self.inter_pr+inter_cnt].clone()
        batch_mask_matrix[0] -= self.pr
        self.inter_pr += inter_cnt
        self.pr += self.step
        return [batch_users, batch_mask_matrix]

    def _get_pos_items_per_u(self, eval_users):
        uid_field = self.additional_dataset.uid_field
        iid_field = self.additional_dataset.iid_field
        uid_freq = self.additional_dataset.df.groupby(uid_field)[iid_field]
        u_ids = []
        i_ids = []
        for i, u in enumerate(eval_users):
            u_ls = uid_freq.get_group(u).values
            i_len = len(u_ls)
            self.train_pos_len_list.append(i_len)
            u_ids.extend([i]*i_len)
            i_ids.extend(u_ls)
        return torch.tensor([u_ids, i_ids]).type(torch.LongTensor)

    def _get_eval_items_per_u(self, eval_users):
        uid_field = self.dataset.uid_field
        iid_field = self.dataset.iid_field
        uid_freq = self.dataset.df.groupby(uid_field)[iid_field]
        for u in eval_users:
            u_ls = uid_freq.get_group(u).values
            self.eval_len_list.append(len(u_ls))
            self.eval_items_per_u.append(u_ls)
        self.eval_len_list = np.asarray(self.eval_len_list)

    def get_eval_items(self):
        return self.eval_items_per_u

    def get_eval_len_list(self):
        return self.eval_len_list

    def get_eval_users(self):
        return self.eval_u.cpu()
