#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import os
import re
import glob
import math
import json
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
from argparse import ArgumentParser

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class ArgumentParserExtended(ArgumentParser):
    """
    The main purpose of this class is to standardize and simplify definition of arguments
    and allow processing of True, False, and None values.
    There are 4 types of arguments (bool, int, float, str). All accept None.
    
    Usage:

    parser = ArgumentParserExtended()
    
    parser.add_str('--str', default='/home/user/data')
    parser.add_int('--int', default=220)
    parser.add_float('--float', default=3.58)
    parser.add_bool('--bool', default=True)
    
    args = parser.parse_args()
    print(parser.args_repr(args, True))
    """

    def __init__(self, *args, **kwargs):
        super(ArgumentParserExtended, self).__init__(*args, **kwargs)

    def bool_none_type(self, x):
        if x == 'True':
            return True
        elif x == 'False':
            return False
        elif x == 'None':
            return None
        else:
            raise ValueError('Unexpected literal for bool type')

    def int_none_type(self, x):
        return None if x == 'None' else int(x)

    def float_none_type(self, x):
        return None if x == 'None' else float(x)

    def str_none_type(self, x):
        return None if x == 'None' else str(x)

    def add_str(self, name, default=None, choices=None, help='str or None'):
        """
        Returns str or None
        """
        _ = self.add_argument(name, type=self.str_none_type, default=default, choices=choices, help=help)

    def add_int(self, name, default=None, choices=None, help='int or None'):
        """
        Returns int or None
        'hello' or 'none' or 1.2 will cause an error
        """
        _ = self.add_argument(name, type=self.int_none_type, default=default, choices=choices, help=help)

    def add_float(self, name, default=None, choices=None, help='float or None'):
        """
        Returns float or None
        'hello' or 'none' will cause an error
        """
        _ = self.add_argument(name, type=self.float_none_type, default=default, choices=choices, help=help)

    def add_bool(self, name, default=None, help='bool'):
        """
        Returns True, False, or None
        Anything except 'True' or 'False' or 'None' will cause an error

        `choices` are checked after type conversion of argument passed in fact
            i.e. `choices` value must be True instead of 'True'
        Default value is NOT checked using `choices`
        Default value is NOT converted using `type`
        """
        _ = self.add_argument(name, type=self.bool_none_type, default=default, choices=[True, False, None], help=help)

    @staticmethod
    def args_repr(args, print_types=False):
        ret = ''
        props = vars(args)
        keys = sorted([key for key in props])
        vals = [str(props[key]) for key in props]
        max_len_key = len(max(keys, key=len))
        max_len_val = len(max(vals, key=len))
        if print_types:
            for key in keys:
                ret += '%-*s  %-*s  %s\n' % (max_len_key, key, max_len_val, props[key], type(props[key]))
        else:   
            for key in keys:
                ret += '%-*s  %s\n' % (max_len_key, key, props[key])
        return ret.rstrip()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def init_tpu(tpu_ip_or_name=None):
    """
    Initializes `TPUStrategy` or appropriate alternative.

    tpu_ip_or_name : str or None
        e.g. 'grpc://10.70.50.202:8470' or 'node-1'

    Usage:    
    tpu, topology, strategy = init_tpu('node-1')
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_ip_or_name)
        tf.config.experimental_connect_to_cluster(tpu)
        topology = tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print('--> Master:      ', tpu.master())
        print('--> Num replicas:', strategy.num_replicas_in_sync)
        return tpu, topology, strategy
    except:
        print('--> TPU was not found!')
        # strategy = tf.distribute.get_strategy() # CPU or single GPU
        strategy = tf.distribute.MirroredStrategy() # GPU or multi-GPU
        # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy() # clusters of multi-GPU
        print('--> Num replicas:', strategy.num_replicas_in_sync)
        return None, None, strategy

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def init_tfdata(files_glob, deterministic=True, batch_size=32, auto=-1, 
                parse_example=None, mod=None, aug=None, aug2=None, tta=None, norm=None, 
                repeat=False, buffer_size=None, cache=False, drop_remainder=False):
    """
    Creates tf.data.TFRecordDataset with appropriate parameters.

    files_glob : str
        Glob wildcard for TFRecord files
    deterministic : bool
    batch_size : int
    auto : int
    parse_example, mod, aug, tta, norm : callable
        Processing functions
    repeat : bool
        Whether to repeat dataset
    buffer_size : int or None
        Shuffle buffer size. None means do NOT shuffle.
    cache : bool
        Whether to cache data
    drop_remainder : bool
        Whether to drop remainder (incomplete batch)
    """
    options = tf.data.Options()
    options.experimental_deterministic = deterministic
    files = tf.data.Dataset.list_files(files_glob, shuffle=not deterministic).with_options(options)
    print('N tfrec files:', len(files))
    #
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=auto)
    ds = ds.with_options(options)
    ds = ds.map(parse_example, num_parallel_calls=auto)
    #
    if mod:
        ds = ds.map(mod, num_parallel_calls=auto)
    if aug:
        ds = ds.map(aug, num_parallel_calls=auto)
    if aug2:
        ds = ds.map(aug2, num_parallel_calls=auto)
    if tta:
        ds = ds.map(tta, num_parallel_calls=auto)
    if norm:
        ds = ds.map(norm, num_parallel_calls=auto)
    if repeat:
        ds = ds.repeat()
    if buffer_size:
        ds = ds.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(auto)
    if cache:
        ds = ds.cache()
    #
    return ds

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class KeepLastCKPT(tf.keras.callbacks.Callback):
    """
    Sort all ckpt files matching the wildcard and remove all except last.
    If there is only one ckpt file it will not be removed.
    If save_best_only=True in ModelCheckpoint and 
        naming is consistent e.g. "model-best-f0-e001-25.3676.h5"
        then KeepLastCKPT will keep OVERALL best ckpt
    """
    #
    def __init__(self, wildcard):
        super(KeepLastCKPT, self).__init__()
        self.wildcard = wildcard
    #
    def on_epoch_begin(self, epoch, logs=None):
        # files = sorted(glob.glob(self.wildcard))
        files = sorted(tf.io.gfile.glob(self.wildcard))
        if len(files):
            for file in files[:-1]:
                # os.remove(file)
                tf.io.gfile.remove(file)
            print('Kept ckpt: %s' % files[-1])
        else:
            print('No ckpt to keep')
    #
    def on_train_end(self, logs=None):
        # files = sorted(glob.glob(self.wildcard))
        files = sorted(tf.io.gfile.glob(self.wildcard))
        if len(files):
            for file in files[:-1]:
                # os.remove(file)
                tf.io.gfile.remove(file)
            print('\nKept ckpt (final): %s' % files[-1])
        else:
            print('\nNo ckpt to keep (final)')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def apk(actual, predicted, k=5):
    """
    Computes the average precision at k.
    Args:
      actual: The turtle ID to be predicted.
      predicted : A list of predicted turtle IDs (order does matter).
      k : The maximum number of predicted elements.
    Returns:
      The average precision at k.
    """
    if len(predicted) > k:
        predicted = predicted[:k]
    #
    score = 0.0
    num_hits = 0.0
    #
    for i, p in enumerate(predicted):
        if p == actual and p not in predicted[:i]:
          num_hits += 1.0
          score += num_hits / (i + 1.0)
    #
    return score


def mapk(actual, predicted, k=5):
    """ 
    Computes the mean average precision at k.
    The turtle ID at actual[i] will be used to score predicted[i][:k] so order
    matters throughout!
    actual: A list of the true turtle IDs to score against.
    predicted: A list of lists of predicted turtle IDs.
    k: The size of the window to score within.
    Returns:
      The mean average precision at k.
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def create_cv_split(data_dir, n_splits):
    # Load
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv')) # (2145, 3)
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))   # ( 490, 2)
    
    extra_df = pd.read_csv(os.path.join(data_dir, 'extra_images.csv')) # (10658, 2)
    extra_df['image_location'] = 'top'
    
    # Concat
    train_df = pd.concat([train_df, extra_df], ignore_index=True) # (12803, 3)
    # print('N classes:', len(train_df['turtle_id'].unique())) # 2265 (not 2331 because there is intersection of 66 classes between train and extra)
    
    # Lowercase
    train_df['image_location'] = train_df['image_location'].str.lower()
    test_df['image_location']  = test_df['image_location'].str.lower()
    
    # Imitate label_id (turtle_id) for test
    test_df['turtle_id'] = 't_id_nan'
    
    # Image path
    train_df['image'] = data_dir + '/images/' + train_df['image_id'] + '.JPG'
    test_df['image']  = data_dir + '/images/' + test_df['image_id'] + '.JPG'
    
    # Label encoded turtle_id == label
    le = LabelEncoder()
    train_df['label'] = le.fit_transform(train_df['turtle_id'])
    test_df['label'] = 0
    
    # Template column for fold_id
    train_df['fold_id'] = 0
    test_df['fold_id'] = 0
    
    # Split
    # kf = KFold(n_splits=n_splits, shuffle=True, random_state=33)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=33)
    # kf = GroupKFold(n_splits=n_splits)
    #
    for fold_id, (train_index, val_index) in enumerate(kf.split(train_df, train_df['label'].values)):
        train_df.loc[train_df.index.isin(val_index), 'fold_id'] = fold_id
    # Shuffle
    train_df = train_df.sample(frac=1.0, random_state=34)
    
    return train_df, test_df

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def compute_cv_scores(data_dir, preds_dir, n_folds, tta_number, print_scores=True):
    # Load csv
    train_df, test_df = create_cv_split(data_dir, n_folds)

    # Collect all preds
    all_tta = []
    for tta_id in range(tta_number + 1):
        all_folds = []
        for fold_id in range(n_folds):
            all_folds.append(np.load(os.path.join(preds_dir, 'y_pred_val_fold_%d_tta_%d.npy' % (fold_id, tta_id))))
        all_tta.append(np.vstack(all_folds))
    
    # Collect coresponding true label
    y_true_list = []
    for fold_id in range(n_folds):
        y_true_list.append(train_df.loc[train_df['fold_id'] == fold_id, 'label'].values.ravel())
    y_true = np.hstack(y_true_list)
    
    # Compute score for original image and each TTA
    for tta_id, y_pred in enumerate(all_tta):
        score_acc = accuracy_score(y_true, np.argmax(y_pred, axis=1))
        score_acc5 = top_k_accuracy_score(y_true, y_pred, k=5)
        score_mapk = mapk(y_true, np.argsort(y_pred, axis=1)[:, ::-1][:, :5]) 
        if print_scores:
            print('TTA %d acc: %.4f,    acc5: %.4f,    mapk: %.4f' % (tta_id, score_acc, score_acc5, score_mapk))

    # Compute score for mean of all TTA
    score_acc = accuracy_score(y_true, np.argmax(np.mean(all_tta, axis=0), axis=1))
    score_acc5 = top_k_accuracy_score(y_true, np.mean(all_tta, axis=0), k=5)
    score_mapk = mapk(y_true, np.argsort(np.mean(all_tta, axis=0), axis=1)[:, ::-1][:, :5]) 
    if print_scores:
        print('-----------------------------------------------------')
        print('MEAN: acc: %.4f,    acc5: %.4f,    mapk: %.4f' % (score_acc, score_acc5, score_mapk))

    return 0

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def create_submission(data_dir, preds_dir, n_folds, tta_number, file_name=None):
    if file_name is None:
        file_name = 's-' + os.getcwd().split('/')[-1][:17] + '.csv'
    # Load csv
    train_df, test_df = create_cv_split(data_dir, n_folds)
    train_orig_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    subm_df = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))

    # Collect test preds
    y_preds_test = []
    for tta_id in range(tta_number + 1):
        for fold_id in range(n_folds):
            y_preds_test.append(np.load(os.path.join(preds_dir, 'y_pred_test_fold_%d_tta_%d.npy' % (fold_id, tta_id))))
    
    y_pred = np.argsort(np.mean(y_preds_test, axis=0), axis=1)[:, ::-1][:, :5]

    # convert label: int to str and filter "new_turtle"
    label_str = []
    le = LabelEncoder()
    le = le.fit(train_df['turtle_id'])
    turtle_ids_orig = sorted(train_orig_df['turtle_id'].unique()) # 100 unique

    for ind in range(5):
        turtle_ids_predicted = le.inverse_transform(y_pred[:, ind])
        turtle_ids_replaced = []
        # replace
        for turtle_id in turtle_ids_predicted:
            if turtle_id in turtle_ids_orig:
                turtle_ids_replaced.append(turtle_id)
            else:
                turtle_ids_replaced.append('new_turtle')
        label_str.append(turtle_ids_replaced)
    
    # Write submission
    subm_df.iloc[:, 1:] = np.stack(label_str, axis=1)
    subm_df.to_csv(file_name, index=False)
    
    return file_name

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
