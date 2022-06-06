#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import os
import sys
sys.path.append('lib')
import glob
import warnings
warnings.simplefilter('ignore', UserWarning)
import collections
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
import tensorflow as tf
print('tf:', tf.__version__)
from vecxoz_utils import create_cv_split
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--data_dir', default='data', type=str, help='Data directory')
parser.add_argument('--out_dir', default='data/tfrec', type=str, help='Out directory')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
  
class TFRecordProcessor(object):
    def __init__(self):
        self.n_examples = 0
    #
    def _bytes_feature(self, value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    #
    def _int_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    #
    def _float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    #
    def _process_example(self, ind, A, B, C, D):
        self.n_examples += 1
        feature = collections.OrderedDict()
        #
        feature['image_id'] = self._bytes_feature(A[ind].encode('utf-8'))
        feature['image'] =    self._bytes_feature(tf.io.read_file(B[ind]))
        feature['label_id'] = self._bytes_feature(C[ind].encode('utf-8'))
        feature['label'] =    self._int_feature(D[ind])
        #
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        self._writer.write(example_proto.SerializeToString())
    #
    def write_tfrecords(self, A, B, C, D, n_shards=1, file_out='train.tfrecord'):
        n_examples_per_shard = A.shape[0] // n_shards
        n_examples_remainder = A.shape[0] %  n_shards   
        self.n_examples = 0
        #
        for shard in range(n_shards):
            self._writer = tf.io.TFRecordWriter('%s-%05d-of-%05d' % (file_out, shard, n_shards))
            #
            start = shard * n_examples_per_shard
            if shard == (n_shards - 1):
                end = (shard + 1) * n_examples_per_shard + n_examples_remainder
            else:
                end = (shard + 1) * n_examples_per_shard
            #
            print('Shard %d of %d: (%d examples)' % (shard, n_shards, (end - start)))
            for i in range(start, end):
                self._process_example(i, A, B, C, D)
                print(i, end='\r')
            #
            self._writer.close()
        #
        return self.n_examples

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

train_df, test_df = create_cv_split(args.data_dir, n_splits=5)

tfrp = TFRecordProcessor()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

for fold_id in range(len(train_df['fold_id'].unique())):
    print('Fold:', fold_id)
    n_written = tfrp.write_tfrecords(
        train_df[train_df['fold_id'] == fold_id]['image_id'].values,
        train_df[train_df['fold_id'] == fold_id]['image'].values,
        train_df[train_df['fold_id'] == fold_id]['turtle_id'].values,
        train_df[train_df['fold_id'] == fold_id]['label'].values,
        #
        n_shards=1, 
        file_out=os.path.join(args.out_dir, 'fold.%d.tfrecord' % fold_id))

n_written = tfrp.write_tfrecords(
    test_df['image_id'].values,
    test_df['image'].values,
    test_df['turtle_id'].values,
    test_df['label'].values,
    #
    n_shards=1,
    file_out=os.path.join(args.out_dir, 'test.tfrecord'))

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


