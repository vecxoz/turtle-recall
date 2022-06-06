#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import os
import sys
sys.path.append('lib')
import warnings
warnings.simplefilter('ignore', UserWarning)
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from vecxoz_utils import create_cv_split

# List of models to ensemble
dirs = [
    'run-20220310-1926-ef1b7',
    'run-20220316-1310-beitl',
    'run-20220317-1954-ef1l2',
    'run-20220318-1121-ef2xl',
    'run-20220322-2024-ef1l2',
    'run-20220325-1527-ef2l',
]

model_dir = 'models'
data_dir = 'data'
n_folds = 5
n_tta = 0

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# Load predictions from all models
y_preds_test = []
for counter, d in enumerate(dirs):
    for tta_id in range(n_tta + 1):
        for fold_id in range(n_folds):
            y_preds_test.append(np.load(os.path.join(model_dir, d, 'preds', 'y_pred_test_fold_%d_tta_%d.npy' % (fold_id, tta_id))))
    print(counter, end='\r')
assert len(y_preds_test) == (n_tta + 1) * len(dirs) * n_folds

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# Compute mean and argsort
probas = np.mean(y_preds_test, axis=0)
preds = np.argsort(probas, axis=1)[:, ::-1]

# train_df contains train + extra data of 2265 classes
# train_orig_df contains 100 original classes
train_df, _ = create_cv_split(data_dir, 5)
train_orig_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
turtle_ids_orig = sorted(train_orig_df['turtle_id'].unique()) # 100 unique

# Fit LabelEncoder on 2265 clases to decode our predictions
le = LabelEncoder()
le = le.fit(train_df['turtle_id'])

# Replace all predicted labels outside of 100 train ids with a "new_turtle"
label_str = []
for row in preds: # 490
    turtle_ids_predicted = le.inverse_transform(row) # transform a row of length 2265
    turtle_ids_replaced = []
    for turtle_id in turtle_ids_predicted:
        if turtle_id in turtle_ids_orig:
            turtle_ids_replaced.append(turtle_id)
        else:
            turtle_ids_replaced.append('new_turtle')
    label_str.append(turtle_ids_replaced)
label_str_npy = np.array(label_str) # (490, 2265)

# There may be more than 1 "new_turtle" prediction for any given example
# We replace all repetitions except the first with the most probable predictions form 100 train ids
rows_by_5 = []
for row in label_str_npy:
    cand = [x for x in row[row != 'new_turtle'] if x not in row[:5]][:4]
    row_new = []
    for t_id in row[:5]:
        if t_id not in row_new:
            row_new.append(t_id)
    for _ in range(5 - len(row_new)):
        row_new.append(cand.pop(0))
    rows_by_5.append(np.array(row_new))
rows_by_5_npy = np.array(rows_by_5) 

# Crete submission file
subm_df = pd.read_csv('/home/vecxoz/data/sample_submission.csv')
subm_df['prediction1'] = rows_by_5_npy[:, 0]
subm_df['prediction2'] = rows_by_5_npy[:, 1]
subm_df['prediction3'] = rows_by_5_npy[:, 2]
subm_df['prediction4'] = rows_by_5_npy[:, 3]
subm_df['prediction5'] = rows_by_5_npy[:, 4]

subm_df.to_csv('submission.csv', index=False)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
