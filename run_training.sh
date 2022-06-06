#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

cd $HOME/solution/models/run-20220310-1926-ef1b7
python3 run.py --tpu_ip_or_name=node-01 --data_tfrec_dir=gs://bucket/dir

cd $HOME/solution/models/run-20220316-1310-beitl
python3 run.py --tpu_ip_or_name=node-01 --data_tfrec_dir=gs://bucket/dir

cd $HOME/solution/models/run-20220317-1954-ef1l2
python3 run.py --tpu_ip_or_name=node-01 --data_tfrec_dir=gs://bucket/dir

cd $HOME/solution/models/run-20220318-1121-ef2xl
python3 run.py --tpu_ip_or_name=node-01 --data_tfrec_dir=gs://bucket/dir

cd $HOME/solution/models/run-20220322-2024-ef1l2
python3 run.py --tpu_ip_or_name=node-01 --data_tfrec_dir=gs://bucket/dir

cd $HOME/solution/models/run-20220325-1527-ef2l
python3 run.py --tpu_ip_or_name=node-01 --data_tfrec_dir=gs://bucket/dir

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

