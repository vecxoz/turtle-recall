#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# Batch size is configured to run inference on a single V100 (P100) GPU

cd $HOME/solution/models/run-20220310-1926-ef1b7
python3 run.py --job=test --batch_size=16

cd $HOME/solution/models/run-20220316-1310-beitl
python3 run.py --job=test --batch_size=16

cd $HOME/solution/models/run-20220317-1954-ef1l2
python3 run.py --job=test --batch_size=16

cd $HOME/solution/models/run-20220318-1121-ef2xl
python3 run.py --job=test --batch_size=16

cd $HOME/solution/models/run-20220322-2024-ef1l2
python3 run.py --job=test --batch_size=16

cd $HOME/solution/models/run-20220325-1527-ef2l
python3 run.py --job=test --batch_size=16


cd $HOME/solution
python3 ensemble.py

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


