import sys
sys.path.append('/mnt/batch/tasks/shared/LS_root/mounts/clusters/churn-model-instance/code/Users/<folder_path_here>')
from py_files.inference import make_predictions
import pandas as pd

user_data_df = pd.read_csv('/mnt/batch/tasks/shared/LS_root/mounts/clusters/churn-model-instance/code/Users/<folder_path_here>')

predictions = make_predictions(user_data_df)

print(predictions)