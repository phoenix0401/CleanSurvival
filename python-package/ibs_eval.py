import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import integrated_brier_score
import glob
import os
import re

missingness = 40
mechanism = "MCAR"
pipeline = "R"
model = "COX"
data_name = "rotterdam"
pattern = (f"save/{data_name}_cox_{pipeline}/*_*_{missingness}_{mechanism}.csv_"
           f"pipeline_*_{model}_cleaned.csv") #can choose folder for the preprocessed and trained data
data_paths = sorted(glob.glob(pattern))[:10] 
print("Found data files:", data_paths)
time_col = 'dtime'
event_col = 'death'
models = ['COX', 
          'RSF', 
         # 'NN'
         ]
N_Runs = 20

def load_survival_data(data_path, time_col=time_col, event_col=event_col, test_size=0.2, random_state=42):
    df = pd.read_csv(data_path)
    time = df[time_col].to_numpy()
    event = df[event_col].to_numpy().astype(bool)
    X = df.drop(columns=[time_col, event_col, "rtime", "recur"], errors='ignore')

    X_train, X_test, time_train, time_test, event_train, event_test = train_test_split(
        X, time, event, test_size=0.2, random_state=random_state
)   

    y_train = Surv.from_arrays(event = event_train, time = time_train)
    y_test = Surv.from_arrays(event = event_test, time = time_test)
    return X_train, X_test, y_train, y_test, time_test

def time_grid(time_test):
    t_min = np.min(time_test[time_test > 0])
    t_max = np.max(time_test)
    return np.linspace(t_min, t_max, 100)

def cox_ibs(X_train, X_test, y_train, y_test, times_test):
    cox_model = CoxPHSurvivalAnalysis().fit(X_train, y_train)
    surv_funcs = cox_model.predict_survival_function(X_test)
    
    max_train_time = float(np.max(y_train["time"]))
    min_train_time = float(np.min(y_train["time"]))
    max_domain = min(fn.domain[1] for fn in surv_funcs)
    min_domain = max(fn.domain[0] for fn in surv_funcs)
    test_min = float(np.min(times_test[times_test > 0]))
    test_max = float(np.max(times_test[times_test > 0]))
    
    t_min = max(min_domain, test_min, min_train_time)
    t_max = min(max_domain, test_max, max_train_time)

    if t_max <= t_min:
        raise ValueError(f"Invalid time window: t_min={t_min}, t_max={t_max}, "
                         f"max_domain={max_domain}, min_domain={min_domain},"
                         f"test_max={test_max}, test_min={test_min}",
                         f"max_train_time={max_train_time}, min_train_time={min_train_time}")
    t_max_adj = np.nextafter(t_max, t_min)
    times = np.linspace(t_min, t_max_adj, 100)
    
    estimate = np.asarray([[fn(t) for t in times] for fn in surv_funcs])
    print(
    "DEBUG IBS window:",
    "min(times) =", times.min(),
    "max(times) =", times.max(),
    "max_train_time =", np.max(y_train["time"])
    )
    ibs = integrated_brier_score(y_train, y_test, estimate, times)
    return float(ibs)


def rsf_ibs(X_train, X_test, y_train, y_test, times_test):
    rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10,
                               min_samples_leaf=15,
                               n_jobs=-1, random_state=42)
    rsf.fit(X_train, y_train)
    surv_funcs = rsf.predict_survival_function(X_test)
    max_train_time = float(np.max(y_train["time"]))
    min_train_time = float(np.min(y_train["time"]))
    max_domain = min(fn.domain[1] for fn in surv_funcs)
    min_domain = max(fn.domain[0] for fn in surv_funcs)
    test_min = float(np.min(times_test[times_test > 0]))
    test_max = float(np.max(times_test[times_test > 0]))
    
    t_min = max(min_domain, test_min, min_train_time)
    t_max = min(max_domain, test_max, max_train_time)

    if t_max <= t_min:
        raise ValueError(f"Invalid time window: t_min={t_min}, t_max={t_max}, "
                         f"max_domain={max_domain}, min_domain={min_domain},"
                         f"test_max={test_max}, test_min={test_min}",
                         f"max_train_time={max_train_time}, min_train_time={min_train_time}")
    t_max_adj = np.nextafter(t_max, t_min)
    times = np.linspace(t_min, t_max_adj, 100)
    estimate = np.asarray([[fn(t) for t in times] for fn in surv_funcs])
    ibs = integrated_brier_score(y_train, y_test, estimate, times)
    return float(ibs)

def nn_ibs(X_train, X_test, y_train, y_test, times_test):
    # Placeholder for Neural Network IBS calculation
    return float('nan')

def infer_dataset_name(data_path):
    base = os.path.basename(data_path)
    return base.split("_cleaned.csv")[0]

out_directory = f"ibs_results/{data_name}/{mechanism}_{pipeline}_{model}"
os.makedirs(out_directory, exist_ok=True)
IBS_results = f"ibs_results_raw_{data_name}_{model}_{mechanism}_{missingness}_{pipeline}.csv"
raw_result_path = os.path.join(out_directory, IBS_results)

all_results = []
for run in range(N_Runs):
    print(f"Run {run+1}/{N_Runs}")
    seed = run
    for data_path in data_paths:
        dataset_name = infer_dataset_name(data_path)
        X_train, X_test, y_train, y_test, times_test = load_survival_data(data_path, time_col=time_col, event_col=event_col, test_size=0.2, random_state=seed)
        base = os.path.basename(data_path)
        dataset_name = os.path.splitext(base)[0]
        row = {"run": run, "dataset": dataset_name, "file_path": os.path.basename(data_path)}
        if 'COX' in models:
            try:
                cox_ibs_value = cox_ibs(X_train, X_test, y_train, y_test, times_test)
            except ValueError as e:
                if "time must be smaller than largest observed time point" in str(e):
                    cox_ibs_value = np.nan
                else:
                    raise
            row["ibs_cox"] = cox_ibs_value
        if 'RSF' in models:
            try:
                rsf_ibs_value = rsf_ibs(X_train, X_test, y_train, y_test, times_test)
            except ValueError as e:
                if "time must be smaller than largest observed time point" in str(e):
                    cox_ibs_value = np.nan
                else:
                    raise
            row["ibs_rsf"] = rsf_ibs_value
        #if 'NN' in models:
        #    nn_ibs_value = nn_ibs(X_train, X_test, y_train, y_test, times_test)
        #    row["ibs_nn"] = nn_ibs_value
        all_results.append(row)
        print(f"Results for {data_path}: Cox IBS = {cox_ibs_value}, RSF IBS = {rsf_ibs_value}")
    

pd.DataFrame(all_results).to_csv(raw_result_path, index=False)
print(f"All results saved to {raw_result_path}")

df = pd.read_csv(raw_result_path)
summary = (
    df.groupby('dataset')
    .agg(
        mean_ibs_cox=('ibs_cox', 'mean'),
        sd_ibs_cox=('ibs_cox', 'std'),
        mean_ibs_rsf=('ibs_rsf', 'mean'),
        sd_ibs_rsf=('ibs_rsf', 'std'),
        #mean_ibs_nn=('ibs_nn', 'mean'),
        #sd_ibs_nn=('ibs_nn', 'std'),
    )   .reset_index()
)
save_name_data = f"ibs_results_summary_by_{data_name}_{missingness}{mechanism}_{pipeline}_{model}.csv"
save_name_data_path = os.path.join(out_directory, save_name_data)
summary.to_csv(save_name_data_path, index=False)
print("Summary of results:", summary)

file_summary = (
    df.groupby(['dataset', 'file_path'])
    .agg(
        mean_ibs_cox=('ibs_cox', 'mean'),
        sd_ibs_cox=('ibs_cox', 'std'),
        mean_ibs_rsf=('ibs_rsf', 'mean'),
        sd_ibs_rsf=('ibs_rsf', 'std'),
        #mean_ibs_nn=('ibs_nn', 'mean'),
        #sd_ibs_nn=('ibs_nn', 'std'),
    )   .reset_index()
)
save_name_file = f"ibs_results_summary_by_file_{data_name}_{missingness}{mechanism}_{pipeline}_{model}.csv"
save_name_file_path = os.path.join(out_directory, save_name_file)
file_summary.to_csv(save_name_file_path, index=False)