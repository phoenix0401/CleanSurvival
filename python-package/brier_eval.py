import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import brier_score
from sklearn.preprocessing import StandardScaler
import os
import glob

missingness = 10
mechanism = "MNAR"
pipeline = "L"
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

def load_survival_data(data_path, time_col=time_col, event_col=event_col, test_size=0.2, random_state=42):
    df = pd.read_csv(data_path)
    time = df[time_col].values
    event = df[event_col].values == 1
    X = df.drop(columns=[time_col, event_col], errors='ignore')

    X_train, X_test, time_train, time_test, event_train, event_test = train_test_split(
        X, time, event, test_size=0.2, random_state=random_state
)   

    y_train = Surv.from_arrays(event = event_train, time = time_train)
    y_test = Surv.from_arrays(event = event_test, time = time_test)
    return X_train, X_test, y_train, y_test, time_test

def timepoints_grid(y_train, y_test, n_pts=50):
    train_times, test_times = y_train["time"], y_test["time"]
    t_min = max(train_times.min(), test_times.min())
    t_max = min(train_times.max(), test_times.max()) * 0.95
    times = np.linspace(t_min, t_max, n_pts)
    return times

def cox_brier(X_train, X_test, y_train, y_test, times_test):
    scaling = StandardScaler() #needed for survival in COX models
    X_train_scaled = scaling.fit_transform(X_train)
    X_test_scaled = scaling.transform(X_test)
    
    cox = CoxPHSurvivalAnalysis(alpha=1e-2)
    cox.fit(X_train_scaled, y_train)
    surv_funcs = cox.predict_survival_function(X_test_scaled)
    step_f = surv_funcs[0]
    times = timepoints_grid(y_train, y_test)
    estimate = np.asarray([fn(times) for fn in surv_funcs])
    
    times_out, brier_scores = brier_score(
        survival_train=y_train,
        survival_test = y_test,
        estimate = estimate,
        times= times
    )    
    #print(f"  COX Brier: [{brier_scores.min():.4f}, {brier_scores.max():.4f}]")
    return {
        "brier_scores": brier_scores,
        "times": times_out
    }
    
def rsf_brier(X_train, X_test, y_train, y_test, times_test):
    rsf = RandomSurvivalForest(
        n_estimators=100,
        min_samples_split = 10,
        min_samples_leaf = 15,
        random_state=42,
        n_jobs=1
    )
    rsf.fit(X_train, y_train)
    surv_funcs = rsf.predict_survival_function(X_test)

    times = timepoints_grid(y_train, y_test)
    estimate = np.asarray([fn(times) for fn in surv_funcs])
    
    times_out, brier_scores = brier_score(
        survival_train=y_train,
        survival_test = y_test,
        estimate = estimate,
        times= times
    )    
    #print(f"  RSF Brier: [{brier_scores.min():.4f}, {brier_scores.max():.4f}]")
    return {
        "brier_scores": brier_scores,
        "times": times_out
    }
    

def infer_dataset_name(data_path):
    base = os.path.basename(data_path)
    return base.split("_cleaned.csv")[0]

out_directory = f"ibs_results/{data_name}/{mechanism}_{pipeline}_{model}"
os.makedirs(out_directory, exist_ok=True)

all_brier_scores = []
for data_path in data_paths:
    print("Computing Brier for data_path:", data_path)
    dataset_name = infer_dataset_name(data_path)
    X_train, X_test, y_train, y_test, _ = load_survival_data(data_path)
    row = {"dataset": dataset_name, "file_path": os.path.basename(data_path)}

    cox_result = None
    if "COX" in models:
        try:
            cox_result = cox_brier(X_train, X_test, y_train, y_test, None)
        except Exception as e:
            print(f"Cox error: {e}")
            
    rsf_result = None
    if "RSF" in models:
        try:
            rsf_result = rsf_brier(X_train, X_test, y_train, y_test, None)
        except Exception as e:
            print(f"RSF error: {e}")
            
    if cox_result is not None:
        for t, brier in zip(cox_result["times"], cox_result["brier_scores"]):
            all_brier_scores.append({
                "dataset": dataset_name, "file": os.path.basename(data_path),
                    "model": "COX", "time": float(t), "brier_score": float(brier)
                })
    if rsf_result is not None:
        for t, brier in zip(rsf_result["times"], rsf_result["brier_scores"]):
            all_brier_scores.append({
                "dataset": dataset_name, "file": os.path.basename(data_path),
                    "model": "RSF", "time": float(t), "brier_score": float(brier)
                })

brier_df = pd.DataFrame(all_brier_scores)
print(brier_df.head(3))
print(f"Brier range: {brier_df.brier_score.min():.4f}-{brier_df.brier_score.max():.4f}")
brier_df.to_csv(f"{out_directory}/{missingness}_{mechanism}_brier_scores.csv")

brier_df['pipeline'] = brier_df['dataset'].str.extract(r'pipeline_(\d+)').astype(int)

# Compute summary across pipelines
summary_df = (
    brier_df.groupby(["model", "time"])
            .agg(mean_brier=("brier_score", "mean"),
                 std_brier=("brier_score", "std"))
            .reset_index()
)

# Save summary CSV
summary_csv = f"{out_directory}/{missingness}_{mechanism}_brier_summary_across_pipelines.csv"
summary_df.to_csv(summary_csv, index=False)

