import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss


def macro_f1(y_true, proba, threshold=0.5):
    y_pred = (np.asarray(proba) >= threshold).astype(int) # boolean converted ti int, true  = 1, false = 0
    return f1_score(y_true, y_pred, average="macro") #for each class we calculate the f1 score and then we take their avg, that is the macro f1 score, more balanced 

def safe_log_loss(y_true, proba):
    return log_loss(y_true, proba, labels=[0, 1])             # labels=[0,1]: no crash if a fold has one class

def top1_accuracy(cand_df, proba, obs_df):
    df = cand_df[["trip_id", "obs_idx", "arc_id"]].copy()
    df["score"] = np.asarray(proba)                           # 1. attach scores to candidate rows

    best = df.loc[df.groupby(["trip_id", "obs_idx"])["score"].idxmax()]   # 2. best candidate per GPS point

    labelled = obs_df[obs_df["true_arc"] != -1][["trip_id", "obs_idx", "true_arc"]]   # 3. labelled points only

    merged = labelled.merge(best[["trip_id", "obs_idx", "arc_id"]],
                            on=["trip_id", "obs_idx"], how="left")   # 4. points with no candidates → NaN

    correct = merged["arc_id"] == merged["true_arc"]          # 5. NaN == number → False → counts as a miss
    return correct.mean()                                     # 6. fraction of labelled points correct