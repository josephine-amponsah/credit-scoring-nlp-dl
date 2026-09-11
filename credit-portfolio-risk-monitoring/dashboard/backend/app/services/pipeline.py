import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


DEFAULT_MODEL_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "models", "xgbmodel1.pkl"),
    os.path.join(os.path.dirname(__file__), "models", "xgbmodel2.pkl"),
    os.path.join(os.path.dirname(__file__), "models", "rfmodel.pkl"),
]

DEFAULT_SCALER = os.path.join(os.path.dirname(__file__), "..", "..", "credit-risk-scorecard", "artefacts", "scaler.pkl")
DEFAULT_LE = os.path.join(os.path.dirname(__file__), "..", "..", "credit-risk-scorecard", "artefacts", "le_dict.pkl")

# Features list taken from lending_club_ml notebook (training keep_cols)

def load_artifacts(model_candidates=None, scaler_path=None, le_path=None):
    model_candidates = model_candidates or DEFAULT_MODEL_CANDIDATES
    scaler_path = scaler_path or DEFAULT_SCALER
    le_path = le_path or DEFAULT_LE

    model = None
    for p in model_candidates:
        try:
            model = joblib.load(p)
            break
        except Exception:
            model = None
    scaler = None
    try:
        scaler = joblib.load(scaler_path)
    except Exception:
        scaler = None
    le_dict = None
    try:
        le_dict = joblib.load(le_path)
    except Exception:
        le_dict = None
    return model, scaler, le_dict


def prepare_features(df, le_dict=None, numeric_fill_strategy="median"):
    # ensure we operate on a copy
    df = df.copy()

    # numeric features
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        if numeric_fill_strategy == "median":
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        else:
            df[num_cols] = df[num_cols].fillna(0)

    # categorical features: encode to integers (either using saved encoders or factorize)
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    X_cat = pd.DataFrame(index=df.index)
    for c in cat_cols:
        col_vals = df[c].fillna("__nan__").astype(str)
        if le_dict and c in le_dict:
            try:
                enc = le_dict[c]
                encoded = enc.transform(col_vals)
            except Exception:
                encoded = pd.factorize(col_vals)[0]
        else:
            encoded = pd.factorize(col_vals)[0]
        X_cat[c] = encoded

    X_num = df[num_cols] if num_cols else pd.DataFrame(index=df.index)
    if not X_num.empty and not X_cat.empty:
        X = pd.concat([X_num, X_cat], axis=1)
    elif not X_num.empty:
        X = X_num
    else:
        X = X_cat

    return X, num_cols, cat_cols


def infer_pd_for_df(df, model=None, scaler=None, le_dict=None):
    X, num_cols, cat_cols = prepare_features(df, le_dict=le_dict)
    X_scaled = None
    if X.shape[1] == 0:
        return pd.Series([np.nan] * len(df), index=df.index)
    try:
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            tmp_scaler = StandardScaler()
            X_scaled = tmp_scaler.fit_transform(X)
    except Exception:
        X_scaled = X.values

    if model is None:
        return pd.Series([np.nan] * len(df), index=df.index)

    try:
        probs = model.predict_proba(X_scaled)
        pd_vals = pd.Series(probs[:, 1], index=df.index)
    except Exception:
        try:
            preds = model.predict(X_scaled)
            pd_vals = pd.Series(preds, index=df.index)
        except Exception:
            pd_vals = pd.Series([np.nan] * len(df), index=df.index)
    return pd_vals


def process_and_save_quarters(df, out_dir="data_splits", model_candidates=None, scaler_path=None, le_path=None, keep_cols=None):
    os.makedirs(out_dir, exist_ok=True)
    model, scaler, le_dict = load_artifacts(model_candidates=model_candidates, scaler_path=scaler_path, le_path=le_path)

    # If keep_cols provided, subset but keep missing columns as NaN
    if keep_cols is not None:
        missing = [c for c in keep_cols if c not in df.columns]
        if missing:
            # create missing columns with NaN so final output includes them
            for c in missing:
                df[c] = np.nan
        df = df[keep_cols + [c for c in df.columns if c not in keep_cols]]

    df["quarter"] = df["issue_d"].dt.to_period("Q")
    for q, grp in df.groupby("quarter"):
        grp = grp.copy()
        # clean object columns
        for col in grp.select_dtypes(include=["object"]).columns:
            grp[col] = grp[col].where(~grp[col].isnull(), None)
            grp[col] = grp[col].apply(lambda v: v.decode("utf-8", errors="replace") if isinstance(v, (bytes, bytearray)) else (str(v) if v is not None else None))

        out_df = grp.copy()
        pd_vals = infer_pd_for_df(grp, model=model, scaler=scaler, le_dict=le_dict)
        out_df["PD"] = pd_vals
        fname = os.path.join(out_dir, f"data_{q}.parquet")
        out_df.to_parquet(fname, index=False, compression="snappy")

    return True
