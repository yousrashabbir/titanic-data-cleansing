"""
Collects the before / after numbers of every cleansing step.

It runs the cleaning pipeline step by step and returns a dictionary of
metrics that the dashboard (visuals.py, app.py) draws from.
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import data_cleansing as dc  # noqa: E402

ROOT = dc.ROOT


def quiet(func, *args):
    """Run a pipeline step without printing its console output."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args)


def py(v):
    """Convert numpy / pandas scalars to plain JSON-friendly Python."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else round(float(v), 4)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return v


def records(frame):
    return [{k: py(v) for k, v in row.items()} for row in frame.to_dict("records")]


def compute():
    """Return (metrics dict, raw DataFrame, cleaned DataFrame)."""
    raw = quiet(dc.load_data)

    # ---- run each step, keeping the intermediate states ------------------
    s_dup = quiet(dc.check_duplicates, raw)
    s_text = quiet(dc.clean_text, s_dup)
    s_title = quiet(dc.extract_title, s_text)
    s_emb = quiet(dc.impute_embarked, s_title)
    s_age = quiet(dc.impute_age, s_emb)
    s_cab = quiet(dc.handle_cabin, s_age)
    s_fare = quiet(dc.fix_zero_fares, s_cab)
    s_types = quiet(dc.convert_dtypes, s_fare)
    s_feat = quiet(dc.engineer_features, s_types)
    clean = s_feat

    D = {}

    # ---- overview ---------------------------------------------------------
    total_before = int(raw.isnull().sum().sum())
    D["overview"] = {
        "rows_before": raw.shape[0], "cols_before": raw.shape[1],
        "rows_after": clean.shape[0], "cols_after": clean.shape[1],
        "missing_before": total_before,
        "missing_after": int(clean.isnull().sum().sum()),
        "completeness_before": round(100 * (1 - total_before / raw.size), 2),
        "completeness_after": 100.0,
        "zero_fares": int((raw["Fare"] == 0).sum()),
        "whitespace": int((raw["Name"] != raw["Name"].str.strip()).sum()),
        "file_kb": round(dc.RAW_PATH.stat().st_size / 1024, 1),
        "rows_if_dropna": int(raw.dropna().shape[0]),
    }

    # ---- column profile (raw) -------------------------------------------
    prof = []
    for c in raw.columns:
        s = raw[c]
        item = {"col": c, "dtype": str(s.dtype), "non_null": int(s.notnull().sum()),
                "missing": int(s.isnull().sum()), "unique": int(s.nunique())}
        if pd.api.types.is_numeric_dtype(s):
            item["min"], item["max"] = py(s.min()), py(s.max())
            item["mean"] = py(round(s.mean(), 2))
        else:
            item["top"] = str(s.mode().iloc[0])[:22]
        prof.append(item)
    D["profile"] = prof

    # ---- duplicates --------------------------------------------------------
    D["duplicates"] = {
        "full_rows": int(raw.duplicated().sum()),
        "passenger_id": int(raw["PassengerId"].duplicated().sum()),
        "ignoring_id": int(raw.drop(columns="PassengerId").duplicated().sum()),
        "name_ticket": int(raw.duplicated(subset=["Name", "Ticket"]).sum()),
    }

    # ---- missing values ----------------------------------------------------
    D["missing_before"] = [{"col": c, "n": int(v), "pct": round(100 * v / len(raw), 2)}
                           for c, v in raw.isnull().sum().items()]
    D["missing_after"] = [{"col": c, "n": int(v), "pct": 0.0}
                          for c, v in clean.isnull().sum().items()]
    # Missing-value matrix, one string of 0/1 per column
    D["matrix_before"] = {c: "".join(raw[c].isnull().astype(int).astype(str))
                          for c in raw.columns}
    D["matrix_after"] = {c: "".join(clean[c].isnull().astype(int).astype(str))
                         for c in clean.columns}

    # ---- text cleaning ------------------------------------------------------
    ws_ids = raw.loc[raw["Name"] != raw["Name"].str.strip(), "PassengerId"]
    D["whitespace"] = [{"id": int(i),
                        "before": raw.loc[raw.PassengerId == i, "Name"].iloc[0],
                        "after": clean.loc[clean.PassengerId == i, "Name"].iloc[0]}
                       for i in ws_ids]

    # ---- titles -----------------------------------------------------------
    raw_titles = raw["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
    mapped = s_title.set_index("PassengerId")["Title"]
    t = pd.DataFrame({"raw": raw_titles.values, "group": mapped.values})
    D["titles_raw"] = [{"title": k, "n": int(v), "group": t.loc[t.raw == k, "group"].iloc[0]}
                       for k, v in raw_titles.value_counts().items()]
    D["titles_group"] = [{"title": k, "n": int(v)}
                         for k, v in mapped.value_counts().items()]

    # ---- embarked -----------------------------------------------------------
    names = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
    eb = raw["Embarked"].value_counts(dropna=False)
    ea = clean["Embarked"].value_counts()
    D["embarked"] = {
        "before": [{"port": names.get(k, "Missing") if isinstance(k, str) else "Missing",
                    "code": k if isinstance(k, str) else "NaN", "n": int(v)} for k, v in eb.items()],
        "after": [{"port": names[k], "code": k, "n": int(ea[k])} for k in ["S", "C", "Q"]],
        "missing_ids": raw.loc[raw.Embarked.isnull(), "PassengerId"].tolist(),
        "mode": "S",
    }

    # ---- age ----------------------------------------------------------------
    bins = list(range(0, 85, 5))
    rec = pd.cut(raw["Age"].dropna(), bins, right=False).value_counts(sort=False)
    imp = pd.cut(clean.loc[clean.Age_was_missing == 1, "Age"], bins,
                 right=False).value_counts(sort=False)
    D["age"] = {
        "bins": [f"{b}-{b + 4}" for b in bins[:-1]],
        "recorded": [int(x) for x in rec.values],
        "imputed": [int(x) for x in imp.values],
        "missing": int(raw["Age"].isnull().sum()),
        "mean_before": round(float(raw["Age"].mean()), 2),
        "median_before": float(raw["Age"].median()),
        "mean_after": round(float(clean["Age"].mean()), 2),
        "median_after": float(clean["Age"].median()),
    }
    med = s_emb.groupby(["Title", "Pclass"], observed=True)["Age"].median().unstack()
    D["age_medians"] = {"titles": list(med.index),
                        "values": [[py(v) for v in row] for row in med.values]}
    filled = s_age.loc[s_age.Age_was_missing == 1].groupby(
        ["Title", "Pclass"], observed=True).size().unstack(fill_value=0)
    D["age_filled_counts"] = {str(ti): {str(k): int(v) for k, v in filled.loc[ti].items()}
                              for ti in filled.index}

    # ---- cabin --------------------------------------------------------------
    D["cabin"] = {
        "has": int(clean["HasCabin"].sum()), "unknown": int((clean["HasCabin"] == 0).sum()),
        "decks": [{"deck": k, "n": int(v)} for k, v in
                  clean.loc[clean.Deck != "U", "Deck"].value_counts().sort_index().items()
                  if v > 0],
    }

    # ---- fare ---------------------------------------------------------------
    z = raw[raw.Fare == 0]
    D["fare_zero"] = {
        "by_class": {str(k): int(v) for k, v in z["Pclass"].value_counts().sort_index().items()},
        "medians": {str(k): round(float(v), 2) for k, v in
                    s_cab[s_cab.Fare > 0].groupby("Pclass")["Fare"].median().items()},
        "rows": [{"id": int(r.PassengerId), "name": r.Name[:28], "pclass": int(r.Pclass),
                  "before": 0.0,
                  "after": float(clean.loc[clean.PassengerId == r.PassengerId, "Fare"].iloc[0])}
                 for r in z.itertuples()],
    }

    # ---- before / after rows --------------------------------------------------
    ids = [6, 18, 20, 29, 33, 37, 62, 830]
    cols = ["PassengerId", "Name", "Pclass", "Sex", "Age", "Fare", "Cabin", "Embarked"]
    b = raw[raw.PassengerId.isin(ids)][cols].copy()
    a = clean[clean.PassengerId.isin(ids)][cols].copy()
    for f in (b, a):
        f["Name"] = f["Name"].str.slice(0, 26)
        f["Pclass"] = f["Pclass"].astype(int)
        f["Sex"] = f["Sex"].astype(str)
        f["Embarked"] = f["Embarked"].astype(object)
    D["rows"] = {"cols": cols, "before": records(b), "after": records(a)}

    # ---- dtypes ---------------------------------------------------------------
    D["dtypes"] = [{"col": c, "before": str(raw[c].dtype) if c in raw else "-",
                    "after": str(clean[c].dtype), "new": c not in raw}
                   for c in clean.columns]

    # ---- outliers ---------------------------------------------------------------
    out = []
    for c in ["Age", "Fare", "SibSp", "Parch"]:
        q1, q3 = clean[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out.append({"col": c, "q1": py(q1), "q3": py(q3), "low": round(lo, 2),
                    "high": round(hi, 2), "min": py(clean[c].min()), "max": py(clean[c].max()),
                    "n": int(((clean[c] < lo) | (clean[c] > hi)).sum())})
    D["outliers"] = out
    box = []
    for cl in [1, 2, 3]:
        f = clean.loc[clean.Pclass == cl, "Fare"]
        q1, q2, q3 = f.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        lo_w = f[f >= q1 - 1.5 * iqr].min()
        hi_w = f[f <= q3 + 1.5 * iqr].max()
        pts = sorted(set(round(float(x), 2) for x in f[(f > hi_w) | (f < lo_w)]))
        box.append({"cls": cl, "q1": py(q1), "med": py(q2), "q3": py(q3),
                    "lo": py(lo_w), "hi": py(hi_w), "points": pts,
                    "n_out": int(((f > hi_w) | (f < lo_w)).sum())})
    D["fare_box"] = box

    # ---- derived features ----------------------------------------------------------
    D["features"] = {
        "family": [{"size": int(k), "n": int(v)} for k, v in
                   clean["FamilySize"].value_counts().sort_index().items()],
        "alone": int(clean["IsAlone"].sum()),
        "with_family": int((clean["IsAlone"] == 0).sum()),
        "agegroup": [{"g": str(k), "n": int(v)} for k, v in
                     clean["AgeGroup"].value_counts().sort_index().items()],
    }

    # ---- validation --------------------------------------------------------------
    D["validation"] = [
        {"check": "No missing values", "detail": f"{int(clean.isnull().sum().sum())} missing cells",
         "ok": bool(clean.isnull().sum().sum() == 0)},
        {"check": "No duplicate rows", "detail": f"{int(clean.duplicated().sum())} duplicates",
         "ok": bool(clean.duplicated().sum() == 0)},
        {"check": "Age between 0 and 100", "detail": f"range {clean.Age.min()} - {clean.Age.max()}",
         "ok": bool(clean["Age"].between(0, 100).all())},
        {"check": "Every Fare above 0", "detail": f"minimum fare {clean.Fare.min():.2f}",
         "ok": bool((clean["Fare"] > 0).all())},
        {"check": "Sex is male / female only", "detail": ", ".join(sorted(clean.Sex.astype(str).unique())),
         "ok": set(clean["Sex"].astype(str).unique()) == {"male", "female"}},
        {"check": "Embarked is S / C / Q only", "detail": ", ".join(sorted(clean.Embarked.astype(str).unique())),
         "ok": set(clean["Embarked"].astype(str).unique()) <= {"S", "C", "Q"}},
        {"check": "Row count preserved", "detail": f"{len(raw)} → {len(clean)} rows",
         "ok": len(raw) == len(clean)},
    ]

    # ---- cleaned preview ---------------------------------------------------------
    prev = clean.head(12)[["PassengerId", "Survived", "Pclass", "Title", "Sex", "Age",
                           "Fare", "Deck", "Embarked", "FamilySize", "IsAlone", "AgeGroup"]].copy()
    for c in prev.columns:
        if isinstance(prev[c].dtype, pd.CategoricalDtype):
            prev[c] = prev[c].astype(str)
    D["preview"] = {"cols": list(prev.columns), "rows": records(prev)}

    return D, raw, clean


if __name__ == "__main__":
    D, raw, clean = compute()
    print("Metrics collected:", ", ".join(D))
