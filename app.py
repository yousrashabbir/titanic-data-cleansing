"""
Interactive data-cleansing dashboard (Streamlit).

    pip install -r requirements.txt
    streamlit run app.py

Opens in the browser at http://localhost:8501
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from dashboard_data import compute  # noqa: E402
import visuals as vz  # noqa: E402

st.set_page_config(page_title="Titanic Data Cleansing Dashboard",
                   page_icon="🚢", layout="wide")


@st.cache_data
def load():
    return compute()


D, raw, clean = load()
o = D["overview"]


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# ---- sidebar ----------------------------------------------------------------
st.sidebar.title("🚢 Data Cleansing")
st.sidebar.caption("Titanic dataset · Kaggle train.csv")
section = st.sidebar.radio("Go to", [
    "Overview", "Explore raw data", "Duplicates", "Missing values",
    "Text & titles", "Imputation", "Hidden missing values",
    "Data types & outliers", "Derived columns", "Validation", "Cleaned data"])
st.sidebar.markdown("---")
st.sidebar.markdown("[Dataset on Kaggle](https://www.kaggle.com/competitions/titanic/data)")

# ---- header + KPIs --------------------------------------------------------------
st.title("Titanic Dataset · Data Cleansing Dashboard")
st.caption("Python · pandas · Matplotlib · Streamlit")
k = st.columns(6)
k[0].metric("Rows", o["rows_after"], "0 removed", delta_color="off")
k[1].metric("Columns", o["cols_after"], f"+{o['cols_after'] - o['cols_before']} derived")
k[2].metric("Missing cells", o["missing_after"], f"-{o['missing_before']}", delta_color="inverse")
k[3].metric("Completeness", "100%", f"+{100 - o['completeness_before']:.1f} pts")
k[4].metric("Duplicates", 0, "none found", delta_color="off")
k[5].metric("Zero fares fixed", o["zero_fares"])
st.markdown("---")

# ---- sections -----------------------------------------------------------------------
if section == "Overview":
    c1, c2 = st.columns(2)
    with c1:
        show(vz.overview_panel(D))
    with c2:
        show(vz.pipeline_panel())
    c1, c2 = st.columns(2)
    with c1:
        show(vz.missing_panel(D, "before"))
    with c2:
        show(vz.matrix_panel(D))

elif section == "Explore raw data":
    show(vz.profile_panel(D))
    st.subheader("Raw dataset")
    only_missing = st.checkbox("Show only rows with missing values")
    view = raw[raw.isnull().any(axis=1)] if only_missing else raw
    st.dataframe(view, height=380)
    st.caption(f"{len(view)} of {len(raw)} rows shown")

elif section == "Duplicates":
    show(vz.duplicates_panel(D))
    st.success("No duplicate values found - all 891 rows are unique.")

elif section == "Missing values":
    state = st.radio("Show", ["Before cleaning", "After cleaning"], horizontal=True)
    c1, c2 = st.columns(2)
    with c1:
        show(vz.missing_panel(D, "before" if state.startswith("Before") else "after"))
    with c2:
        show(vz.matrix_panel(D))
    show(vz.rows_panel(D, "before" if state.startswith("Before") else "after"))

elif section == "Text & titles":
    c1, c2 = st.columns(2)
    with c1:
        show(vz.text_panel(D))
    with c2:
        show(vz.titles_panel(D))

elif section == "Imputation":
    tab1, tab2, tab3 = st.tabs(["Age", "Embarked", "Cabin"])
    with tab1:
        show(vz.age_panel(D))
    with tab2:
        show(vz.embarked_panel(D))
    with tab3:
        show(vz.cabin_panel(D))
    c1, c2 = st.columns(2)
    with c1:
        show(vz.rows_panel(D, "before"))
    with c2:
        show(vz.rows_panel(D, "after"))

elif section == "Hidden missing values":
    show(vz.fare_panel(D))

elif section == "Data types & outliers":
    c1, c2 = st.columns(2)
    with c1:
        show(vz.dtypes_panel(D))
    with c2:
        show(vz.outliers_panel(D))

elif section == "Derived columns":
    show(vz.features_panel(D))

elif section == "Validation":
    c1, c2 = st.columns(2)
    with c1:
        show(vz.validation_panel(D))
    with c2:
        show(vz.missing_panel(D, "after"))

elif section == "Cleaned data":
    show(vz.preview_panel(D))
    st.subheader("Cleaned dataset")
    cols = st.multiselect("Columns", list(clean.columns), default=list(clean.columns))
    st.dataframe(clean[cols], height=420)
    st.download_button("Download titanic_cleaned.csv",
                       clean.to_csv(index=False).encode("utf-8"),
                       "titanic_cleaned.csv", "text/csv")
