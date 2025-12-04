import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Sleep Quality & Academic Productivity Analytics",
    layout="wide"
)

# --------------------------------------------------
# SIDEBAR – RESEARCHER & MENU
# --------------------------------------------------
with st.sidebar:
    st.title("Analysis Menu")

    section = st.radio(
        "Select analysis:",
        [
            "1. Dataset overview",
            "2. Descriptive statistics",
            "3. Association analysis (X & Y)",
        ]
    )

    st.markdown("---")
    st.header("Researcher")

    researcher_name = st.text_input("Name", value="Your Name")
    researcher_id = st.text_input("Student ID", value="Your Student ID")
    researcher_class = st.text_input("Class", value="Your Class")
    group_members = st.text_area(
        "Group members (one per line)",
        value="Member 1\nMember 2\nMember 3"
    )

    st.markdown(
        f"""
        **Name:** {researcher_name}  
        **ID:** {researcher_id}  
        **Class:** {researcher_class}
        """
    )

    st.markdown("---")
    st.caption(
        "Upload your survey data (CSV or Excel). "
        "The app will automatically detect Likert-scale items "
        "and compute X_total (sleep quality) and Y_total (academic productivity)."
    )

# --------------------------------------------------
# MAIN TITLE
# --------------------------------------------------
st.title("Analysis of Sleep Quality and Academic Productivity")
st.write(
    "This Streamlit app is built for the Statistics 1 final project. "
    "It computes descriptive statistics and association (correlation) "
    "between sleep quality (X) and academic productivity (Y)."
)

# --------------------------------------------------
# 1. DATA UPLOAD
# --------------------------------------------------
st.header("1. Data upload")

uploaded_file = st.file_uploader(
    "Upload your dataset (CSV or Excel exported from Google Forms):",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is None:
    st.info("Please upload a CSV or Excel file to start the analysis.")
    st.stop()

file_name = uploaded_file.name.lower()

# Read file
if file_name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

# If all columns are "Unnamed", assume the first row is the header
if all(str(c).startswith("Unnamed") for c in df.columns):
    new_header = df.iloc[0].astype(str)      # first row as header
    df = df[1:].reset_index(drop=True)       # drop first row
    df.columns = new_header                  # set column names

st.subheader("Dataset preview (all rows)")
st.dataframe(df)   # show ALL rows

st.markdown("**Column names detected:**")
st.write(list(df.columns))

# --------------------------------------------------
# AUTOMATIC DETECTION OF X & Y ITEMS
# --------------------------------------------------
# 1) detect numeric / numeric-like columns (Likert 1–5)
numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]

if len(numeric_cols) < 2:
    numeric_cols = []
    for col in df.columns:
        try:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().mean() >= 0.7:
                numeric_cols.append(col)
        except Exception:
            continue

if len(numeric_cols) < 2:
    st.error(
        "The app could not detect enough numeric / Likert-type columns.\n\n"
        "Please check that your Likert responses are stored as numbers (1–5) "
        "or as text that can be converted to numbers."
    )
    st.stop()

# 2) identify X items and Y items by prefix (X*, Y*), else split
x_auto = [c for c in numeric_cols if str(c).upper().startswith("X")]
y_auto = [c for c in numeric_cols if str(c).upper().startswith("Y")]

if len(x_auto) == 0 or len(y_auto) == 0:
    half = len(numeric_cols) // 2
    x_auto = numeric_cols[:half]
    y_auto = numeric_cols[half:]

st.markdown("### Automatically detected variables")
st.write(f"Auto X items (sleep quality): {x_auto}")
st.write(f"Auto Y items (academic productivity): {y_auto}")

# -------- VARIABLE SELECTION (ALL or PER VARIABLE) --------
st.markdown("### Variable selection for analysis")

X_COLS = st.multiselect(
    "Select X items to include in analysis (default: all auto-detected X):",
    options=x_auto,
    default=x_auto
)

Y_COLS = st.multiselect(
    "Select Y items to include in analysis (default: all auto-detected Y):",
    options=y_auto,
    default=y_auto
)

if len(X_COLS) == 0 or len(Y_COLS) == 0:
    st.error("Please select at least one X item and one Y item.")
    st.stop()

st.write(f"**X items used in analysis:** {X_COLS}")
st.write(f"**Y items used in analysis:** {Y_COLS}")

# Build Likert dataframe & composite scores for ALL rows using selected items
likert_df = df[X_COLS + Y_COLS].apply(pd.to_numeric, errors="coerce")
df["X_total"] = likert_df[X_COLS].sum(axis=1)
df["Y_total"] = likert_df[Y_COLS].sum(axis=1)

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def descriptive_table(data: pd.DataFrame) -> pd.DataFrame:
    """Return mean, median, mode, min, max, std for all columns (using all rows)."""
    desc = data.describe().T
    mode_vals = data.mode().iloc[0]
    desc["mode"] = mode_vals
    desc_show = desc[["mean", "50%", "mode", "min", "max", "std"]].rename(
        columns={"50%": "median", "std": "std_dev"}
    )
    return desc_show

def cronbach_alpha(df_items: pd.DataFrame) -> float:
    """Cronbach's alpha for a set of items."""
    df_clean = df_items.dropna()
    k = df_clean.shape[1]
    if k < 2:
        return np.nan
    item_variances = df_clean.var(axis=0, ddof=1)
    total_variance = df_clean.sum(axis=1).var(ddof=1)
    if total_variance == 0:
        return np.nan
    alpha = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)
    return alpha

# ==================================================
# SECTION 1 – DATASET OVERVIEW
# ==================================================
if section == "1. Dataset overview":
    st.header("2. Dataset overview")

    st.subheader("Basic information")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Shape (rows, columns):**", df.shape)
        st.write("**Number of X items (selected):**", len(X_COLS))
        st.write("**Number of Y items (selected):**", len(Y_COLS))

    with col2:
        st.write("**Missing values per column:**")
        st.write(df.isna().sum())

    st.subheader("Composite scores (X_total & Y_total) – descriptive statistics")
    comp = descriptive_table(df[["X_total", "Y_total"]])
    st.dataframe(comp.style.format("{:.2f}"))

    c1, c2 = st.columns(2)
    with c1:
        st.write("Histogram of X_total")
        fig, ax = plt.subplots()
        ax.hist(df["X_total"].dropna(), bins=10)
        ax.set_xlabel("X_total")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    with c2:
        st.write("Histogram of Y_total")
        fig2, ax2 = plt.subplots()
        ax2.hist(df["Y_total"].dropna(), bins=10)
        ax2.set_xlabel("Y_total")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)

    st.subheader("Scatter plot of X_total vs Y_total")
    fig3, ax3 = plt.subplots()
    ax3.scatter(df["X_total"], df["Y_total"])
    ax3.set_xlabel("X_total (Sleep quality)")
    ax3.set_ylabel("Y_total (Academic productivity)")
    ax3.set_title("Scatter plot of composite scores")
    st.pyplot(fig3)

# ==================================================
# SECTION 2 – DESCRIPTIVE STATISTICS
# ==================================================
elif section == "2. Descriptive statistics":
    st.header("3. Descriptive statistics")

    # 3.1 Descriptive per item
    st.subheader("3.1 Descriptive statistics – all selected Likert items (X & Y)")
    desc_items = descriptive_table(likert_df)
    st.dataframe(
        desc_items.style.format(
            {"mean": "{:.2f}", "median": "{:.2f}", "mode": "{:.0f}", "std_dev": "{:.2f}"}
        )
    )

    # 3.2 Descriptive composite scores
    st.subheader("3.2 Descriptive statistics – composite scores (X_total & Y_total)")
    desc_comp = descriptive_table(df[["X_total", "Y_total"]])
    st.dataframe(desc_comp.style.format("{:.2f}"))

    # 3.3 Frequency table + multiple plots for one item
    st.markdown("---")
    st.subheader("3.3 Frequency & percentage table for a selected item")

    item = st.selectbox("Choose an item (X or Y):", X_COLS + Y_COLS)
    series = likert_df[item]   # all respondents

    freq = series.value_counts(dropna=False).sort_index()
    percent = (freq / len(series)) * 100

    freq_table = pd.DataFrame({
        "Value": freq.index,
        "Frequency": freq.values,
        "Percent (%)": percent.round(2).values
    })

    st.write("Frequency & percentage table:")
    st.dataframe(freq_table)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.write("Histogram")
        fig_h, ax_h = plt.subplots()
        ax_h.hist(series.dropna(), bins=5)
        ax_h.set_xlabel("Score")
        ax_h.set_ylabel("Frequency")
        st.pyplot(fig_h)

    with c2:
        st.write("Boxplot")
        fig_b, ax_b = plt.subplots()
        ax_b.boxplot(series.dropna(), vert=True)
        ax_b.set_ylabel("Score")
        st.pyplot(fig_b)

    with c3:
        st.write("Pie chart")
        fig_p, ax_p = plt.subplots()
        ax_p.pie(freq.values, labels=freq.index, autopct="%1.1f%%")
        ax_p.axis("equal")
        st.pyplot(fig_p)

    # 3.4 Mean bar chart for all items
    st.markdown("---")
    st.subheader("3.4 Mean score per item (bar chart)")

    item_means = likert_df.mean()
    fig_m, ax_m = plt.subplots()
    ax_m.bar(item_means.index, item_means.values)
    ax_m.set_xticklabels(item_means.index, rotation=45, ha="right")
    ax_m.set_ylabel("Mean score")
    ax_m.set_title("Mean score of each selected Likert item")
    st.pyplot(fig_m)

    # 3.5 Reliability (Cronbach's Alpha)
    st.markdown("---")
    st.subheader("3.5 Reliability (Cronbach's Alpha)")

    alpha_x = cronbach_alpha(likert_df[X_COLS])
    alpha_y = cronbach_alpha(likert_df[Y_COLS])

    st.write(f"**Cronbach's Alpha for X items (sleep quality):** {alpha_x:.3f}")
    st.write(f"**Cronbach's Alpha for Y items (academic productivity):** {alpha_y:.3f}")
    st.caption(
        "Rule of thumb: α ≥ 0.70 is often considered acceptable reliability, "
        "but this depends on context."
    )

    # 3.6 Correlation heatmap of all items + totals
    st.markdown("---")
    st.subheader("3.6 Correlation heatmap (all selected items + totals)")

    corr_matrix = pd.concat(
        [likert_df[X_COLS + Y_COLS], df[["X_total", "Y_total"]]],
        axis=1
    ).corr()

    fig_hm, ax_hm = plt.subplots(figsize=(6, 5))
    im = ax_hm.imshow(corr_matrix, aspect="auto")
    ax_hm.set_xticks(range(len(corr_matrix.columns)))
    ax_hm.set_yticks(range(len(corr_matrix.index)))
    ax_hm.set_xticklabels(corr_matrix.columns, rotation=90)
    ax_hm.set_yticklabels(corr_matrix.index)
    fig_hm.colorbar(im, ax=ax_hm)
    st.pyplot(fig_hm)

# ==================================================
# SECTION 3 – ASSOCIATION ANALYSIS (X & Y)
# ==================================================
elif section == "3. Association analysis (X & Y)":
    st.header("4. Association analysis between X and Y")

    st.write(
        "This section focuses on association between sleep quality and "
        "academic productivity using composite scores X_total and Y_total. "
        "X_total and Y_total are computed from the **selected items** above."
    )

    method = st.selectbox(
        "Choose main correlation method (for the assignment, pick one):",
        ["Pearson (for numeric totals)", "Spearman (rank-based)"]
    )

    valid = df[["X_total", "Y_total"]].dropna()

    if len(valid) < 3:
        st.error("Not enough valid observations for correlation.")
    else:
        if method.startswith("Pearson"):
            r, p = stats.pearsonr(valid["X_total"], valid["Y_total"])
            method_name = "Pearson correlation"
        else:
            r, p = stats.spearmanr(valid["X_total"], valid["Y_total"])
            method_name = "Spearman rank correlation"

        st.subheader(f"{method_name} (X_total vs Y_total)")
        st.write(f"**Correlation coefficient (r):** `{r:.3f}`")
        st.write(f"**p-value:** `{p:.4f}`")

        # Interpretation: direction & strength
        direction = "positive" if r > 0 else "negative"
        abs_r = abs(r)

        if abs_r < 0.20:
            strength = "very weak"
        elif abs_r < 0.40:
            strength = "weak"
        elif abs_r < 0.60:
            strength = "moderate"
        elif abs_r < 0.80:
            strength = "strong"
        else:
            strength = "very strong"

        st.markdown(f"- **Direction:** {direction}")
        st.markdown(f"- **Strength:** {strength}")

        if p < 0.05:
            st.success("The association is **statistically significant** (p < 0.05).")
        else:
            st.info("The association is **not statistically significant** (p ≥ 0.05).")

        # ---- SUMMARY kalimat: “kalau tidur kurang, produktivitas bagaimana?” ----
        st.markdown("---")
        st.subheader("Interpretation summary")

        # split X_total into low / medium / high
        q = valid["X_total"].quantile([0.33, 0.66])
        bins = [-np.inf, q.iloc[0], q.iloc[1], np.inf]
        labels = ["Low sleep quality", "Medium", "High sleep quality"]
        valid["X_group"] = pd.cut(valid["X_total"], bins=bins, labels=labels)

        group_means = valid.groupby("X_group")["Y_total"].mean()

        st.write("Average academic productivity (Y_total) by sleep quality group:")
        st.dataframe(group_means.round(2))

        if "Low sleep quality" in group_means.index and "High sleep quality" in group_means.index:
            low_y = group_means.loc["Low sleep quality"]
            high_y = group_means.loc["High sleep quality"]
            diff = high_y - low_y

            if diff > 0:
                tendency = "students who sleep better tend to have **higher** academic productivity."
            elif diff < 0:
                tendency = "students who sleep better tend to have **lower** academic productivity."
            else:
                tendency = "academic productivity is **about the same** between low and high sleep quality."

            st.markdown(
                f"- On average, students with **high sleep quality** have Y_total ≈ `{high_y:.2f}`, "
                f"while students with **low sleep quality** have Y_total ≈ `{low_y:.2f}` "
                f"(difference ≈ `{diff:.2f}`)."
            )
            st.markdown(
                f"- In simple words: **{tendency}**"
            )

        st.markdown("---")
        st.subheader("Scatter plot of X_total vs Y_total")

        fig, ax = plt.subplots()
        ax.scatter(valid["X_total"], valid["Y_total"])
        ax.set_xlabel("X_total (Sleep quality)")
        ax.set_ylabel("Y_total (Academic productivity)")
        ax.set_title("Scatter plot for association analysis")
        st.pyplot(fig)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption(
    f"Statistics 1 final project – Sleep Quality & Academic Productivity. "
    f"Prepared by {researcher_name} ({researcher_id}, {researcher_class})."
)
st.caption("Group members:\n" + group_members)
