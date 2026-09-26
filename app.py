import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

st.set_page_config(
    page_title="Data Drift Early Warning System",
    page_icon="📊",
    layout="wide"
)

# ---------------- CSS ----------------

st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.main-title {
    font-size: 38px;
    font-weight: 700;
    color: #111827;
}

.subtitle {
    color: #6b7280;
    font-size: 16px;
}

.online {
    background-color: #dcfce7;
    color: #15803d;
    padding: 8px 15px;
    border-radius: 20px;
    font-weight: 600;
    text-align: center;
}

.card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    border: 1px solid #e5e7eb;
}

.card-title {
    color: #6b7280;
    font-size: 14px;
}

.card-value {
    color: #111827;
    font-size: 30px;
    font-weight: 700;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    color: #111827;
    margin-top: 15px;
}

.devops-box {
    background-color: #1f2937;
    padding: 15px;
    border-radius: 12px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- Functions ----------------

def drift_status(score):
    if score < 0.30:
        return "🟢 Normal"
    elif score < 0.60:
        return "🟡 Warning"
    else:
        return "🔴 Drift"


def numeric_drift(old, new):
    old = old.dropna()
    new = new.dropna()

    if len(old) == 0 or len(new) == 0:
        return 0

    score, _ = ks_2samp(old, new)
    return round(float(score), 3)


def categorical_drift(old, new):
    old_dist = old.value_counts(normalize=True)
    new_dist = new.value_counts(normalize=True)

    categories = set(old_dist.index).union(set(new_dist.index))

    score = 0

    for category in categories:
        p = old_dist.get(category, 0)
        q = new_dist.get(category, 0)
        score += abs(p - q)

    return round(float(score / 2), 3)


def calculate_drift(train, new):
    results = []

    common_columns = [
        column for column in train.columns
        if column in new.columns
    ]

    for column in common_columns:

        if (
            pd.api.types.is_numeric_dtype(train[column])
            and
            pd.api.types.is_numeric_dtype(new[column])
        ):
            score = numeric_drift(
                train[column],
                new[column]
            )
            data_type = "Numerical"

        else:
            score = categorical_drift(
                train[column].astype(str),
                new[column].astype(str)
            )
            data_type = "Categorical"

        results.append({
            "Feature": column,
            "Type": data_type,
            "Drift Score": score,
            "Status": drift_status(score)
        })

    return pd.DataFrame(results)


# ---------------- Sidebar ----------------

with st.sidebar:

    st.markdown(
        "<h2>📊 Data Drift</h2>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='color:#9ca3af;'>Early Warning System</p>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📤 Upload & Analyze",
            "📈 Feature Analysis",
            "ℹ️ About"
        ]
    )

    st.markdown("---")

    st.markdown("### ⚙️ DevOps Pipeline")

    st.markdown("""
    <div class="devops-box">
    🔧 Git<br><br>
    🐙 GitHub<br><br>
    🔄 Jenkins<br><br>
    🐳 Docker<br><br>
    ☸️ Kubernetes
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.caption("AI + DevOps Project")
    st.caption("Streamlit • Python • Scikit-learn")


# ---------------- Header ----------------

if page != "ℹ️ About":

    col1, col2 = st.columns([5, 1])

    with col1:
        st.markdown(
            '<div class="main-title">Data Drift Early Warning System</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="subtitle">'
            'Monitor changes in incoming data before they affect ML model performance.'
            '</div>',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            '<div class="online">● System Online</div>',
            unsafe_allow_html=True
        )


# ---------------- Upload Page ----------------

if page == "📤 Upload & Analyze":

    st.markdown(
        '<div class="section-title">📤 Upload & Analyze Data</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload the original training data and the new incoming data."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 📁 Training Data")

        train_file = st.file_uploader(
            "Upload training CSV",
            type=["csv"],
            key="train_upload"
        )

    with col2:

        st.markdown("### 📁 New Data")

        new_file = st.file_uploader(
            "Upload new CSV",
            type=["csv"],
            key="new_upload"
        )

    if train_file and new_file:

        train_data = pd.read_csv(train_file)
        new_data = pd.read_csv(new_file)

        # FIXED SESSION STATE
        st.session_state["train_data"] = train_data
        st.session_state["new_data"] = new_data

        st.success("Both datasets uploaded successfully.")

        st.markdown("### Training Data Preview")

        st.dataframe(
            train_data.head(),
            use_container_width=True
        )

        st.markdown("### New Data Preview")

        st.dataframe(
            new_data.head(),
            use_container_width=True
        )

        if st.button(
            "🔍 Analyze Data Drift",
            use_container_width=True
        ):

            result = calculate_drift(
                train_data,
                new_data
            )

            st.session_state["result"] = result

            st.success("Drift analysis completed!")

    else:

        st.info(
            "Upload both Training Data and New Data to start the analysis."
        )


# ---------------- Dashboard ----------------

if page == "🏠 Dashboard":

    if (
        "train_data" not in st.session_state
        or
        "new_data" not in st.session_state
    ):

        st.info(
            "👈 Go to **Upload & Analyze** and upload your two CSV files."
        )

        st.markdown("### 📌 How the system works")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown("""
            <div class="card">
            <b>1️⃣ Upload</b><br>
            Training and new data
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div class="card">
            <b>2️⃣ Compare</b><br>
            Compare common features
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown("""
            <div class="card">
            <b>3️⃣ Detect</b><br>
            Calculate drift score
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown("""
            <div class="card">
            <b>4️⃣ Alert</b><br>
            Show warning
            </div>
            """, unsafe_allow_html=True)

    else:

        train_data = st.session_state["train_data"]
        new_data = st.session_state["new_data"]

        if "result" not in st.session_state:

            st.session_state["result"] = calculate_drift(
                train_data,
                new_data
            )

        result = st.session_state["result"]

        overall_drift = result["Drift Score"].max()


        # ---------------- KPI Cards ----------------

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
            <div class="card">
            <div class="card-title">Training Records</div>
            <div class="card-value">{len(train_data):,}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="card">
            <div class="card-title">New Records</div>
            <div class="card-value">{len(new_data):,}</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="card">
            <div class="card-title">Features Checked</div>
            <div class="card-value">{len(result)}</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="card">
            <div class="card-title">Overall Drift</div>
            <div class="card-value">{overall_drift:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")


        # ---------------- Alert ----------------

        if overall_drift >= 0.60:

            st.error(
                "🚨 DATA DRIFT DETECTED — "
                "Significant changes were found in incoming data."
            )

        elif overall_drift >= 0.30:

            st.warning(
                "⚠️ WARNING — "
                "Some features show noticeable data changes."
            )

        else:

            st.success(
                "✅ DATA NORMAL — "
                "No significant drift detected."
            )


        # ---------------- Feature Table ----------------

        st.markdown(
            '<div class="section-title">📊 Feature Analysis</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True
        )


        # ---------------- Chart ----------------

        st.markdown(
            '<div class="section-title">'
            '📈 Feature Distribution Comparison'
            '</div>',
            unsafe_allow_html=True
        )

        chart_data = result.set_index(
            "Feature"
        )["Drift Score"]

        fig, ax = plt.subplots(
            figsize=(10, 4)
        )

        chart_data.plot(
            kind="bar",
            ax=ax
        )

        ax.set_ylabel("Drift Score")
        ax.set_xlabel("Features")
        ax.set_ylim(0, 1)

        ax.axhline(
            0.30,
            linestyle="--",
            label="Warning"
        )

        ax.axhline(
            0.60,
            linestyle="--",
            label="Drift"
        )

        ax.legend()

        st.pyplot(fig)


        # ---------------- Threshold ----------------

        st.markdown("### 🎯 Drift Thresholds")

        t1, t2, t3 = st.columns(3)

        with t1:
            st.success(
                "🟢 0.00 – 0.29\n\nNormal"
            )

        with t2:
            st.warning(
                "🟡 0.30 – 0.59\n\nWarning"
            )

        with t3:
            st.error(
                "🔴 0.60 – 1.00\n\nDrift"
            )


        # ---------------- Recommendation ----------------

        st.markdown(
            "### 💡 Key Insights & Recommendations"
        )

        if overall_drift >= 0.60:

            st.error(
                "Significant drift detected. "
                "Check the affected features and "
                "consider retraining the ML model."
            )

        elif overall_drift >= 0.30:

            st.warning(
                "Moderate drift detected. "
                "Monitor the affected features and "
                "investigate the source of the change."
            )

        else:

            st.success(
                "Data distribution is stable. "
                "Continue monitoring incoming data."
            )


# ---------------- Feature Analysis ----------------

if page == "📈 Feature Analysis":

    st.markdown(
        '<div class="main-title">Feature Analysis</div>',
        unsafe_allow_html=True
    )

    if "train_data" not in st.session_state:

        st.info(
            "Upload data first from the Upload & Analyze page."
        )

    else:

        train_data = st.session_state["train_data"]
        new_data = st.session_state["new_data"]

        result = st.session_state.get(
            "result",
            calculate_drift(
                train_data,
                new_data
            )
        )

        feature = st.selectbox(
            "Select Feature",
            result["Feature"].tolist()
        )

        old = train_data[feature]
        new = new_data[feature]

        score = result[
            result["Feature"] == feature
        ]["Drift Score"].iloc[0]

        st.markdown(
            f"### {feature} — Drift Score: **{score:.3f}**"
        )


        if pd.api.types.is_numeric_dtype(old):

            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            ax.hist(
                old.dropna(),
                bins=20,
                alpha=0.6,
                label="Training Data"
            )

            ax.hist(
                new.dropna(),
                bins=20,
                alpha=0.6,
                label="New Data"
            )

            ax.set_xlabel(feature)
            ax.set_ylabel("Frequency")
            ax.set_title(
                "Training vs New Data"
            )

            ax.legend()

            st.pyplot(fig)

        else:

            old_counts = old.astype(
                str
            ).value_counts(
                normalize=True
            )

            new_counts = new.astype(
                str
            ).value_counts(
                normalize=True
            )

            categories = list(
                set(old_counts.index).union(
                    set(new_counts.index)
                )
            )

            compare = pd.DataFrame(
                {
                    "Training": [
                        old_counts.get(
                            x,
                            0
                        )
                        for x in categories
                    ],

                    "New Data": [
                        new_counts.get(
                            x,
                            0
                        )
                        for x in categories
                    ]
                },
                index=categories
            )

            st.bar_chart(compare)


# ---------------- About ----------------

if page == "ℹ️ About":

    st.markdown(
        '<div class="main-title">'
        'About the Project'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    ### 🎯 Data Drift Early Warning System

    This project detects changes between **training data**
    and **new incoming data**.

    ML models are trained using historical data.
    When real-world data changes, model performance
    can decrease.

    This system gives an early warning when significant
    data drift is detected.
    """)


    st.markdown("### 🔄 System Workflow")

    st.code("""
Training Data
      ↓
New Incoming Data
      ↓
Compare Features
      ↓
Calculate Drift Score
      ↓
Normal / Warning / Drift
      ↓
Recommendation
      ↓
Model Monitoring / Retraining
""")


    st.markdown("### 🛠️ Technologies Used")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.info("🐍 Python")

    with c2:
        st.info("📊 Pandas")

    with c3:
        st.info("🎨 Streamlit")

    with c4:
        st.info("🤖 Scikit-learn")


    st.markdown("### 🚀 DevOps Workflow")

    st.markdown("""
    **Git → GitHub → Jenkins → Docker → Kubernetes → Streamlit**

    - **Git:** Track code changes
    - **GitHub:** Store project remotely
    - **Jenkins:** Automate build and testing
    - **Docker:** Create application container
    - **Kubernetes:** Manage containers
    - **Streamlit:** Display the monitoring dashboard
    """)


    st.markdown("### 📌 Project Objective")

    st.success(
        "Detect significant changes in incoming data early "
        "so that the ML model can be checked or retrained "
        "before its performance decreases."
    )