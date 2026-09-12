import streamlit as st
import pandas as pd
import plotly.express as px


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="GridGuard AI",
    page_icon="⚡",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("data/predictions.csv")


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚡ GridGuard AI")

st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "About"]
)

st.sidebar.markdown("---")


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.title("⚡ GridGuard AI")

    st.subheader(
        "Smart Electricity Consumption Anomaly Detection"
    )

    st.write(
        "Identify unusual electricity consumption patterns "
        "using Machine Learning."
    )

    # -----------------------------------------------------
    # SIDEBAR FILTERS
    # -----------------------------------------------------

    st.sidebar.subheader("Filters")

    risk_filter = st.sidebar.selectbox(
        "Risk Level",
        ["All", "Low", "Medium", "High"]
    )

    consumer_filter = st.sidebar.selectbox(
        "Consumer ID",
        ["All"] + sorted(
            df["consumer_id"].unique().tolist()
        )
    )

    # -----------------------------------------------------
    # APPLY FILTERS
    # -----------------------------------------------------

    filtered_df = df.copy()

    if risk_filter != "All":

        filtered_df = filtered_df[
            filtered_df["risk_level"] == risk_filter
        ]

    if consumer_filter != "All":

        filtered_df = filtered_df[
            filtered_df["consumer_id"] == consumer_filter
        ]

    # -----------------------------------------------------
    # MODEL DETAILS BUTTON
    # -----------------------------------------------------

    show_model = st.button(
        "🤖 Model Details"
    )

    if show_model:

        st.info(
            """
            **Machine Learning Model: Isolation Forest**

            **Learning Type:** Unsupervised Learning

            **Purpose:** Electricity consumption anomaly detection

            **Why Isolation Forest?**

            Confirmed electricity theft labels are difficult to
            obtain. Therefore, this project uses an unsupervised
            learning algorithm to identify unusual consumption
            patterns without requiring labeled theft data.

            **Features Used:**

            • Current consumption  
            • Previous consumption  
            • Consumption change  
            • Percentage change  
            • Average consumption  
            • Consumption standard deviation  
            • Minimum consumption  
            • Maximum consumption  

            **Output:**

            The model produces an anomaly score. This score is
            converted into a 0–100 anomaly risk score and then
            classified as Low, Medium, or High risk.

            **Important:**

            A suspicious reading does not confirm electricity
            theft. It indicates an unusual consumption pattern
            that may require further investigation.
            """
        )

        st.markdown("---")

        model_col1, model_col2, model_col3 = st.columns(3)

        model_col1.metric(
            "Algorithm",
            "Isolation Forest"
        )

        model_col2.metric(
            "Learning",
            "Unsupervised"
        )

        model_col3.metric(
            "Contamination",
            "5%"
        )

    # -----------------------------------------------------
    # KPI CALCULATIONS
    # -----------------------------------------------------

    total_consumers = df["consumer_id"].nunique()

    total_readings = len(df)

    suspicious_readings = len(
        df[df["status"] == "Suspicious"]
    )

    high_risk = len(
        df[df["risk_level"] == "High"]
    )

    suspicious_consumers_count = df[
        df["status"] == "Suspicious"
    ]["consumer_id"].nunique()

    # -----------------------------------------------------
    # KPI DISPLAY
    # -----------------------------------------------------

    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Total Consumers",
        total_consumers
    )

    col2.metric(
        "Total Readings",
        total_readings
    )

    col3.metric(
        "Suspicious Readings",
        suspicious_readings
    )

    col4.metric(
        "High Risk Readings",
        high_risk
    )

    col5.metric(
        "Suspicious Consumers",
        suspicious_consumers_count
    )

    st.markdown("---")

    # =====================================================
    # CONSUMPTION DISTRIBUTION
    # =====================================================

    st.subheader(
        "📊 Electricity Consumption Distribution"
    )

    fig1 = px.histogram(
        filtered_df,
        x="consumption_kwh",
        nbins=40,
        title="Consumption Distribution"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # =====================================================
    # MONTHLY CONSUMPTION
    # =====================================================

    st.subheader(
        "📈 Average Monthly Consumption"
    )

    monthly_usage = (
        filtered_df
        .groupby("month")["consumption_kwh"]
        .mean()
        .reset_index()
    )

    fig2 = px.line(
        monthly_usage,
        x="month",
        y="consumption_kwh",
        markers=True,
        title="Average Monthly Consumption"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # =====================================================
    # RISK DISTRIBUTION
    # =====================================================

    st.subheader(
        "🚦 Risk Level Distribution"
    )

    risk_counts = (
        filtered_df["risk_level"]
        .value_counts()
        .reset_index()
    )

    risk_counts.columns = [
        "risk_level",
        "count"
    ]

    fig3 = px.bar(
        risk_counts,
        x="risk_level",
        y="count",
        title="Risk Level Distribution"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    # =====================================================
    # CONSUMPTION READINGS
    # =====================================================

    st.subheader(
        "📋 Consumption Readings"
    )

    consumption_table = filtered_df.sort_values(
        "risk_score",
        ascending=False
    )

    st.dataframe(
        consumption_table[
            [
                "consumer_id",
                "month",
                "consumption_kwh",
                "percentage_change",
                "risk_score",
                "risk_level",
                "status"
            ]
        ],
        use_container_width=True
    )

    # =====================================================
    # CONSUMER ANALYSIS
    # =====================================================

    st.subheader(
        "👤 Consumer Analysis"
    )

    selected_consumer = st.selectbox(
        "Select Consumer",
        sorted(
            df["consumer_id"].unique()
        )
    )

    consumer_data = df[
        df["consumer_id"] == selected_consumer
    ].sort_values("month")

    fig4 = px.line(
        consumer_data,
        x="month",
        y="consumption_kwh",
        markers=True,
        title=f"Consumer {selected_consumer} Monthly Consumption"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    # -----------------------------------------------------
    # CONSUMER DETAILS
    # -----------------------------------------------------

    st.write("Consumer Details")

    st.dataframe(
        consumer_data[
            [
                "month",
                "consumption_kwh",
                "percentage_change",
                "risk_score",
                "risk_level",
                "status"
            ]
        ],
        use_container_width=True
    )

    # =====================================================
    # SUSPICIOUS CONSUMERS
    # =====================================================

    st.subheader(
        "🚨 Suspicious Consumers"
    )

    suspicious_consumers = (
        df[df["status"] == "Suspicious"]
        .groupby("consumer_id")
        .agg(
            suspicious_readings=("status", "count"),
            average_risk=("risk_score", "mean"),
            maximum_risk=("risk_score", "max")
        )
        .reset_index()
    )

    suspicious_consumers = (
        suspicious_consumers
        .sort_values(
            "maximum_risk",
            ascending=False
        )
    )

    suspicious_consumers["average_risk"] = (
        suspicious_consumers["average_risk"]
        .round(2)
    )

    suspicious_consumers["maximum_risk"] = (
        suspicious_consumers["maximum_risk"]
        .round(2)
    )

    st.dataframe(
        suspicious_consumers,
        use_container_width=True
    )

    # =====================================================
    # CSV DOWNLOAD
    # =====================================================

    csv_data = suspicious_consumers.to_csv(
        index=False
    )

    st.download_button(
        label="📥 Download Suspicious Consumers",
        data=csv_data,
        file_name="suspicious_consumers.csv",
        mime="text/csv"
    )

    # =====================================================
    # IMPORTANT NOTE
    # =====================================================

    st.markdown("---")

    st.warning(
        """
        ⚠️ **Important:** A suspicious reading does not
        confirm electricity theft. GridGuard AI identifies
        unusual consumption patterns that may require
        further investigation.
        """
    )


# =========================================================
# ABOUT PAGE
# =========================================================

elif page == "About":

    st.title("⚡ About GridGuard AI")

    st.subheader(
        "Smart Electricity Consumption Anomaly Detection"
    )

    st.write(
        """
        GridGuard AI is an unsupervised Machine Learning
        project designed to identify unusual electricity
        consumption patterns.
        """
    )

    st.markdown("---")

    # -----------------------------------------------------
    # HOW IT WORKS
    # -----------------------------------------------------

    st.subheader("🔍 How the Model Works")

    st.write(
        """
        1. Electricity consumption data is generated or collected.

        2. Historical consumption features are created.

        3. Isolation Forest analyzes the consumption patterns.

        4. Anomaly scores are generated.

        5. Anomaly scores are converted into a 0–100 risk score.

        6. Risk levels are classified as Low, Medium, or High.

        7. Results are displayed through the Streamlit dashboard.
        """
    )

    # -----------------------------------------------------
    # MODEL DETAILS
    # -----------------------------------------------------

    st.subheader("🤖 Machine Learning Model")

    st.write(
        """
        **Algorithm:** Isolation Forest

        **Learning Type:** Unsupervised Learning

        **Purpose:** Detect unusual electricity consumption
        patterns.

        Isolation Forest is useful for anomaly detection because
        it can identify unusual observations without requiring
        labeled examples.
        """
    )

    # -----------------------------------------------------
    # FEATURES
    # -----------------------------------------------------

    st.subheader("📌 Features Used")

    st.write(
        """
        • consumption_kwh

        • previous_consumption

        • consumption_change

        • percentage_change

        • avg_consumption

        • std_consumption

        • min_consumption

        • max_consumption
        """
    )

    # -----------------------------------------------------
    # TECHNOLOGY STACK
    # -----------------------------------------------------

    st.subheader("🛠️ Technology Stack")

    st.write(
        """
        Python  
        Pandas  
        NumPy  
        Scikit-learn  
        Isolation Forest  
        Plotly  
        Streamlit  
        Joblib
        """
    )

    # -----------------------------------------------------
    # PROJECT PIPELINE
    # -----------------------------------------------------

    st.subheader("🔄 Project Pipeline")

    st.code(
        """
Electricity Data
       ↓
Data Preprocessing
       ↓
Feature Engineering
       ↓
Isolation Forest
       ↓
Anomaly Score
       ↓
Risk Score
       ↓
Risk Level
       ↓
GridGuard Dashboard
        """
    )

    # -----------------------------------------------------
    # IMPORTANT NOTE
    # -----------------------------------------------------

    st.subheader("⚠️ Important Note")

    st.warning(
        """
        GridGuard AI does not confirm electricity theft.

        A suspicious reading only means that the consumption
        pattern is unusual and may require further investigation.

        Factors such as vacation, low occupancy, seasonal usage,
        equipment changes, or other legitimate reasons can also
        cause unusual consumption.
        """
    )