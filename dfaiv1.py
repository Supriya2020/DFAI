import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px
import io
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew, kurtosis
import seaborn as sns
import matplotlib.pyplot as plt
st.set_page_config(layout="wide")

# Attempt to import TensorFlow and set availability flag
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.optimizers import Adam
    import tensorflow as tf
    if tf.__version__.startswith('2'):
        TENSORFLOW_AVAILABLE = True
    else:
        TENSORFLOW_AVAILABLE = False
        st.warning("TensorFlow 2.x not detected. Deep Neural Network model will not be available.")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.warning("TensorFlow not installed. Deep Neural Network model will not be available.")


st.title("📊 Federated Learning with ML/DL Models")

# Helper function to get model weights (for Keras)
def get_model_weights(model):
    return model.get_weights()

# Helper function to set model weights (for Keras)
def set_model_weights(model, weights):
    model.set_weights(weights)

# Helper function to create a new Keras model with the same architecture
def create_keras_model(input_dim, num_classes, learning_rate):
    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    # Compile with sparse_categorical_crossentropy by default, can be changed later if y is one-hot
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

steps = [
    "1. Upload Data", "2. Select Variables", "3. Feature Preprocessing",
    "4. Feature Processing + validations", "5. Run Federated Learning",
    "6. Results for paper"
]

step = st.sidebar.radio("Navigation", steps)

# Session state initialization
for key in ["data", "categorical", "continuous", "target", "exclude",
            "processed_X", "processed_y", "target_classes",
            "selected_model_name", "fl_params", "X_test_global", "y_test_global", "X_train_global_scaled",
            "comparison_results"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Initialize comparison_results as a list of dictionaries if not present
if st.session_state.comparison_results is None:
    st.session_state.comparison_results = []

# Step 0: Upload Data
if step == steps[0]:
    st.header("📤 Upload Your Data")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        try:
            df = pd.read_csv(file)
            st.session_state.data = df
            st.write("Preview of your data (first 5 rows):")
            st.dataframe(df.head())
            st.success("Data uploaded successfully! Proceed to '2. Select Variables'.")
            # Clear dependent states if new data is uploaded
            for key_to_clear in ["categorical", "continuous", "target", "exclude",
                                 "processed_X", "processed_y", "target_classes",
                                 "X_test_global", "y_test_global", "X_train_global_scaled"]:
                st.session_state[key_to_clear] = None
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")


# Step 1: Select variables
elif step == steps[1]:
    st.header("🔍 Select Variables")
    if st.session_state.data is None:
        st.warning("Please upload data first in '1. Upload Data'.")
        st.stop()
    
    df_current = st.session_state.data.copy() # Use a copy for selection to avoid issues if user navigates back
    cols = df_current.columns.tolist()
    
    # Initialize selections if they are None or if columns changed (e.g. after exclusion)
    if st.session_state.target not in cols:
        st.session_state.target = None
    
    st.session_state.target = st.selectbox(
        "🎯 Select Target Variable (Y)", cols, 
        index=cols.index(st.session_state.target) if st.session_state.target else 0,
        help="This is the variable your model will predict."
    )
    
    feature_cols = [col for col in cols if col != st.session_state.target]

    st.session_state.categorical = st.multiselect(
        "🏷️ Categorical Features", feature_cols, 
        default=[col for col in feature_cols if df_current[col].dtype == 'object' and col in (st.session_state.categorical or [])] or \
                [col for col in feature_cols if df_current[col].dtype == 'object'],
        help="Select variables that represent categories (e.g., 'Gender', 'City')."
    )
    st.session_state.continuous = st.multiselect(
        "🔢 Continuous Features", feature_cols, 
        default=[col for col in feature_cols if df_current[col].dtype in ['int64', 'float64'] and col in (st.session_state.continuous or [])] or \
                [col for col in feature_cols if df_current[col].dtype in ['int64', 'float64']],
        help="Select variables that represent numerical measurements (e.g., 'Age', 'Income')."
    )
    st.session_state.exclude = st.multiselect(
        "🚫 Exclude Variables", cols, 
        default=st.session_state.exclude if st.session_state.exclude else [],
        help="Select variables you want to completely remove from the dataset."
    )

    if st.button("Apply Variable Selections"):
        temp_df = st.session_state.data.copy() # Operate on a copy of the original uploaded data
        if st.session_state.exclude:
            temp_df.drop(columns=st.session_state.exclude, inplace=True, errors='ignore')
            st.session_state.data_after_exclusion = temp_df # Store this intermediate state
            st.success(f"Variables excluded: {st.session_state.exclude}. The dataset for preprocessing now has {temp_df.shape[1]} columns.")
        else:
            st.session_state.data_after_exclusion = temp_df # No exclusion, use the original
            st.info("No variables were excluded.")
        
        # Validate selections
        if not st.session_state.target or st.session_state.target in st.session_state.exclude:
            st.error("Target variable must be selected and cannot be in the excluded list.")
        elif not (st.session_state.categorical or st.session_state.continuous):
            st.error("Please select at least one categorical or continuous feature.")
        else:
            st.success("Variable selections applied. Proceed to '3. Feature Preprocessing'.")
            # Clear downstream processing states
            st.session_state.processed_X = None
            st.session_state.processed_y = None
# Step 2: Fature Processing
elif step == steps[2]:
    st.subheader("🔍 Feature Processing + validations")
    df = st.session_state.data.copy()
    target = st.session_state.target
    categorical = st.session_state.categorical
    continuous = st.session_state.continuous
    exclude = st.session_state.exclude

    # Step 1: Drop excluded columns
    df = df.drop(columns=exclude)

    # Step 2: Separate target
    y = df[target]
    X = df.drop(columns=[target])

    # Step 3: Encode categorical features
    label_encoders = {}
    for col in categorical:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Step 4: Feature extraction for continuous variables
    st.subheader("📊 Continuous Feature Statistics")
    stats_table = pd.DataFrame()

    for col in continuous:
        stats_table[col] = {
            "Mean": X[col].mean(),
            "Std Dev": X[col].std(),
            "Min": X[col].min(),
            "Max": X[col].max(),
            "Skewness": skew(X[col]),
            "Kurtosis": kurtosis(X[col])
        }

    st.dataframe(stats_table.T.style.format("{:.4f}"))

    # Step 5: Correlation heatmap
    st.subheader("🔗 Correlation Heatmap")
    corr_matrix = X[continuous + categorical].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    # Step 6: Distributions of continuous features
    st.subheader("📈 Distributions of Continuous Features")
    for col in continuous:
        fig, ax = plt.subplots()
        sns.histplot(X[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

    # Step 7: Boxplots for outlier detection
    st.subheader("🧪 Boxplots for Outlier Detection")
    for col in continuous:
        fig, ax = plt.subplots()
        sns.boxplot(x=X[col], ax=ax)
        ax.set_title(f"Boxplot of {col}")
        st.pyplot(fig)
#step 3: Feature processing and validation
elif step == steps[3]:

    st.subheader("🔍 Feature Processing + validations")
    df = st.session_state.data.copy()
    target = st.session_state.target
    categorical = st.session_state.categorical
    continuous = st.session_state.continuous
    exclude = st.session_state.exclude

    # Step 1: वगळलेले कॉलम काढा
    df = df.drop(columns=exclude)

    # Step 2: Target आणि features वेगळे करा
    y = df[target]
    X = df.drop(columns=[target])

    # Step 3: वैशिष्ट्य प्रक्रिया + विश्लेषण
    st.subheader("📊 Step 3: Feature Processing and Analysis")

    # 3.1 Categorical Features Encode करा
    st.markdown("#### 🏷️ Categorical Features Encoding")
    for col in categorical:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        st.success(f"✅ Encoded: {col}")

    # 3.2 Continuous Features चे विश्लेषण
    st.markdown("#### 🔍 Continuous Feature Insights")

    stats_summary = []
    for col in continuous:
        vals = X[col]
        stats_summary.append({
            "Feature": col,
            "Mean": vals.mean(),
            "Std": vals.std(),
            "Min": vals.min(),
            "Max": vals.max(),
            "Skew": skew(vals),
            "Kurtosis": kurtosis(vals)
        })

        # Histogram
        fig_hist, ax_hist = plt.subplots()
        sns.histplot(vals, kde=True, ax=ax_hist)
        ax_hist.set_title(f"Distribution of {col}")
        st.pyplot(fig_hist)

        # Boxplot
        fig_box, ax_box = plt.subplots()
        sns.boxplot(x=vals, ax=ax_box)
        ax_box.set_title(f"Boxplot of {col}")
        st.pyplot(fig_box)

        # Intelligent टिप्पण्या (Insight)
        skewness = skew(vals)
        kurt = kurtosis(vals)
        if kurt > 50 and skewness > 2:
            st.warning(f"⚠️ {col} हे अत्यंत skewed आणि kurtotic आहे — शक्यतो binary/anomaly indicator. Scale करू नका.")
        elif skewness > 1.5:
            st.info(f"ℹ️ {col} मध्ये थोडा skew आहे. Log किंवा sqrt transformation उपयोगी ठरू शकतो.")
        else:
            st.success(f"✅ {col} हे symmetric आहे आणि ML साठी योग्य आहे.")

    # 3.3 सारांश टेबल
    st.markdown("#### 📋 Continuous Features Summary")
    st.dataframe(pd.DataFrame(stats_summary).set_index("Feature").style.format("{:.4f}"))

    # 3.4 Correlation Heatmap
    st.markdown("#### 🔗 Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    sns.heatmap(X[continuous + categorical].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)

    # Process झालेला डेटा पुढील स्टेपसाठी साठवा
    st.session_state.processed_X = X
    st.session_state.processed_y = y
#step 4: Run the code
elif step == steps[4]:
    if 'processed_X' not in st.session_state or 'processed_y' not in st.session_state:
        st.error("❌ Please upload and preprocess your dataset first. `processed_X` and `processed_y` not found in session state.")
        st.stop()

# ---- Sidebar UI ----
    MODEL_OPTIONS = [
        "Random Forest", "Logistic Regression", "SVM", "Naive Bayes",
        "Decision Tree", "KNN", "Gradient Boosting", "AdaBoost"
    ]

    AVERAGING_POLICIES = [
        "Simple Average",
        "Weighted Average (by data size)",
        "Dynamic Averaging (mean Q-value based)",
        "Dynamic Averaging (variance-based)",
        "Hybrid Averaging (mean * inverse variance)",
        "Confidence-Weighted Adaptive Averaging (CWAA)",
        "Dynamic Federated Averaging with Intelligence (DFAI)"
    ]

    st.sidebar.header("🧠 Federated Learning Simulation")
    selected_models = st.sidebar.multiselect("Select Models", MODEL_OPTIONS, default=["Random Forest"])
    selected_averagings = st.sidebar.multiselect("Select Averaging Methods", AVERAGING_POLICIES, default=["Simple Average"])
    clients = st.sidebar.slider("Number of Clients", 2, 10, 5)
    rounds = st.sidebar.slider("Federated Rounds", 1, 20, 5)
    epochs = st.sidebar.slider("Local Epochs", 1, 10, 3)
    learning_rate = st.sidebar.number_input("Learning Rate (for DNN, optional)", min_value=0.0001, max_value=1.0, value=0.01, step=0.01, format="%f")

    st.header("🚀 Federated Learning with Real-Time Feedback")

    # ---- Data preparation ----
    X = st.session_state.processed_X
    y = st.session_state.processed_y

    X_train_global, X_test_global, y_train_global, y_test_global = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train_global_scaled = scaler.fit_transform(X_train_global)
    X_test_global_scaled = scaler.transform(X_test_global)

    X_parts = np.array_split(X_train_global_scaled, clients)
    y_parts = np.array_split(y_train_global, clients)

    # ---- Model factory ----
    def get_model_instance(name, seed):
        if name == "Random Forest": return RandomForestClassifier(random_state=seed)
        elif name == "Logistic Regression": return LogisticRegression(solver='liblinear', max_iter=1000)
        elif name == "SVM": return SVC(probability=True)
        elif name == "Naive Bayes": return GaussianNB()
        elif name == "Decision Tree": return DecisionTreeClassifier(random_state=seed)
        elif name == "KNN": return KNeighborsClassifier()
        elif name == "Gradient Boosting": return GradientBoostingClassifier(random_state=seed)
        elif name == "AdaBoost": return AdaBoostClassifier(random_state=seed)

    # ---- Aggregation ----
    def aggregate_predictions(client_probs, method, client_data_sizes, client_models):
        if not client_probs:
            return None
        if method == "Simple Average":
            weights = np.ones(len(client_probs)) / len(client_probs)
        elif method == "Weighted Average (by data size)":
            weights = np.array(client_data_sizes) / sum(client_data_sizes)
        elif method == "Dynamic Averaging (mean Q-value based)":
            weights = np.array([np.mean(np.max(p, axis=1)) for p in client_probs])
            weights /= weights.sum()
        elif method == "Dynamic Averaging (variance-based)":
            weights = np.array([1 / (np.mean(np.var(p, axis=1)) + 1e-6) for p in client_probs])
            weights /= weights.sum()
        elif method == "Hybrid Averaging (mean * inverse variance)":
            weights = np.array([np.mean(np.max(p, axis=1)) / (np.mean(np.var(p, axis=1)) + 1e-6) for p in client_probs])
            weights /= weights.sum()
        elif method == "Confidence-Weighted Adaptive Averaging (CWAA)":
            weights = np.array([
                np.mean(np.max(p, axis=1)) * m.score(X_test_global_scaled, y_test_global)
                for p, m in zip(client_probs, client_models)
            ])
            weights /= weights.sum()
        elif method == "Dynamic Federated Averaging with Intelligence (DFAI)":
            weights = np.array([
                (np.mean(np.max(p, axis=1)) * m.score(X_test_global_scaled, y_test_global)) /
                (np.mean(np.var(p, axis=1)) + 1e-6)
                for p, m in zip(client_probs, client_models)
            ])
            weights /= weights.sum()
        else:
            weights = np.ones(len(client_probs)) / len(client_probs)

        probs = sum(w * p for w, p in zip(weights, client_probs))
        return np.argmax(probs, axis=1)

    # ---- Federated Training ----
    final_results = []

    for model_name in selected_models:
        for averaging in selected_averagings:
            st.markdown(f"### Model: **{model_name}** | Averaging: **{averaging}**")
            for round_num in range(rounds):
                client_probs, client_models, client_data_sizes = [], [], []

                for i in range(clients):
                    model = get_model_instance(model_name, seed=42 + round_num + i)
                    X_c, y_c = X_parts[i], y_parts[i]
                    if len(np.unique(y_c)) < 2:
                        continue
                    model.fit(X_c, y_c)
                    client_models.append(model)
                    client_data_sizes.append(len(X_c))
                    if hasattr(model, 'predict_proba'):
                        client_probs.append(model.predict_proba(X_test_global_scaled))

                global_predictions = aggregate_predictions(client_probs, averaging, client_data_sizes, client_models)

                if global_predictions is not None:
                    acc = accuracy_score(y_test_global, global_predictions)
                    prec = precision_score(y_test_global, global_predictions, average='weighted', zero_division=0)
                    rec = recall_score(y_test_global, global_predictions, average='weighted', zero_division=0)
                    f1 = f1_score(y_test_global, global_predictions, average='weighted', zero_division=0)
                    st.write(f"**Round {round_num+1}**: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1 Score={f1:.4f}")

                    final_results.append({
                        "Model": model_name,
                        "Averaging": averaging,
                        "Round": round_num + 1,
                        "Accuracy": acc,
                        "Precision": prec,
                        "Recall": rec,
                        "F1 Score": f1
                    })

    # ---- Summary ----
    results_df = pd.DataFrame(final_results)
    if not results_df.empty:
        st.subheader("📊 Aggregated Results Summary")
        avg_df = results_df.groupby(["Model", "Averaging"])[["Accuracy", "Precision", "Recall", "F1 Score"]].mean().reset_index()
        numeric_cols = avg_df.select_dtypes(include=np.number).columns
        st.dataframe(avg_df.style.format({col: "{:.4f}" for col in numeric_cols}))

        fig = px.bar(
            avg_df,
            x="Model",
            y=["Accuracy", "Precision", "Recall", "F1 Score"],
            color="Averaging",
            barmode="group",
            title="Average Model Performance across Averaging Strategies"
        )
        st.plotly_chart(fig, use_container_width=True)
    st.session_state["final_results_df"] = results_df
    st.session_state["average_results_df"] = avg_df
#step 5: paper analysis
elif step == steps[5]: 
    st.header("📈 Results and Paper-Ready Analysis")

    # Ensure final results are available
    if "final_results_df" not in st.session_state or st.session_state.final_results_df is None:
        st.warning("⚠️ Please run the Federated Learning simulation first.")
        st.stop()

    final_df = st.session_state.final_results_df
    avg_df = st.session_state.average_results_df

    st.subheader("📌 Accuracy, Precision, Recall, F1 per Round")
    st.dataframe(final_df)

    # Plot: Accuracy per Round per Averaging Policy
    fig = px.line(
        final_df,
        x="Round",
        y="Accuracy",
        color="Averaging",
        line_dash="Model",
        title="Accuracy per Round per Averaging Policy",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

    # Plot: F1 Score Trend
    fig_f1 = px.line(
        final_df,
        x="Round",
        y="F1 Score",
        color="Averaging",
        line_dash="Model",
        title="F1 Score Trend across Rounds",
        markers=True
    )
    st.plotly_chart(fig_f1, use_container_width=True)

    # --- Time-based metrics if available ---
    if "Training Time" in final_df.columns:
        st.subheader("⏱️ Training Time per Round")
        fig_train_time = px.line(
            final_df,
            x="Round",
            y="Training Time",
            color="Model",
            title="Training Time per Round",
            markers=True
        )
        st.plotly_chart(fig_train_time, use_container_width=True)

    if "Aggregation Time" in final_df.columns:
        st.subheader("⏱️ Aggregation Time per Round")
        fig_agg_time = px.line(
            final_df,
            x="Round",
            y="Aggregation Time",
            color="Model",
            title="Aggregation Time per Round",
            markers=True
        )
        st.plotly_chart(fig_agg_time, use_container_width=True)

    if "Total Time" in final_df.columns:
        st.subheader("⏱️ Total Time per Round")
        fig_total_time = px.line(
            final_df,
            x="Round",
            y="Total Time",
            color="Model",
            title="Total Computation Time per Round",
            markers=True
        )
        st.plotly_chart(fig_total_time, use_container_width=True)

    # Aggregated Accuracy Comparison
    st.subheader("📊 Average Metrics by Model & Averaging")
    numeric_cols = avg_df.select_dtypes(include=[np.number]).columns
    st.dataframe(avg_df.style.format({col: "{:.4f}" for col in numeric_cols}))

    fig_avg = px.bar(
        avg_df,
        x="Model",
        y=["Accuracy", "Precision", "Recall", "F1 Score"],
        color="Averaging",
        barmode="group",
        title="Model Performance Comparison across Averaging Strategies"
    )
    st.plotly_chart(fig_avg, use_container_width=True)

    # Accuracy Boxplot
    fig_box = px.box(
        final_df,
        x="Averaging",
        y="Accuracy",
        color="Model",
        title="Distribution of Accuracy across Averaging Policies"
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # Time-series trend for performance if 'Round' is temporal
    st.subheader("📉 Time-Series Trends")
    fig_ts = px.line(
        final_df,
        x="Round",
        y="Accuracy",
        color="Model",
        title="Accuracy Trend per Round"
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # New metric: Accuracy - F1 difference (indicates class imbalance effects)
    final_df["Acc-F1 Gap"] = final_df["Accuracy"] - final_df["F1 Score"]
    fig_gap = px.line(
        final_df,
        x="Round",
        y="Acc-F1 Gap",
        color="Model",
        title="Accuracy vs F1 Score Gap (Lower is Better)"
    )
    st.plotly_chart(fig_gap, use_container_width=True)

    # Optional: Export buttons
    st.subheader("📥 Export Data")
    csv_all = final_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Round-wise Results", csv_all, "roundwise_results.csv", "text/csv")

    csv_avg = avg_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Averaged Results", csv_avg, "averaged_results.csv", "text/csv")



#nmbmb