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
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, roc_curve, auc
)
from sklearn.model_selection import StratifiedKFold

import numpy as np
import traceback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew, kurtosis
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
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
        #default=[col for col in feature_cols if df_current[col].dtype == 'object' and col in (st.session_state.categorical or [])] or \
        #        [col for col in feature_cols if df_current[col].dtype == 'object'],
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

    # Step 2.1: Show class distribution
    st.markdown("### 🧮 Class Distribution")
    st.write(y.value_counts())
    st.write(y.value_counts(normalize=True).rename("Proportion"))

    # Step 2.2: Label encode target if categorical
    if y.dtype == 'object':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y)
        st.info("🎯 Target variable was encoded using LabelEncoder.")

    # Step 2.3: SMOTE balancing if needed
    try:
        st.write("SMOTE STARTED")
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        X_bal, y_bal = sm.fit_resample(X, y)
        X= X_bal
        y= y_bal
        st.success("✅ Applied SMOTE for class balancing.")

        # Show class distribution after balancing
        st.markdown("### ⚖️ Class Distribution After SMOTE")
        st.write(pd.Series(y_bal).value_counts())
        st.write(pd.Series(y_bal).value_counts(normalize=True).rename("Proportion"))

        # Confusion matrix on train/test split of balanced data
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import confusion_matrix
        X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42)
        clf = RandomForestClassifier()
        clf.fit(X_train_b, y_train_b)
        y_pred_b = clf.predict(X_test_b)
        cm = confusion_matrix(y_test_b, y_pred_b)
        st.markdown("#### 📉 Confusion Matrix (After SMOTE)")
        st.write(cm)
    except Exception as e:
        st.warning(f"⚠️ SMOTE balancing failed: {e}")

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
        kurt_val = kurtosis(vals)
        if kurt_val > 50 and skewness > 2:
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

    # Step 4: Data leakage quick test via dummy classifier
    st.markdown("#### 🧪 Data Leakage Sanity Check")
    try:
        from sklearn.dummy import DummyClassifier
        X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(X, y, test_size=0.2, stratify=y)
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train_tmp, y_train_tmp)
        dummy_preds = dummy.predict(X_test_tmp)
        dummy_acc = accuracy_score(y_test_tmp, dummy_preds)
        st.info(f"🧪 Dummy Classifier Accuracy (Most Frequent Strategy): {dummy_acc:.4f}")
        st.markdown("_If your real model accuracy is close to this, your data might be imbalanced or leaking._")
    except Exception as e:
        st.warning(f"Could not perform dummy classifier check: {e}")

    # Process झालेला डेटा पुढील स्टेपसाठी साठवा
    st.session_state.processed_X = X
    st.session_state.processed_y = y
    #step 4: Run the code
elif step == steps[4]:
    if 'processed_X' not in st.session_state or 'processed_y' not in st.session_state:
        st.error("❌ Please upload and preprocess your dataset first. `processed_X` and `processed_y` not found in session state.")
        st.stop()

    MODEL_OPTIONS = [
        "Random Forest", "Logistic Regression", "SVM", "Naive Bayes",
        "Decision Tree", "KNN", "Gradient Boosting", "AdaBoost", "CNN"
    ]

    AVERAGING_POLICIES = [
        "Simple Average", "Weighted Average (by data size)",
        "Dynamic Averaging (mean Q-value based)",
        "Dynamic Averaging (variance-based)",
        "Hybrid Averaging (mean * inverse variance)",
        "Confidence-Weighted Adaptive Averaging (CWAA)",
        "Dynamic Federated Averaging with Intelligence (DFAI)"
    ]

    def get_model(name, input_shape=None, num_classes=None):
        models = {
            "Random Forest": RandomForestClassifier(),
            "Logistic Regression": LogisticRegression(solver='liblinear'),
            "SVM": SVC(probability=True),
            "Naive Bayes": GaussianNB(),
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "AdaBoost": AdaBoostClassifier()
        }
        if name == "CNN":
            model = Sequential([
                Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
                MaxPooling1D(pool_size=2),
                Dropout(0.2),
                Conv1D(64, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(64, activation='relu'),
                Dense(num_classes, activation='softmax')
            ])
            model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
            return model
        else:
            return models.get(name)

    st.sidebar.header("🧠 Federated Learning Simulation")
    num_clients = st.sidebar.slider("Number of Clients", 2, 100, 3)
    num_rounds = st.sidebar.slider("Number of Federated Rounds", 1, 100, 3)

    if 'client_configs' not in st.session_state:
        st.session_state.client_configs = [{} for _ in range(num_clients)]
    elif len(st.session_state.client_configs) < num_clients:
        st.session_state.client_configs.extend([{} for _ in range(num_clients - len(st.session_state.client_configs))])
    elif len(st.session_state.client_configs) > num_clients:
        st.session_state.client_configs = st.session_state.client_configs[:num_clients]

    st.sidebar.subheader("🔧 Configure Each Client")
    for i in range(num_clients):
        st.sidebar.markdown(f"**Client {i+1}**")
        try:
            model_choice = st.sidebar.selectbox(f"Model for Client {i+1}", MODEL_OPTIONS, key=f"model_{i}")
            avg_policy = st.sidebar.selectbox(f"Averaging for Client {i+1}", AVERAGING_POLICIES, key=f"avg_{i}")
            st.session_state.client_configs[i] = {"model": model_choice, "averaging": avg_policy}
        except Exception as e:
            st.warning(f"⚠️ Error setting config for Client {i+1}: {e}")

    if st.button("Run Federated Learning"):
        try:
            X = st.session_state.get('processed_X')
            y = st.session_state.get('processed_y')

            if X is None or y is None:
                st.error("❌ 'processed_X' or 'processed_y' is missing. Please upload and preprocess the dataset first.")
                st.stop()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            y_test_labels = np.argmax(y_test, axis=1) if y_test.ndim > 1 and y_test.shape[1] > 1 else y_test

            skf = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=42)
            X_parts, y_parts = [], []
            for train_idx, test_idx in skf.split(X_train_scaled, y_train.to_numpy()):
                X_parts.append(X_train_scaled[test_idx])
                y_parts.append(y_train.iloc[test_idx])

            round_wise_metrics, round_wise_weights = [], []
            round_wise_times, round_wise_losses = [], []
            round_wise_aucs, round_wise_disagreements = [], []
            communication_overhead = []
            roc_data_per_round = []

            for rnd in range(num_rounds):
                st.markdown(f"### 🔄 Round {rnd+1}")
                client_models, client_probs, client_preds = [], [], []
                client_metrics, client_times, data_sizes = [], [], []
                client_trust_scores = []
                total_comm = 0

                for i in range(num_clients):
                    try:
                        model_type = st.session_state.client_configs[i]["model"]

                        if model_type == "CNN":
                            X_cnn = X_parts[i].reshape((X_parts[i].shape[0], X_parts[i].shape[1], 1))
                            y_cnn = to_categorical(y_parts[i])
                            X_test_cnn = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
                            y_test_cnn = to_categorical(y_test_labels)
                            input_shape = (X_cnn.shape[1], 1)
                            num_classes = y_cnn.shape[1]
                            model = get_model("CNN", input_shape, num_classes)

                            start_time = time.time()
                            model.fit(X_cnn, y_cnn, epochs=10, batch_size=32, verbose=0)
                            elapsed = time.time() - start_time
                            probs = model.predict(X_test_cnn)
                            preds = np.argmax(probs, axis=1)
                        else:
                            model = get_model(model_type)
                            start_time = time.time()
                            model.fit(X_parts[i], y_parts[i])
                            elapsed = time.time() - start_time
                            if hasattr(model, 'predict_proba'):
                                probs = model.predict_proba(X_test_scaled)
                            else:
                                probs = np.zeros((X_test_scaled.shape[0], len(np.unique(y))))
                            preds = model.predict(X_test_scaled)

                        acc_val = accuracy_score(y_test_labels, preds)
                        trust_score = acc_val * (1 / (elapsed + 1e-6))
                        client_trust_scores.append(trust_score)
                        client_times.append(elapsed)
                        client_models.append(model)
                        client_probs.append(probs)
                        client_preds.append(preds)
                        data_sizes.append(len(X_parts[i]))

                        if probs.shape[1] == 2:
                            auc_val = roc_auc_score(y_test_labels, probs[:, 1])
                        else:
                            y_bin = label_binarize(y_test_labels, classes=np.arange(probs.shape[1]))
                            auc_val = roc_auc_score(y_bin, probs, multi_class='ovr')
                        loss_val = log_loss(y_test_labels, probs)
                        total_comm += probs.nbytes

                        client_metrics.append({
                            "Client": f"Client {i+1}",
                            "Model": model_type,
                            "Accuracy": acc_val,
                            "Precision": precision_score(y_test_labels, preds, average='weighted', zero_division=0),
                            "Recall": recall_score(y_test_labels, preds, average='weighted', zero_division=0),
                            "F1 Score": f1_score(y_test_labels, preds, average='weighted', zero_division=0),
                            "AUC": auc_val,
                            "Loss": loss_val,
                            "Train T(s)": elapsed,
                            "Trust Score": trust_score
                        })
                    except Exception as e:
                        st.error(f"❌ Error in client {i+1}: {str(e)}")
                        st.text(traceback.format_exc())

                def aggregate_predictions(policy):
                    weights = np.ones(len(client_probs)) / len(client_probs)
                    try:
                        if policy == "Simple Average":
                            weights = np.ones(len(client_probs)) / len(client_probs)
                        elif policy == "Weighted Average (by data size)":
                            weights = np.array(data_sizes) / np.sum(data_sizes)
                        elif policy == "Dynamic Averaging (mean Q-value based)":
                            weights = np.array([np.mean(np.max(p, axis=1)) for p in client_probs])
                            weights /= weights.sum()
                        elif policy == "Dynamic Averaging (variance-based)":
                            weights = np.array([1 / (np.mean(np.var(p, axis=1)) + 1e-6) for p in client_probs])
                            weights /= weights.sum()
                        elif policy == "Hybrid Averaging (mean * inverse variance)":
                            weights = np.array([np.mean(np.max(p, axis=1)) / (np.mean(np.var(p, axis=1)) + 1e-6) for p in client_probs])
                            weights /= weights.sum()
                        elif policy == "Confidence-Weighted Adaptive Averaging (CWAA)":
                            weights = np.array([
                                np.mean(np.max(p, axis=1)) * accuracy_score(y_test_labels, np.argmax(p, axis=1))
                                for p in client_probs
                            ])
                            weights /= weights.sum()
                        elif policy == "Dynamic Federated Averaging with Intelligence (DFAI)":
                            weights = np.array(client_trust_scores)
                            weights /= weights.sum()
                    except Exception as e:
                        st.error(f"❌ Error in aggregation logic: {str(e)}")
                        st.text(traceback.format_exc())

                    combined = sum(w * p for w, p in zip(weights, client_probs))
                    return np.argmax(combined, axis=1), weights, combined

                if client_probs:
                    try:
                        selected_policy = st.session_state.client_configs[0]["averaging"]
                        final_preds, weights, combined_probs = aggregate_predictions(selected_policy)

                        acc = accuracy_score(y_test_labels, final_preds)
                        prec = precision_score(y_test_labels, final_preds, average='weighted', zero_division=0)
                        rec = recall_score(y_test_labels, final_preds, average='weighted', zero_division=0)
                        f1 = f1_score(y_test_labels, final_preds, average='weighted', zero_division=0)
                        disagreement = np.mean([np.mean(final_preds != cp) for cp in client_preds])
                        loss = log_loss(y_test_labels, combined_probs)

                        if combined_probs.shape[1] == 2:
                            auc_val = roc_auc_score(y_test_labels, combined_probs[:, 1])
                        else:
                            y_bin = label_binarize(y_test_labels, classes=np.arange(combined_probs.shape[1]))
                            auc_val = roc_auc_score(y_bin, combined_probs, multi_class='ovr')

                        def theil_index(values):
                            values = np.array(values) + 1e-6
                            mean_val = np.mean(values)
                            return np.mean((values / mean_val) * np.log(values / mean_val))

                        f1_scores_clients = [m["F1 Score"] for m in client_metrics]
                        fairness_val = theil_index(f1_scores_clients)
                        disagreements = [np.mean(final_preds != cp) for cp in client_preds]
                        noise_robustness = 1 - np.mean(disagreements)

                        round_wise_metrics.append({"Round": rnd + 1, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1, "Fairness": fairness_val, "Noise Robustness": noise_robustness})
                        round_wise_weights.append(weights.tolist())
                        round_wise_times.append(client_times)
                        round_wise_losses.append(loss)
                        round_wise_aucs.append(auc_val)
                        round_wise_disagreements.append(disagreement)
                        communication_overhead.append(total_comm)

                        st.success(f"📈 Round {rnd+1} → Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc_val:.4f}, Loss: {loss:.4f}")
                        st.write("🔍 Averaging Weights:", {f"Client {i+1}": round(w, 4) for i, w in enumerate(weights)})
                        df_client = pd.DataFrame(client_metrics)
                        float_cols = df_client.select_dtypes(include='float').columns
                        st.dataframe(df_client.style.format({col: "{:.4f}" for col in float_cols}))

                        fpr = {}
                        tpr = {}
                        for i in range(combined_probs.shape[1]):
                            fpr[i], tpr[i], _ = roc_curve((y_test_labels == i).astype(int), combined_probs[:, i])
                        roc_data_per_round.append((rnd + 1, fpr, tpr))

                        plt.close('all')
                    except Exception as e:
                        st.error(f"❌ Error during aggregation or final metric computation: {str(e)}")
                        st.text(traceback.format_exc())
                        break

            if round_wise_metrics:
                st.subheader("📊 Metrics Over Rounds")
                
                df_summary = pd.DataFrame(round_wise_metrics)
                print(df_summary)
                df_summary["Loss"] = round_wise_losses
                df_summary["AUC"] = round_wise_aucs
                df_summary["Disagreement"] = round_wise_disagreements
                df_summary["Com.Overhead"] = communication_overhead
                st.dataframe(df_summary.style.format("{:.4f}"))

                fig = px.line(df_summary, x="Round", y=["Accuracy",	"Precision","Recall",	"F1 Score",	"Fairness",	"Noise Robustness",	"Loss",	"AUC",	"Disagreement",	"Com.Overhead"],
                            markers=True, title="Federated Metrics Over Rounds")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("📈 ROC Curve (Last Round)")
                last_round, fpr_dict, tpr_dict = roc_data_per_round[-1]
                fig_roc = go.Figure()
                for i in fpr_dict:
                    fig_roc.add_trace(go.Scatter(x=fpr_dict[i], y=tpr_dict[i], mode='lines', name=f'Class {i}'))
                fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                st.plotly_chart(fig_roc)

                st.subheader("🕒 Time Complexity per Client")
                for r, times in enumerate(round_wise_times):
                    st.markdown(f"**Round {r+1}:** " + ", ".join([f"Client {i+1}: {t:.4f}s" for i, t in enumerate(times)]))

                st.subheader("✅ Final Federated Metrics (Last Round)")
                st.write({k: round(v, 4) for k, v in round_wise_metrics[-1].items() if k != "Round"})

        except Exception as e:
            st.error(f"❌ Global Error during simulation: {str(e)}")
            st.text(traceback.format_exc())


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