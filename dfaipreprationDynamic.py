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
    "4. Select Model & FL Parameters", "5. Run Federated Learning",
    "6. Compare Results"
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

# Step 1: Upload Data
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


# Step 2: Select Variables
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


# Step 3: Feature Preprocessing
elif step == steps[2]:
    st.header("⚙️ Feature Preprocessing")
    data_to_process = st.session_state.get('data_after_exclusion', st.session_state.data)

    if data_to_process is None or st.session_state.target is None:
        st.warning("Please upload data and apply variable selections first.")
        st.stop()
    
    df = data_to_process.copy()
    
    all_selected_features = list(set((st.session_state.categorical or []) + (st.session_state.continuous or [])))
    
    # Ensure features and target are in the current dataframe columns
    if st.session_state.target not in df.columns:
        st.error(f"Target variable '{st.session_state.target}' not found in the dataset after exclusions. Please re-select variables.")
        st.stop()
    
    missing_features = [f for f in all_selected_features if f not in df.columns]
    if missing_features:
        st.error(f"Selected features not found after exclusions: {missing_features}. Please re-select variables.")
        st.stop()

    if not all_selected_features:
        st.warning("Please select at least one categorical or continuous feature in '2. Select Variables'.")
        st.stop()

    X = df[all_selected_features].copy() # Use .copy() to avoid SettingWithCopyWarning
    y_series = df[st.session_state.target].copy()

    st.write("Applying preprocessing:")
    # Handle missing values
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            if X[col].dtype == 'object' or col in st.session_state.categorical:
                mode_val = X[col].mode()
                if not mode_val.empty:
                    X.loc[:, col] = X[col].fillna(mode_val[0])
                    st.info(f"Missing values in **{col}** (categorical) imputed with mode ('{mode_val[0]}').")
                else: # Handle cases where mode is empty (e.g., all NaN)
                    X.loc[:, col] = X[col].fillna("Unknown") 
                    st.info(f"Missing values in **{col}** (categorical) imputed with 'Unknown' as mode was empty.")
            else: # Continuous
                mean_val = X[col].mean()
                X.loc[:, col] = X[col].fillna(mean_val)
                st.info(f"Missing values in **{col}** (continuous) imputed with mean ({mean_val:.2f}).")

    # Label Encode categorical features
    if st.session_state.categorical:
        for col in st.session_state.categorical:
            if col in X.columns:
                le = LabelEncoder()
                X.loc[:, col] = le.fit_transform(X[col].astype(str))
                st.info(f"Categorical feature **{col}** label encoded.")

    # Label Encode target variable
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y_series.astype(str))
    st.session_state.target_classes = le_target.classes_.tolist()
    st.info(f"Target variable **{st.session_state.target}** encoded. Classes: {st.session_state.target_classes}")
    
    if len(st.session_state.target_classes) < 2:
        st.error(f"Target variable '{st.session_state.target}' has only one unique class after encoding. Classification requires at least two classes.")
        st.stop()

    st.session_state.processed_X = X
    st.session_state.processed_y = y_encoded # Store the NumPy array

    st.write("Processed Features (first 5 rows):")
    st.dataframe(X.head())
    st.write("Processed Target (first 5 values):")
    st.dataframe(pd.Series(y_encoded).head())
    st.success("Features preprocessed. Proceed to '4. Select Model & FL Parameters'.")
    # Clear global test set if preprocessing changes
    st.session_state.X_test_global = None
    st.session_state.y_test_global = None
    st.session_state.X_train_global_scaled = None


# Step 4: Select Model & FL Parameters
elif step == steps[3]:
    st.header("🤖 Select Model & Configure Federated Learning")
    if st.session_state.processed_X is None or st.session_state.processed_y is None:
        st.warning("Please complete data preprocessing first in '3. Feature Preprocessing'.")
        st.stop()

    model_list = [
        "Random Forest", "Logistic Regression", "SVM", "Naive Bayes",
        "Decision Tree", "KNN", "Gradient Boosting", "AdaBoost"
    ]
    if TENSORFLOW_AVAILABLE:
        model_list.append("Deep Neural Network")
    else:
        st.info("Deep Neural Network model is unavailable because TensorFlow is not installed or not version 2.x.")

    st.session_state.selected_model_name = st.selectbox(
        "Choose a model for Federated Learning", model_list, 
        key="model_selection_box",
        index=model_list.index(st.session_state.selected_model_name) if st.session_state.selected_model_name in model_list else 0
    )

    st.subheader("Configure Federated Learning Parameters")
    # Initialize fl_params if it's None or if model changes, to reset learning_rate presence
    if st.session_state.fl_params is None or \
       st.session_state.fl_params.get("_model_for_params") != st.session_state.selected_model_name:
        st.session_state.fl_params = {} # Reset

    st.session_state.fl_params["_model_for_params"] = st.session_state.selected_model_name # Track current model

    st.session_state.fl_params["clients"] = st.slider(
        "Number of Clients", 2, 10, st.session_state.fl_params.get("clients", 3), 
        key="num_clients_slider", help="How many simulated clients will participate."
    )
    st.session_state.fl_params["rounds"] = st.slider(
        "Federated Rounds (Global Aggregations)", 1, 20, st.session_state.fl_params.get("rounds", 5), 
        key="num_rounds_slider", help="Number of server aggregations."
    )
    st.session_state.fl_params["epochs"] = st.slider(
        "Epochs per Client (Local Training)", 1, 50, st.session_state.fl_params.get("epochs", 5), 
        key="epochs_per_client_slider", help="Local training epochs per client per round."
    )
    st.session_state.fl_params["averaging"] = st.multiselect(
    "Select Averaging Policies", [
        "Simple Average", "Weighted Average (by data size)",
        "Dynamic Averaging (mean Q-value based)",
        "Dynamic Averaging (variance-based)",
        "Hybrid Averaging (mean * inverse variance)",
        "Confidence-Weighted Adaptive Averaging (CWAA)",
        "Dynamic Federated Averaging with Intelligence (DFAI)"
    ], 
    key="averaging_policy_multiselect", 
    default=["Simple Average"],
    help="Choose multiple aggregation methods to execute sequentially."
)

    if st.session_state.selected_model_name == "Deep Neural Network" and TENSORFLOW_AVAILABLE:
        st.session_state.fl_params["learning_rate"] = st.slider(
            "Learning Rate for DNN", 0.0001, 0.05, st.session_state.fl_params.get("learning_rate", 0.001), 
            format="%.4f", key="dnn_lr_slider", help="Learning rate for the DNN optimizer."
        )
    elif "learning_rate" in st.session_state.fl_params: # Remove LR if model is not DNN
        del st.session_state.fl_params["learning_rate"]


    st.success(f"Selected model: **{st.session_state.selected_model_name}**. FL parameters configured. Proceed to '5. Run Federated Learning'.")


# Step 5: Run Federated Learning
elif step == steps[4]:
    st.header("🚀 Run Federated Learning Simulation")
    if st.session_state.processed_X is None or st.session_state.processed_y is None or \
       st.session_state.fl_params is None or st.session_state.selected_model_name is None:
        st.warning("Please complete all previous steps: upload, select variables, preprocess, and set FL parameters.")
        st.stop()
    if st.button("Execute Selected Averaging Policies"):
        all_run_results = []
        progress_bar = st.progress(0)

        for idx, averaging_policy in enumerate(st.session_state.fl_params["averaging"], 1):
            st.info(f"Executing FL run for policy: {averaging_policy}")

            # Your existing federated learning simulation logic here,
            # using 'averaging_policy' as the current policy

            # Collect results for comparison
            accuracy = accuracy_score(st.session_state.y_test_global, y_pred_global)
            precision = precision_score(st.session_state.y_test_global, y_pred_global, average='weighted')
            recall = recall_score(st.session_state.y_test_global, y_pred_global, average='weighted')
            f1 = f1_score(st.session_state.y_test_global, y_pred_global, average='weighted')

            all_run_results.append({
                "Policy": averaging_policy,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            })

            progress_bar.progress(idx / len(st.session_state.fl_params["averaging"]))

        # Display summary results
        results_df = pd.DataFrame(all_run_results)
        st.subheader("Comparative Results")
        st.dataframe(results_df.style.format("{:.4f}"), use_container_width=True)

        # Performance visualization
        fig = px.bar(
            results_df,
            x="Policy",
            y=["Accuracy", "Precision", "Recall", "F1 Score"],
            title="Performance Metrics Comparison",
            barmode="group"
        )
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Start Federated Learning Run"):
        st.info(f"Starting FL simulation for **{st.session_state.selected_model_name}** with **{st.session_state.fl_params['averaging']}** policy...")
        
        X_processed = st.session_state.processed_X
        y_processed = st.session_state.processed_y # This is already a NumPy array
        
        # Global test set (create once or if data/preprocessing changed)
        if st.session_state.X_test_global is None or st.session_state.y_test_global is None or st.session_state.X_train_global_scaled is None:
            st.info("Creating global training/test split and scaling features...")
            X_train_global, X_test_global, y_train_global, y_test_global = train_test_split(
                X_processed, y_processed, test_size=0.2, random_state=42, 
                stratify=y_processed if len(np.unique(y_processed)) > 1 else None
            )
            
            scaler = StandardScaler()
            X_train_global_scaled = scaler.fit_transform(X_train_global)
            X_test_global_scaled = scaler.transform(X_test_global)

            st.session_state.X_train_global_scaled = X_train_global_scaled
            st.session_state.X_test_global = X_test_global_scaled # Store scaled test set
            st.session_state.y_train_global = y_train_global # Store for client splits
            st.session_state.y_test_global = y_test_global
            st.success("Global train/test split created and features scaled.")
        else:
            X_train_global_scaled = st.session_state.X_train_global_scaled
            # y_train_global is implicitly st.session_state.y_train_global
            st.info("Using existing global train/test split.")


        clients = st.session_state.fl_params["clients"]
        rounds = st.session_state.fl_params["rounds"]
        epochs = st.session_state.fl_params["epochs"]
        model_name = st.session_state.selected_model_name
        averaging_policy = st.session_state.fl_params["averaging"]
        learning_rate = st.session_state.fl_params.get("learning_rate", 0.001) # Default for DNN if not set

        # Split training data among clients
        # Ensure y_train_global is used for splitting y_parts
        y_train_for_split = st.session_state.y_train_global 
        
        min_samples_per_client = 5 # Minimum samples required for a client to train
        if len(X_train_global_scaled) < clients * min_samples_per_client:
            st.error(f"Not enough training samples ({len(X_train_global_scaled)}) to distribute among {clients} clients with at least {min_samples_per_client} samples each. Reduce clients or provide more data.")
            st.stop()

        X_parts = np.array_split(X_train_global_scaled, clients)
        y_parts = np.array_split(y_train_for_split, clients)


        global_model_weights_dnn = None # For DNNs
        all_round_metrics = [] 

        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_placeholder = st.empty() # For displaying metrics table dynamically

        for r in range(rounds):
            status_text.text(f"Federated Round {r+1}/{rounds} for Model: {model_name}, Policy: {averaging_policy}")
            client_updates = [] 
            client_data_sizes = []
            client_prediction_means = [] 
            client_prediction_variances = [] 
            client_local_accuracies = [] 

            active_clients_this_round = 0

            for i in range(clients):
                X_client, y_client = X_parts[i], y_parts[i]
                
                if len(X_client) < min_samples_per_client or len(np.unique(y_client)) < 2 : 
                    st.caption(f"Client {i+1}: Skipping (data: {len(X_client)}, classes: {len(np.unique(y_client))}). Needs >={min_samples_per_client} samples & >=2 classes.")
                    continue
                
                active_clients_this_round += 1
                # Split client data for local validation (important for dynamic policies)
                # Ensure stratify is possible
                stratify_y_client = y_client if len(np.unique(y_client)) > 1 else None
                try:
                    X_train_c, X_val_c, y_train_c, y_val_c = train_test_split(
                        X_client, y_client, test_size=0.2, random_state=42+r, stratify=stratify_y_client
                    )
                except ValueError as e: # Handles cases where split is not possible (e.g. too few samples for a class)
                    st.caption(f"Client {i+1}: Could not stratify split due to data imbalance ({e}). Using non-stratified split.")
                    X_train_c, X_val_c, y_train_c, y_val_c = train_test_split(
                        X_client, y_client, test_size=0.2, random_state=42+r
                    )


                if len(X_train_c) == 0 or len(X_val_c) == 0 or len(np.unique(y_train_c)) < 2:
                    st.caption(f"Client {i+1}: Skipping after local split (train: {len(X_train_c)} with {len(np.unique(y_train_c))} classes, val: {len(X_val_c)}).")
                    continue

                client_data_sizes.append(len(X_train_c))
                num_target_classes = len(st.session_state.target_classes)

                if model_name == "Deep Neural Network":
                    client_model_dnn = create_keras_model(X_train_c.shape[1], num_target_classes, learning_rate)
                    if r > 0 and global_model_weights_dnn is not None:
                        try:
                            set_model_weights(client_model_dnn, global_model_weights_dnn)
                        except ValueError as e:
                            st.warning(f"Client {i+1} (DNN): Error setting global weights in round {r+1}: {e}. Training from scratch.")
                            client_model_dnn = create_keras_model(X_train_c.shape[1], num_target_classes, learning_rate) # Re-init

                    y_train_c_cat = y_train_c
                    y_val_c_cat = y_val_c
                    loss_fn = 'sparse_categorical_crossentropy'
                    if num_target_classes > 2: # Multi-class, Keras expects one-hot for categorical_crossentropy
                        # If we stick to sparse_categorical_crossentropy, no need to one-hot y
                        pass # y_train_c and y_val_c are already integer encoded
                    
                    # Re-compile if needed (e.g. if loss function was changed based on y_cat)
                    client_model_dnn.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_fn, metrics=['accuracy'])
                    client_model_dnn.fit(X_train_c, y_train_c_cat, epochs=epochs, batch_size=max(1, len(X_train_c)//10), verbose=0, validation_data=(X_val_c, y_val_c_cat))
                    client_updates.append(get_model_weights(client_model_dnn))
                    
                    _, local_accuracy = client_model_dnn.evaluate(X_val_c, y_val_c_cat, verbose=0)
                    preds_val_proba = client_model_dnn.predict(X_val_c, verbose=0)
                    client_prediction_means.append(np.mean(np.max(preds_val_proba, axis=1)))
                    client_prediction_variances.append(np.mean(np.var(preds_val_proba, axis=1)))
                    client_local_accuracies.append(local_accuracy)

                else: # Traditional ML Models
                    model_instance = None
                    if model_name == "Random Forest": model_instance = RandomForestClassifier(random_state=42+r, n_estimators=max(10, epochs*2)) # epochs can guide n_estimators
                    elif model_name == "Logistic Regression": model_instance = LogisticRegression(max_iter=max(100, epochs*20), random_state=42+r, solver='liblinear')
                    elif model_name == "SVM": model_instance = SVC(probability=True, random_state=42+r, max_iter=max(100, epochs*20))
                    elif model_name == "Naive Bayes": model_instance = GaussianNB()
                    elif model_name == "Decision Tree": model_instance = DecisionTreeClassifier(random_state=42+r)
                    elif model_name == "KNN": model_instance = KNeighborsClassifier(n_neighbors=min(5, len(X_train_c)-1 if len(X_train_c)>1 else 1))
                    elif model_name == "Gradient Boosting": model_instance = GradientBoostingClassifier(random_state=42+r, n_estimators=max(10, epochs*2))
                    elif model_name == "AdaBoost": model_instance = AdaBoostClassifier(random_state=42+r, n_estimators=max(10, epochs*2))
                    
                    if model_instance is None: continue # Should not happen

                    model_instance.fit(X_train_c, y_train_c)
                    client_updates.append(model_instance)
                    
                    local_accuracy = model_instance.score(X_val_c, y_val_c)
                    client_local_accuracies.append(local_accuracy)
                    if hasattr(model_instance, 'predict_proba'):
                        preds_val_proba = model_instance.predict_proba(X_val_c)
                        client_prediction_means.append(np.mean(np.max(preds_val_proba, axis=1)))
                        client_prediction_variances.append(np.mean(np.var(preds_val_proba, axis=1)))
                    else: # Fallback for models without predict_proba
                        client_prediction_means.append(0.5) 
                        client_prediction_variances.append(1.0)


            if not client_updates or not active_clients_this_round:
                st.warning(f"Round {r+1}: No client models were trained successfully. Skipping aggregation for this round.")
                all_round_metrics.append({
                    "Round": r + 1, "Model": model_name, "Averaging Policy": averaging_policy,
                    "Accuracy": np.nan, "Precision": np.nan, "Recall": np.nan, "F1 Score": np.nan,
                    "Learning Rate": learning_rate if model_name == "Deep Neural Network" else "N/A"
                })
                progress_bar.progress((r + 1) / rounds)
                if all_round_metrics: metrics_placeholder.dataframe(pd.DataFrame(all_round_metrics))
                continue

            # --- Global Aggregation ---
            y_pred_global = None
            
            # Calculate weights_factor for aggregation based on policy
            weights_factor_agg = np.ones(len(client_updates)) / len(client_updates) # Default to simple average

            if len(client_updates) > 0: # Ensure there are updates to process
                if averaging_policy == "Weighted Average (by data size)" and client_data_sizes:
                    total_data_size = sum(client_data_sizes)
                    if total_data_size > 0: weights_factor_agg = np.array(client_data_sizes) / total_data_size
                elif averaging_policy == "Dynamic Averaging (mean Q-value based)" and client_prediction_means:
                    total_mean_q = sum(client_prediction_means)
                    if total_mean_q > 0: weights_factor_agg = np.array(client_prediction_means) / total_mean_q
                elif averaging_policy == "Dynamic Averaging (variance-based)" and client_prediction_variances:
                    inverse_variances = [1.0 / (v + 1e-9) for v in client_prediction_variances]
                    total_inverse_variance = sum(inverse_variances)
                    if total_inverse_variance > 0: weights_factor_agg = np.array(inverse_variances) / total_inverse_variance
                elif averaging_policy == "Hybrid Averaging (mean * inverse variance)" and client_prediction_means and client_prediction_variances:
                    hybrid_scores = [m / (v + 1e-9) for m, v in zip(client_prediction_means, client_prediction_variances)]
                    total_hybrid_score = sum(hybrid_scores)
                    if total_hybrid_score > 0: weights_factor_agg = np.array(hybrid_scores) / total_hybrid_score
                elif averaging_policy == "Confidence-Weighted Adaptive Averaging (CWAA)" and client_local_accuracies and client_prediction_means:
                    scores = np.array(client_local_accuracies) * np.array(client_prediction_means)
                    total_score = sum(scores)
                    if total_score > 0: weights_factor_agg = scores / total_score
                elif averaging_policy == "Dynamic Federated Averaging with Intelligence (DFAI)":
                    trust_scores = np.array(client_local_accuracies) * np.array(client_prediction_means) / (np.array(client_prediction_variances) + 1e-9)
                    trust_scores_normalized = trust_scores / np.sum(trust_scores)
                    weights_factor_agg = trust_scores_normalized

                    # Adaptive selection among FedAvg, FedNova, FedProx
                    if np.var(client_data_sizes) < 1e-2:  # Balanced clients
                        weights_factor_agg = np.ones(len(client_updates)) / len(client_updates)  # FedAvg
                    elif np.max(client_data_sizes) / np.min(client_data_sizes) > 2:  # Imbalanced data volumes
                        weights_factor_agg = np.array(client_data_sizes) / np.sum(client_data_sizes)  # FedNova
                    else:  # Noisy data scenario
                        proximal_term = np.array([np.linalg.norm(w - global_model_weights_dnn if global_model_weights_dnn else 0)**2 for w in client_updates])
                        weights_factor_agg = (trust_scores / (proximal_term + 1e-9))
                        weights_factor_agg /= np.sum(weights_factor_agg)  # FedProx
                # Normalize weights_factor_agg to sum to 1
                if sum(weights_factor_agg) > 0:
                    weights_factor_agg = weights_factor_agg / sum(weights_factor_agg)
                else: # Fallback if all weights are zero
                    weights_factor_agg = np.ones(len(client_updates)) / len(client_updates)


            if model_name == "Deep Neural Network":
                aggregated_dnn_weights = [np.zeros_like(w) for w in client_updates[0]]
                for client_idx, client_weight_list in enumerate(client_updates):
                    weight = weights_factor_agg[client_idx]
                    for i, layer_weights in enumerate(client_weight_list):
                        aggregated_dnn_weights[i] += layer_weights * weight
                global_model_weights_dnn = aggregated_dnn_weights
                
                # Evaluate global DNN model
                global_dnn_model = create_keras_model(st.session_state.X_test_global.shape[1], len(st.session_state.target_classes), learning_rate)
                set_model_weights(global_dnn_model, global_model_weights_dnn)
                
                y_test_eval = st.session_state.y_test_global # Already integer encoded
                # No need to to_categorical y_test_global if using sparse_categorical_crossentropy
                
                loss, acc = global_dnn_model.evaluate(st.session_state.X_test_global, y_test_eval, verbose=0)
                y_pred_proba_global = global_dnn_model.predict(st.session_state.X_test_global, verbose=0)
                y_pred_global = np.argmax(y_pred_proba_global, axis=1)

            else: # Traditional ML Models Aggregation (prediction averaging)
                all_client_preds_proba_on_global_test = []
                can_predict_proba_all = all(hasattr(m, 'predict_proba') for m in client_updates)

                if not can_predict_proba_all:
                    st.caption(f"Round {r+1}: Some traditional models lack predict_proba. Using majority vote.")
                    all_client_preds_on_global_test = [m.predict(st.session_state.X_test_global) for m in client_updates]
                    final_preds_transposed = np.array(all_client_preds_on_global_test).T
                    y_pred_global = np.array([np.bincount(row_preds).argmax() for row_preds in final_preds_transposed])
                else:
                    for model_idx, client_model_instance in enumerate(client_updates):
                        all_client_preds_proba_on_global_test.append(client_model_instance.predict_proba(st.session_state.X_test_global))
                    
                    # Weighted average of probabilities
                    averaged_proba_global = np.average(np.array(all_client_preds_proba_on_global_test), axis=0, weights=weights_factor_agg)
                    y_pred_global = np.argmax(averaged_proba_global, axis=1)

            # Calculate metrics for the current round
            if y_pred_global is not None:
                accuracy = accuracy_score(st.session_state.y_test_global, y_pred_global)
                precision = precision_score(st.session_state.y_test_global, y_pred_global, average='weighted', zero_division=0)
                recall = recall_score(st.session_state.y_test_global, y_pred_global, average='weighted', zero_division=0)
                f1 = f1_score(st.session_state.y_test_global, y_pred_global, average='weighted', zero_division=0)
            else: # Should not happen if aggregation occurred
                accuracy, precision, recall, f1 = np.nan, np.nan, np.nan, np.nan


            all_round_metrics.append({
                "Round": r + 1, "Model": model_name, "Averaging Policy": averaging_policy,
                "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1,
                "Learning Rate": learning_rate if model_name == "Deep Neural Network" else "N/A"
            })
            
            progress_bar.progress((r + 1) / rounds)
            if all_round_metrics: metrics_placeholder.dataframe(pd.DataFrame(all_round_metrics))
        
        # Store results of this run for comparison
        run_id = f"{model_name.replace(' ', '')}_{averaging_policy.split('(')[0].strip().replace(' ', '')}_{len(st.session_state.comparison_results) + 1}"
        final_metrics_df = pd.DataFrame(all_round_metrics)
        st.session_state.comparison_results.append({
            "Run ID": run_id,
            "Model": model_name,
            "Averaging Policy": averaging_policy,
            "Parameters": st.session_state.fl_params.copy(), # Store a copy
            "Metrics History": final_metrics_df,
            "Final Predictions": y_pred_global # Store final predictions for this run
        })
        st.success(f"Federated learning simulation complete for Run ID: **{run_id}**!")
        st.write("Final Metrics Table for this Run:")
        st.dataframe(final_metrics_df)
        st.balloons()
        st.info("Proceed to '6. Compare Results' to see how this run stacks against others.")


# Step 6: Compare Results
elif step == steps[5]:
    st.header("📊 Compare Federated Learning Runs")

    if not st.session_state.comparison_results:
        st.warning("No federated learning runs have been completed yet. Please go to '5. Run Federated Learning' to start a simulation.")
        st.stop()

    st.subheader("Available Runs for Comparison")
    
    run_options = [run["Run ID"] for run in st.session_state.comparison_results]
    
    # Default to all runs if some were cleared or if it's the first time
    default_selected_runs = [run_id for run_id in st.session_state.get("selected_runs_for_comparison", run_options) if run_id in run_options]
    if not default_selected_runs and run_options: # If previous selection is invalid, default to all current
        default_selected_runs = run_options

    selected_runs_ids = st.multiselect(
        "Select Runs to Compare", 
        options=run_options, 
        default=default_selected_runs, 
        key="compare_runs_multiselect"
    )
    st.session_state.selected_runs_for_comparison = selected_runs_ids # Remember selection

    if not selected_runs_ids:
        st.info("Please select at least one run to compare.")
        st.stop()

    # Button to clear selected or all comparison results
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Selected Runs from Comparison History"):
            st.session_state.comparison_results = [
                run for run in st.session_state.comparison_results if run["Run ID"] not in selected_runs_ids
            ]
            st.session_state.selected_runs_for_comparison = [ # Update remembered selection
                run_id for run_id in selected_runs_ids if run_id in [r["Run ID"] for r in st.session_state.comparison_results]
            ]
            st.success("Selected runs cleared from history. Refreshing comparison...")
            st.rerun() # Rerun to update the multiselect and plots
    with col2:
        if st.button("Clear ALL Runs from Comparison History", type="primary"):
            if st.checkbox("Confirm clearing ALL history?", key="confirm_clear_all_hist"):
                st.session_state.comparison_results = []
                st.session_state.selected_runs_for_comparison = []
                st.success("All run history cleared. Refreshing comparison...")
                st.rerun()


    combined_df_for_plot = pd.DataFrame()
    final_metrics_summary = []

    for run_info in st.session_state.comparison_results:
        if run_info["Run ID"] in selected_runs_ids:
            df_temp = run_info["Metrics History"].copy()
            df_temp["Run ID"] = run_info["Run ID"]
            df_temp["Model + Policy"] = f"{run_info['Model']} ({run_info['Averaging Policy']})"
            combined_df_for_plot = pd.concat([combined_df_for_plot, df_temp], ignore_index=True)
            
            # Summary of final metrics for table
            final_round_metric = df_temp.iloc[-1]
            summary_entry = {
                "Run ID": run_info["Run ID"],
                "Model": run_info["Model"],
                "Policy": run_info["Averaging Policy"],
                "Clients": run_info["Parameters"]["clients"],
                "Rounds": run_info["Parameters"]["rounds"],
                "Epochs": run_info["Parameters"]["epochs"],
                "LR (DNN)": run_info["Parameters"].get("learning_rate", "N/A"),
                "Final Accuracy": final_round_metric["Accuracy"],
                "Final F1 Score": final_round_metric["F1 Score"]
            }
            final_metrics_summary.append(summary_entry)


    if combined_df_for_plot.empty:
        st.info("No data to display for the selected runs. Please run a simulation or select valid runs.")
        st.stop()

    st.subheader("Summary of Final Metrics for Selected Runs")
    if final_metrics_summary:
        summary_df = pd.DataFrame(final_metrics_summary)
        st.dataframe(summary_df.style.format({
            "Final Accuracy": "{:.4f}", 
            "Final F1 Score": "{:.4f}",
            "LR (DNN)": lambda x: f"{x:.4f}" if isinstance(x, float) else x
        }))

    st.subheader("Performance Comparison Plots")
    metric_to_plot = st.selectbox("Select Metric to Plot", ["Accuracy", "Precision", "Recall", "F1 Score"], key="metric_for_plot")

    fig = px.line(combined_df_for_plot, x="Round", y=metric_to_plot, color="Model + Policy",
                  title=f"{metric_to_plot} Over Federated Rounds for Selected Runs",
                  markers=True,
                  hover_name="Run ID",
                  labels={"Round": "Federated Round", metric_to_plot: metric_to_plot})
    fig.update_layout(hovermode="x unified", legend_title_text='Model (Policy)')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed Classification Report for Final Round of Each Selected Run")
    if st.session_state.y_test_global is not None and st.session_state.target_classes:
        y_true_report = st.session_state.y_test_global
        target_names_report = [str(cls) for cls in st.session_state.target_classes]

        for run_info in st.session_state.comparison_results:
            if run_info["Run ID"] in selected_runs_ids:
                st.markdown(f"--- \n#### Run ID: {run_info['Run ID']}")
                st.markdown(f"*Model: {run_info['Model']}, Policy: {run_info['Averaging Policy']}*")
                
                y_pred_final_report = run_info.get("Final Predictions")

                if y_pred_final_report is not None:
                    try:
                        report_dict = classification_report(
                            y_true_report, y_pred_final_report, 
                            target_names=target_names_report, 
                            output_dict=True, zero_division=0
                        )
                        report_df = pd.DataFrame(report_dict).transpose()
                        st.dataframe(report_df.style.format("{:.3f}"))
                    except Exception as e:
                        st.error(f"Error generating classification report for {run_info['Run ID']}: {e}")
                else:
                    st.warning(f"Final predictions not available for Run ID: {run_info['Run ID']} to generate a detailed classification report. This might be an older run before predictions were stored.")
    else:
        st.warning("Global test set (y_test_global) or target classes not found in session state. Cannot generate classification reports.")

    # Download combined data
    if not combined_df_for_plot.empty:
        csv_export = combined_df_for_plot.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Comparison Data as CSV",
            data=csv_export,
            file_name="fl_comparison_results.csv",
            mime="text/csv",
        )
