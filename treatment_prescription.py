import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

import shap
import matplotlib.pyplot as plt

import os
import json

# ... (imports remain the same)

def load_and_clean_data(filepath):
    # ... (remains the same)
    """Loads and cleans the dataset using logic from the original notebook."""
    print("Loading dataset...")
    df = pd.read_excel(filepath)
    
    # Rename columns
    df.columns = df.columns.str.strip()
    df = df.rename({
        "Age at diagnosis": "diagnosis_age",
        "Regional nodes positive (1988+)": "lymph_nodes",
        "Total number of in situ/malignant tumors for patient": "malignant_tumors",
        "Radiation recode": "radiation_type",
        "Chemotherapy recode": "chemotherapy_done",
        "Radiation sequence with surgery": "radiation_sequence",
        "ER Status Recode Breast Cancer (1990+)": "estrogen_info",
        "PR Status Recode Breast Cancer (1990+)": "progesterone_info",
        "CS tumor size (2004-2015)": "tumor_size",
        "Derived HER2 Recode (2010+)": "her2_info",
        "Regional nodes examined (1988+)": "nodes_examined",
        "COD to site recode": "cause_of_death",
        "Race recode": "race",
        "Sex": "sex",
        "Vital status recode (study cutoff used)": "vital_status",
        "Diagnosis_year": "diagnosis_year",
        "Last_fu _year": "treatment_year",
        "interva_years": "num_screening",
        "stutus_5_years": "vital_status_5y"
    }, axis="columns")

    # Handle missing values and types
    df["race"] = df["race"].replace("Unknown", np.nan)
    
    df["tumor_size"] = (df["tumor_size"]
        .replace("Blank(s)", "999")
        .astype(int)
        .replace(999, np.nan))
        
    df.loc[df["lymph_nodes"] > 90, "lymph_nodes"] = np.nan
    
    # Feature Engineering
    print("Performing feature engineering...")
    
    # Chemotherapy
    df["chemotherapy_done"] = df["chemotherapy_done"].apply(lambda x: 1 if x == "Yes" else 0)
    
    # Radiotherapy
    def radiotherapy_done(value):
        if value in ["None/Unknown", "Refused (1988+)"]:
            return 0
        return 1
    
    df["radiotherapy_done"] = df["radiation_type"].apply(radiotherapy_done)
    
    # Therapeutic Plan (0, 1, 2)
    df["therapeutic_plan"] = df["chemotherapy_done"] + df["radiotherapy_done"]
    
    # Ratios and Interactions
    df["lymph_nodes_ratio"] = df["lymph_nodes"] / (df["nodes_examined"] + 1e-6)
    df["tumor_load"] = df["tumor_size"] * df["malignant_tumors"]
    
    # Age Group
    bins = [-np.inf, 45, 65, np.inf]
    labels = ["less_45", "between_45_65", "greater_65"]
    df['age_group'] = pd.cut(
        df['diagnosis_age'],
        bins=bins,
        labels=labels,
        right=False
    ).astype(object)

    # Drop rows with missing target or essential features if necessary, 
    # but XGBoost handles NaNs. We will fill NaNs for categorical encoding though.
    
    return df

def train_model(df):
    # ... (remains the same)
    """Trains the XGBoost model."""
    print("Preparing data for training...")
    
    # Select features
    # We exclude 'vital_status', 'vital_status_5y', 'cause_of_death', 'treatment_year', 'num_screening' (target or leakage)
    # We also exclude intermediate columns if we want to be strict, but let's keep the main ones.
    # The goal is to predict 'vital_status_5y'
    
    target = "vital_status_5y"
    
    # Encode target: Alive -> 1, Dead -> 0
    df[target] = df[target].map({"Alive": 1, "Dead": 0})
    
    # Identify categorical and numerical columns
    # Based on notebook info:
    categorical_cols = ["radiation_type", "radiation_sequence", "estrogen_info", "progesterone_info", 
                        "her2_info", "race", "sex", "age_group"]
    
    # We need to handle 'radiation_type' carefully because it implies radiotherapy.
    # In S-Learner, we include treatment as a feature.
    # Our treatments are 'chemotherapy_done' and 'radiotherapy_done'.
    # 'radiation_type' is highly correlated with 'radiotherapy_done'. 
    # To avoid leakage or confusion, we should probably rely on the binary flags for prescription simulation,
    # but the model needs to learn from the data.
    # If we change 'radiotherapy_done' to 1, 'radiation_type' should technically change too.
    # For simplicity in this S-Learner, we will use the binary flags as the treatment variables 
    # and drop 'radiation_type' and 'therapeutic_plan' (which is just sum) from the features list 
    # to avoid perfect collinearity/leakage during simulation, OR we treat them as covariates.
    # However, 'radiation_type' contains more info (Beam vs Implants). 
    # For the prescription task "Best Treatment", usually implies "Chemo Yes/No" and "Radio Yes/No".
    # So we will stick to the binary flags as the manipulatable treatments.
    # We will DROP 'radiation_type' and 'therapeutic_plan' from predictors to keep it clean.
    
    feature_cols = [
        "diagnosis_age", "lymph_nodes", "malignant_tumors", 
        "tumor_size", "nodes_examined", "diagnosis_year",
        "lymph_nodes_ratio", "tumor_load",
        "estrogen_info", "progesterone_info", "her2_info", "race", "sex", "age_group", "radiation_sequence",
        "chemotherapy_done", "radiotherapy_done" # Treatments
    ]
    
    X = df[feature_cols].copy()
    y = df[target].copy()
    
    # Encode Categoricals
    label_encoders = {}
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype(str) # Handle NaNs as 'nan' string for encoding
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
            
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost model...")
    model = XGBClassifier(
        use_label_encoder=False, 
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"Model Evaluation Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    
    return model, feature_cols, label_encoders, X_test, y_test

def prescribe_treatment(model, patient_data, feature_cols, output_dir="outputs/prescriptions"):
    """
    Prescribes the best treatment for a given patient (or set of patients).
    treatments: (Chemo, Radio) -> (0,0), (1,0), (0,1), (1,1)
    Saves detailed results to JSON.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    treatments = [
        (0, 0), # No Chemo, No Radio
        (1, 0), # Chemo, No Radio
        (0, 1), # No Chemo, Radio
        (1, 1)  # Chemo, Radio
    ]
    
    treatment_names = [
        "No Treatment",
        "Chemotherapy Only",
        "Radiotherapy Only",
        "Chemo + Radio"
    ]
    
    results = []
    
    # Ensure patient_data is a DataFrame
    if isinstance(patient_data, pd.Series):
        patient_data = patient_data.to_frame().T
        
    for i, row in patient_data.iterrows():
        best_prob = -1
        best_treatment = ""
        
        probs = []
        simulation_details = {}
        
        for t_idx, (chemo, radio) in enumerate(treatments):
            # Create a copy of the patient row to modify treatment
            temp_row = row.copy()
            temp_row["chemotherapy_done"] = chemo
            temp_row["radiotherapy_done"] = radio
            
            # Ensure columns are in correct order
            temp_row = temp_row[feature_cols]
            
            # Predict survival probability (class 1)
            # XGBoost expects 2D array
            prob = float(model.predict_proba(temp_row.values.reshape(1, -1))[0][1])
            probs.append(prob)
            
            t_name = treatment_names[t_idx]
            simulation_details[t_name] = prob
            
            if prob > best_prob:
                best_prob = prob
                best_treatment = t_name
        
        # Save detailed JSON
        patient_id = str(i)
        result_json = {
            "Patient_ID": patient_id,
            "Best_Treatment": best_treatment,
            "Max_Survival_Prob": best_prob,
            "Simulations": simulation_details,
            "Patient_Features": row.to_dict()
        }
        
        # Handle non-serializable types in row.to_dict() if any (like numpy ints)
        def default_converter(o):
            if isinstance(o, (np.int64, np.int32)):
                return int(o)
            if isinstance(o, (np.float64, np.float32)):
                return float(o)
            return str(o)

        with open(f"{output_dir}/patient_{patient_id}.json", "w") as f:
            json.dump(result_json, f, indent=4, default=default_converter)
            
        results.append({
            "Patient_ID": i,
            "Best_Treatment": best_treatment,
            "Max_Survival_Prob": best_prob,
            "Prob_No_Tx": probs[0],
            "Prob_Chemo": probs[1],
            "Prob_Radio": probs[2],
            "Prob_Both": probs[3]
        })
        
    return pd.DataFrame(results)

def explain_model(model, X_test, feature_cols, sample_patients=None, output_dir="outputs/plots"):
    """Generates SHAP plots for model explainability."""
    print("\nGenerating SHAP explanations...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model to JSON to avoid SHAP parsing issues with sklearn wrapper
    model.save_model("model.json")
    
    # Initialize explainer with the saved model file
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model("model.json")
    
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_test)
    
    # Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
    plt.savefig(f"{output_dir}/shap_summary.png", bbox_inches='tight')
    plt.close()
    print(f"SHAP summary plot saved to '{output_dir}/shap_summary.png'")
    
    # Local Interpretability for specific patients
    if sample_patients is not None:
        explainer_obj = shap.TreeExplainer(booster)
        
        for i, row in sample_patients.iterrows():
            # We need to find the index of this patient in X_test to get the correct shap values
            # OR we can just compute shap values for this specific row
            # It's safer to compute for the specific row
            
            # Ensure row is DataFrame
            row_df = row.to_frame().T[feature_cols]
            shap_explanation = explainer_obj(row_df)
            
            plt.figure()
            shap.plots.waterfall(shap_explanation[0], show=False)
            plt.savefig(f"{output_dir}/patient_{i}_explanation.png", bbox_inches='tight')
            plt.close()
            print(f"SHAP explanation saved for patient {i}")

if __name__ == "__main__":
    # File path
    dataset_path = "datasets/dataset.xlsx"
    
    # 1. Load Data
    df = load_and_clean_data(dataset_path)
    
    # 2. Train Model
    model, feature_cols, label_encoders, X_test, y_test = train_model(df)
    
    # 3. Demonstrate Prescription on a few test patients
    print("\nGenerating prescriptions for 5 random patients from test set...")
    sample_patients = X_test.sample(5, random_state=99)
    prescriptions = prescribe_treatment(model, sample_patients, feature_cols)
    
    print("\nPrescription Results:")
    print(prescriptions.to_string(index=False))
    
    # 4. Explain Model with SHAP
    explain_model(model, X_test, feature_cols, sample_patients=sample_patients)

