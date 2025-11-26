import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


class OncologyPredictor:
    def __init__(self, model_path="model_package.joblib"):
        """
        Loads the trained model artifacts and prepares the environment.
        """
        print(f"Carregando sistema: {model_path}")
        try:
            artifact = joblib.load(model_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo do modelo não encontrado em {model_path}.")

        self.pipeline = artifact["pipeline"]
        self.expected_columns = artifact["expected_columns"]
        self.threshold = artifact.get("global_threshold", 0.5)

        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.model_xgb = self.pipeline.named_steps["model"]

        print("Inicializando sistema...")
        self.explainer = shap.TreeExplainer(self.model_xgb)
        print("Sistema pronto para uso.")

    def _prepare_data(self, data_dict):
        """
        Converts the input dictionary into a DataFrame with the correct
        column order and fills missing columns with NaN (for the Imputer).
        """
        df = pd.DataFrame([data_dict])

        for col in self.expected_columns:
            if col not in df.columns:
                df[col] = np.nan

        return df[self.expected_columns]

    def predict_risk(self, data_dict):
        """
        Calculates the risk probability.
        Since the model was trained with age-weighted samples,
        we use a standard threshold.
        """
        df_patient = self._prepare_data(data_dict)
        death_prob = self.pipeline.predict_proba(df_patient)[0][1]

        classification = "ALTO RISCO" if death_prob >= self.threshold else "Baixo Risco"

        return {
            "classification": classification,
            "death_probability": death_prob,
            "survival_probability": 1 - death_prob
        }

    def prescribe_treatment(self, data_dict):
        """
        Simulates treatment scenarios (Counterfactuals) and returns
        a recommendation table sorted by survival chance.
        """
        df_base = self._prepare_data(data_dict)

        scenarios = [
            {"name": "Apenas Cirurgia",        "chemo": 0, "radio": 0},
            {"name": "Cirurgia + Químio",      "chemo": 1, "radio": 0},
            {"name": "Cirurgia + Radiação",    "chemo": 0, "radio": 1},
            {"name": "Combo (Químio+Radio)",   "chemo": 1, "radio": 1}
        ]

        results = []

        for scenario in scenarios:
            clone = df_base.copy()

            clone["chemotherapy_done"] = scenario["chemo"]
            clone["radiation_flag"] = scenario["radio"]

            if "therapeutic_plan" in clone.columns:
                clone["therapeutic_plan"] = scenario["chemo"] + scenario["radio"]

            death_prob = self.pipeline.predict_proba(clone)[0][1]

            results.append({
                "Cenário": scenario["name"],
                "Estimativa de Sobrevivência": 1 - death_prob
            })

        df_rec = pd.DataFrame(results).sort_values(by="Estimativa de Sobrevivência", ascending=False)

        baseline_prob = df_rec[df_rec["Cenário"] == "Apenas Cirurgia"]["Estimativa de Sobrevivência"].values[0]
        df_rec["Ganho vs Linha de Base"] = df_rec["Estimativa de Sobrevivência"] - baseline_prob

        return df_rec

    def explain_with_shap(self, data_dict):
        """
        Generates and displays the SHAP Waterfall plot for the patient.
        """
        df_patient = self._prepare_data(data_dict)

        X_transformed = self.preprocessor.transform(df_patient)
        feature_names = self.preprocessor.get_feature_names_out()
        clean_names = [f.split("__")[-1] for f in feature_names]

        shap_values = self.explainer(X_transformed)
        shap_values.feature_names = clean_names

        plt.figure()
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        plt.title(f"Fatores de Risco (Idade: {data_dict.get("diagnosis_age")})")
        plt.show()


def prescribe_treatment(system, patient):
    """
    Generate all analysis and SHAP values.
    """
    print("\n--- 1. Análise de Risco ---")
    risk = system.predict_risk(patient)
    print(f"Classificação: {risk["classification"]}")
    print(f"Probabilidade de Sobrevivência (Atual): {risk["survival_probability"]:.2%}")

    print("\n--- 2. Simulação de Tratamento ---")
    recommendations = system.prescribe_treatment(patient)

    pd.options.display.float_format = "{:.2%}".format
    print(recommendations.to_string(index=False))

    print("\n--- 3. Explicação Visual (SHAP) ---")
    system.explain_with_shap(patient)


if __name__ == "__main__":
    system = OncologyPredictor("model_package.joblib")

    patient = {
        "diagnosis_age": 42,
        "lymph_nodes": 4,
        "malignant_tumors": 3,
        "radiation_type": "Beam radiation",
        "chemotherapy_done": 1,
        "radiation_sequence": "Intraoperative rad with other rad before/after surgery",
        "estrogen_info": "Positive",
        "progesterone_info": "Positive",
        "tumor_size": "105",
        "her2_info": "Positive",
        "nodes_examined": 15,
        "cause_of_death": "Alive",
        "race": "White",
        "sex": "Female",
        "vital_status": "Alive",
        "diagnosis_year": 2010,
        "treatment_year": 2016,
        "num_screening": 4,
        "vital_status_5y": "Alive",
    }

    prescribe_treatment(system, patient)
