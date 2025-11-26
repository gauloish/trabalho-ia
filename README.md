# Modelo de Prescrição de Tratamento com Machine Learning

Este projeto implementa um modelo de Machine Learning para auxiliar na decisão clínica, prescrevendo o melhor tratamento para pacientes com câncer de mama visando maximizar a probabilidade de sobrevivência em 5 anos.

## 🎯 Objetivo
O objetivo é utilizar uma abordagem **S-Learner** com **XGBoost** para prever a sobrevivência do paciente sob diferentes cenários de tratamento (Quimioterapia, Radioterapia, Ambos ou Nenhum) e recomendar a opção com maior chance de sucesso.

## 🚀 Funcionalidades
- **Processamento de Dados**: Limpeza e engenharia de features a partir do dataset clínico.
- **Treinamento do Modelo**: Treina um classificador XGBoost para prever o status vital em 5 anos.
- **Prescrição de Tratamento**: Simula todas as combinações de tratamento para um paciente e recomenda a melhor.
- **Explicabilidade (SHAP)**: Gera gráficos para explicar quais fatores influenciaram as decisões do modelo (Global e Local).

## 🛠️ Instalação

Certifique-se de ter o Python instalado. Instale as dependências necessárias:

```bash
pip install -r requirements.txt
```

## ▶️ Como Usar

### 1. Executar o Script Principal
Para gerar a previsão de sobrevivência, a prescrição de tratamento e os gráficos de explicação do SHAP para uma amostra de paciente, basta executar:

```bash
python treatment_prescription.py
```

Isso irá:
- Carregar o modelo `model_package.joblib`.
- Mostrar a classificação do paciente (`ALTO RISCO` ou `Baixo Risco`).
- Mostrar a probabilidade de sobrevivência do paciente.
- Mostrar as estimativas para cada cenário de tratamento.
- Gerar o gráfico com os valores SHAP do modelo para o paciente.

### 2. Testar em Qualquer Paciente (Novos Dados)
Para usar o modelo em novos pacientes, você pode importar a função `prescribe_treatment` no seu próprio script ou notebook.

Exemplo de uso:

```python
from treatment_prescription import OncologyPredictor

# 1. Carregar o modelo salvo
system = OncologyPredictor("model_package.joblib")

# 2. Inserir dados do novo paciente
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

# 3. Gerar Prescrição
prescribe_treatment(system, patient)
```

## 📊 Entendendo os Resultados

A função de prescrição retorna os dados:
- **Classificação**: A classificação de risco do paciente ("ALTO RISCO" ou "Baixo Risco).
- **Probabilidade de Sobrevivência**: A probabilidade estimada de sobrevivência atual.
- **Estimativa de Sobrevivência com os Tratamentos**: As probabilidades calculadas para cada opção de tratamento.

## 🔍 Explicabilidade
O script gera automaticamente:

- **Gráfico SHAP**: Mostra quais características (ex: idade, tamanho do tumor) mais impactam a sobrevivência geral.
