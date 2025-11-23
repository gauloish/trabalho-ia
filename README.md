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

*Nota: Este projeto utiliza `xgboost==2.0.3` para compatibilidade com o SHAP.*

## ▶️ Como Usar

### 1. Executar o Script Principal
Para treinar o modelo, avaliar a performance e gerar exemplos de prescrição, execute:

```bash
python treatment_prescription.py
```

Isso irá:
- Carregar os dados de `datasets/dataset.xlsx`.
- Treinar o modelo.
- Exibir a Acurácia e AUC no terminal.
- Mostrar prescrições para 5 pacientes aleatórios.
- Gerar dois arquivos de imagem:
    - `shap_summary.png`: Importância global das variáveis.
    - `shap_patient_explanation.png`: Explicação detalhada para um paciente.

### 2. Testar em Qualquer Paciente (Novos Dados)
Para usar o modelo em novos pacientes, você pode importar a função `prescribe_treatment` no seu próprio script ou notebook.

Exemplo de uso:

```python
import pandas as pd
from treatment_prescription import load_and_clean_data, train_model, prescribe_treatment

# 1. Carregar dados e treinar o modelo (ou carregar um modelo salvo)
df = load_and_clean_data("datasets/dataset.xlsx")
model, feature_cols, label_encoders, _, _ = train_model(df)

# 2. Criar dados de um novo paciente (exemplo)
# Certifique-se de usar as mesmas colunas e codificações usadas no treinamento
novo_paciente = {
    "diagnosis_age": 55,
    "lymph_nodes": 2,
    "malignant_tumors": 1,
    "tumor_size": 25,
    "nodes_examined": 10,
    "diagnosis_year": 2015,
    "estrogen_info": "Positive", # Precisa ser codificado numericamente como no treino
    "progesterone_info": "Positive",
    # ... adicione todas as features necessárias
}

# Nota: Para simplificar, recomenda-se passar um DataFrame com a estrutura correta
# ou reutilizar uma linha do dataset original para teste.

# Exemplo pegando um paciente do dataset original:
paciente_teste = df.iloc[[0]][feature_cols] 

# 3. Gerar Prescrição
resultado = prescribe_treatment(model, paciente_teste, feature_cols)
print(resultado)
```

## 📊 Entendendo os Resultados

A função de prescrição retorna uma tabela com:
- **Best_Treatment**: O tratamento recomendado (ex: "Radiotherapy Only").
- **Max_Survival_Prob**: A probabilidade estimada de sobrevivência com o melhor tratamento.
- **Prob_No_Tx, Prob_Chemo, etc.**: As probabilidades calculadas para cada opção de tratamento.

## 🔍 Explicabilidade
O script gera automaticamente:
- **SHAP Summary Plot**: Mostra quais características (ex: idade, tamanho do tumor) mais impactam a sobrevivência geral.
- **Waterfall Plot**: Mostra passo-a-passo como o modelo chegou à probabilidade de sobrevivência para um paciente específico.
