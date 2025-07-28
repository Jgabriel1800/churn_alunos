# 1. Carregar o modelo
#%%
import pandas as pd
import mlflow
import mlflow.sklearn

model_df = pd.read_pickle("model.pkl")
model = model_df['model']
features = model_df['features']

#%%
# 2. Carregar os dados para predição (OOT, novos alunos, etc)
df = pd.read_csv("..\\churn_alunos\\data\\abt_alunos.csv")
amostra = df[df['dtRef'] == df['dtRef'].max()].copy()

# 3. Usar diretamente o DataFrame com TODAS as colunas (sem filtrar `features`)
# Isso porque o pipeline interno já vai selecionar `best_features` na hora do `fit_transform`
#%%
# 4. Fazer a predição
predicao = model.predict_proba(amostra)[:, 1]



# %%
