#%%
import pandas as pd
import mlflow 

mlflow.set_tracking_uri("http://localhost:5000")
# importe modelo

models= mlflow.search_registered_models(filter_string="name='model_churn'")
latest_version= max([i.version for i in models[0].latest_versions])

model= mlflow.sklearn.load_model(f"models:/model_churn/{latest_version}")
features= model.feature_names_in_

# %%
#Importando os novos dados
df=pd.read_csv('data/abt_alunos.csv', sep=',')
amostra= df[df['dtRef']==df['dtRef'].max()].sample(3)
amostra= amostra.drop('flagDesistencia',axis=1)
# %%
#Predicao
predicao=model.predict_proba(amostra[features])[:, 1]
amostra['proba_new']= predicao
amostra
# %%
models= mlflow.search_registered_models(filter_string="name='model_churn'")
latest_version= max([i.version for i in models[0].latest_versions])
latest_version
# %%
