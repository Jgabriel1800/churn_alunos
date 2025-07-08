# %%

import pandas as pd

df = pd.read_csv('data/abt_alunos.csv', sep=',')
df.head()
#
# %%
oot = df[df['dtRef']==df['dtRef'].max()].copy()
oot
# criando o out of time
# %%
# separando em treino 
df_train= df[df['dtRef']<df['dtRef'].max()].copy()
df_train['dtRef']

# %%
# Separando target e features
features= df_train.columns[2:-1]
target= 'flagDesistencia'

X,y= df_train[features], df_train[target]
# %%
from sklearn import  model_selection 

X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.2,random_state=42)
# %%
y_train

# %%
print("Taxa variável resposta: ",y_train.mean())
print("Taxa variável resposta: ",y_test.mean())
# %%
