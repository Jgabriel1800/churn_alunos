# %%

import pandas as pd

df=pd.read_csv('data/abt_alunos.csv', sep=',')
df.head()

#
# %%
oot = df[df['dtRef']==df['dtRef'].max()].copy()
oot
# criando o out of time
# %%
# separando em treino SS
df_train= df[df['dtRef']<df['dtRef'].max()].copy()
df_train['dtRef']

# %%
# Separando target e features
features= df_train.columns[2:-1]
target= 'flagDesistencia'

X,y= df_train[features], df_train[target]
# %%

# SAMPLE
from sklearn import  model_selection 

X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
# %%
y_train

# %%
# quero saber se quando separei as amostras as respostas são proporcionais
print("Taxa variável resposta geral: ",y.mean())
print("Taxa variável resposta treino: ",y_train.mean())
print("Taxa variável resposta teste: ",y_test.mean())
# Evento raro é quando a taxa de resposta é menor que 5% ou maior que 95%,não é o caso aqui
# %%
X_train.isna().sum().sort_values(ascending=False)
#Não tem dados faltantes 

# %%
df_analise =X_train.copy()
df_analise[target]=y_train
sumario= df_analise.groupby(by=target).agg(['mean','median']).T
sumario
# %%
sumario['diff_abs']=sumario[0]-sumario[1]
sumario['dif_rel']=sumario[0]/sumario[1]
sumario.sort_values(by=['dif_rel'],ascending=False)
# %%
from sklearn import tree
arvore = tree.DecisionTreeClassifier( random_state=42)
arvore.fit(X_train, y_train)
# %%
#analisando quais features foram mais importantes para a árvore de decisão
feature_importance=(pd.Series (arvore.feature_importances_,index= X_train.columns).sort_values(ascending=False).reset_index())
# %%
feature_importance['acum.'] = feature_importance[0].cumsum()
feature_importance
# %%
feature_importance[feature_importance['acum.'] < 0.96]
# %%

best_features = feature_importance[feature_importance['acum.'] < 0.96]['index'].tolist()
best_features
# %%

#MODIFY

from feature_engine import discretisation  

tree_discretization= discretisation.DecisionTreeDiscretiser(
    variables = best_features, regression= False, cv=3
)

tree_discretization.fit(X_train[best_features], y_train)
# %%
X_train.head()
# %%
X_train_transformed = tree_discretization.transform(X_train[best_features])
X_train_transformed
# %%

#MODEL
from sklearn import linear_model
reg= linear_model.LogisticRegression(penalty=None, random_state=42)
reg.fit(X_train_transformed, y_train)
# %%
from sklearn import metrics 

y_train_predict= reg.predict(X_train_transformed)
y_train_proba= reg.predict_proba(X_train_transformed)[:, 1]

acc_train= metrics.accuracy_score(y_train, y_train_predict)
auc_train= metrics.roc_auc_score(y_train, y_train_proba)

print("Acurácia treino: ", acc_train)
print("AUC treino: ", auc_train)
# %%
X_test_transform = tree_discretization.transform(X_test[best_features])    

y_test_predict= reg.predict(X_test_transform)
y_test_proba= reg.predict_proba(X_test_transform)[:, 1]

acc_test= metrics.accuracy_score(y_test, y_test_predict)
auc_test= metrics.roc_auc_score(y_test, y_test_proba)

print("Acurácia test: ", acc_test)
print("AUC test: ", auc_test)
# %%

oot_transform = tree_discretization.transform(oot[best_features])

y_oot_predict= reg.predict(oot_transform)
y_oot_proba= reg.predict_proba(oot_transform)[:, 1]

acc_oot= metrics.accuracy_score(oot[target], y_oot_predict)
auc_oot= metrics.roc_auc_score(oot[target], y_oot_proba)

print("Acurácia oot: ", acc_oot)
print("AUC oot: ", auc_oot)
# %%
