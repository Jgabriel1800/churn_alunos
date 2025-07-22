# %%

import mlflow.sklearn
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

from feature_engine import discretisation  , encoding
from sklearn import pipeline

#Discretizar
tree_discretization= discretisation.DecisionTreeDiscretiser(
    variables = best_features, regression= False, cv=3
)



#ONEHOT
onehot= encoding.OneHotEncoder(variables=best_features, ignore_format= True)


#%%
# %%

#MODEL
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import ensemble

#model= linear_model.LogisticRegression(penalty=None, random_state=42,max_iter=1000000)
#model=naive_bayes.BernoulliNB()
#model=ensemble.RandomForestClassifier(random_state=42, min_samples_leaf=20,n_jobs=-1,
         #                             n_estimators=500)

model= tree.DecisionTreeClassifier(random_state=42, min_samples_leaf=20)

model_pipeline=pipeline.Pipeline(steps=[('discetrizar',tree_discretization),
                                        ('onehot',onehot),
                                        ('model',model)])

import mlflow 
from sklearn import metrics 


model_pipeline.fit(X_train, y_train)


y_train_predict= model_pipeline.predict(X_train)
y_train_proba= model_pipeline.predict_proba(X_train)[:, 1]

acc_train= metrics.accuracy_score(y_train, y_train_predict)
auc_train= metrics.roc_auc_score(y_train, y_train_proba)
roc_train=metrics.roc_curve(y_train, y_train_proba)

print("Acurácia treino: ", acc_train)
print("AUC treino: ", auc_train)
y_test_predict= model_pipeline.predict(X_test)
y_test_proba= model_pipeline.predict_proba(X_test)[:, 1]

acc_test= metrics.accuracy_score(y_test, y_test_predict)
auc_test= metrics.roc_auc_score(y_test, y_test_proba)
roc_test=metrics.roc_curve(y_test, y_test_proba)

print("Acurácia test: ", acc_test)
print("AUC test: ", auc_test)


y_oot_predict= model_pipeline.predict(oot[features])
y_oot_proba= model_pipeline.predict_proba(oot[features])[:, 1]

acc_oot= metrics.accuracy_score(oot[target], y_oot_predict)
auc_oot= metrics.roc_auc_score(oot[target], y_oot_proba)
roc_oot=metrics.roc_curve(oot[target], y_oot_proba)

print("Acurácia oot: ", acc_oot)
print("AUC oot: ", auc_oot)

    
# %%
from matplotlib import pyplot as plt
plt.plot(roc_train[0], roc_train[1])
plt.plot(roc_test[0], roc_test[1])
plt.plot(roc_oot[0], roc_oot[1])
plt.grid(True)
plt.title("Curva ROC")
plt.legend([
    f"Treino : {100*auc_train:.2f}",
    f"Teste : {100*auc_test:.2f}",
    f"OOT : {100*auc_oot:.2f}" 
])
# %%
