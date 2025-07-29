# Predição de Evasão Escolar com Machine Learning

Este projeto tem como objetivo construir um modelo preditivo capaz de identificar estudantes com alto risco de evasão em cursos educacionais, utilizando técnicas de Machine Learning e um pipeline integrado com rastreamento via MLflow.

##  Objetivo

Antecipar comportamentos de desistência entre alunos, permitindo que instituições educacionais possam intervir proativamente com estratégias de retenção.

##  Estrutura do Projeto

- `train.py`: script principal de treinamento e avaliação do modelo, com pipeline completo e rastreamento automático de experimentos via MLflow.
- `predict.py`: script de inferência que carrega automaticamente a última versão registrada do modelo para realizar previsões em novos dados.
- `data/abt_alunos.csv`: dataset contendo informações dos alunos com data de referência, atributos descritivos e variável alvo (`flagDesistencia`).
- `mlruns/`: diretório local de rastreamento de experimentos via MLflow (gerado automaticamente ao rodar o `train.py`).

##  Pipeline de Modelagem

1. **Separação Temporal dos Dados**  
   - Dados anteriores à última `dtRef` são utilizados para treino e teste.  
   - A última `dtRef` é reservada como conjunto de validação out-of-time (OOT).

2. **Feature Engineering**  
   - Seleção de variáveis com base na importância em uma árvore de decisão.  
   - Discretização supervisionada com `DecisionTreeDiscretiser`.  
   - Codificação categórica com `OneHotEncoder`.

3. **Modelagem e Validação**  
   - Algoritmo: `RandomForestClassifier`  
   - Otimização com `GridSearchCV` (validação cruzada 3-fold)  
   - Métrica de avaliação: AUC (Área sob a Curva ROC)

4. **Rastreamento de Experimentos**  
   - Uso de `mlflow.sklearn.autolog()` para registro automático de hiperparâmetros, métricas, modelo e artefatos.

##  Resultados

O modelo foi avaliado em três diferentes conjuntos de dados:

- **Treinamento**  
  - Acurácia: 76,1%  
  - AUC: 84,7%

- **Teste**  
  - Acurácia: 75,6%  
  - AUC: 82,6%

- **Out-of-Time (OOT)**  
  - Acurácia: 77,8%  
  - AUC: 83,5%


##  Inferência

Para realizar predições com o modelo treinado:

```bash
python predict.py
