# passo 1: impotar a base de dados 
import pandas as pd

tabela = pd.read_csv('treino.csv')
display(tabela)


# passo 2: preparar a base de dados para a IA
tabela.info()

# y e quem eu quero prever
y = tabela['target']

# x e o que eu tenho para prever
x = tabela.drop(columns=['id','target'])

#separar dados de treino e dados de teste
from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y)



# passo 3: treinar a IA 
# criar modelo

#arvore de desisao -> RandomForest
#visinhos procimos -> Neirest Neiighbor

#importar modelo
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

#criar IA
modelo_arvore = DecisionTreeClassifier()
modelo_floresta = RandomForestClassifier()
modelo_knn = KNeighborsClassifier()
modelo_naive_bayes = GaussianNB()
modelo_svm = SVC()
modelo_linear_model = LogisticRegression(max_iter=1000)
modelo_xgboost = XGBClassifier(objective='multi:softprob', learning_rate=0.1, n_estimators=500, 
                               max_depth=6, min_child_weight=1, colsample_bytree=0.8, subsample=0.8, gamma=0.1)


#treinar IA
modelo_arvore.fit(x_treino, y_treino)
modelo_floresta.fit(x_treino, y_treino)
modelo_knn.fit(x_treino, y_treino)
modelo_naive_bayes.fit(x_treino, y_treino)
modelo_svm.fit(x_treino, y_treino)
modelo_linear_model.fit(x_treino, y_treino)
modelo_xgboost.fit(x_treino, y_treino)


# passo 4: qual o melhor modelo de IA
previsao_arvore = modelo_arvore.predict(x_teste)
previsao_floresta = modelo_floresta.predict(x_teste)
previsao_knn = modelo_knn.predict(x_teste) 
previsao_naive_bayes = modelo_naive_bayes.predict(x_teste)
previsao_svm = modelo_svm.predict(x_teste)
previsao_linear_model = modelo_linear_model.predict(x_teste)
previsao_xgboost = modelo_xgboost.predict(x_teste)

from sklearn.metrics import accuracy_score
display(accuracy_score(y_teste, previsao_arvore))
display(accuracy_score(y_teste, previsao_floresta))
display(accuracy_score(y_teste, previsao_knn))
display(accuracy_score(y_teste, previsao_naive_bayes))
display(accuracy_score(y_teste, previsao_svm))
display(accuracy_score(y_teste, previsao_linear_model))
display(accuracy_score(y_teste, previsao_xgboost))



# novo treiramento com ajustes
# comsultar arquivo .txt

modelo_floresta2 = RandomForestClassifier(
    n_estimators=300, 
    max_depth=30, 
    min_samples_split=5, 
    min_samples_leaf=2, 
    max_features='sqrt', 
    bootstrap=True,  
    class_weight='balanced', 
    random_state=42)

modelo_knn2 = KNeighborsClassifier(
    n_neighbors=5,
    weights='uniform',
    algorithm='auto',
    leaf_size=30,
    p=2,
    metric='minkowski',
    metric_params=None,
    n_jobs=None)

modelo_xgboost2 = XGBClassifier(
    objective='multi:softprob',
    learning_rate=0.1,
    n_estimators=500,
    max_depth=6, 
    min_child_weight=1, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    gamma=0.1)

modelo_floresta2.fit(x_treino, y_treino)
modelo_knn2.fit(x_treino, y_treino)
modelo_xgboost2.fit(x_treino, y_treino)

previsao_floresta2 = modelo_floresta2.predict(x_teste)
previsao_knn2 = modelo_knn2.predict(x_teste) 
previsao_xgboost2 = modelo_xgboost2.predict(x_teste) 

display(accuracy_score(y_teste, previsao_floresta2))
display(accuracy_score(y_teste, previsao_knn2))
display(accuracy_score(y_teste, previsao_xgboost2))


# passo 5: usar o melhor modelo para faser a previsao
# melhor modelo arvore de desisao

#importar tabela de novos cliemtes
tabela_nova = pd.read_csv('teste.csv')

# Armazena a coluna 'id' separadamente
ids = tabela_nova["id"]

# Remove a coluna 'id' antes da previsão
tabela_nova_sem_id = tabela_nova.drop(columns=['id'], errors='ignore')

# Faz a previsão
nova_previsao = modelo_arvore.predict(tabela_nova_sem_id)

# Criar um DataFrame com os IDs e as previsões
resultado = pd.DataFrame({"id": ids, "target": nova_previsao})

display(tabela_nova)

display(nova_previsao)

# Exibir o resultado
display(resultado)


# Salvando para CSV
df = pd.DataFrame(resultado)

df.to_csv("resposta.csv", index=False, header=True, encoding="utf-8")
