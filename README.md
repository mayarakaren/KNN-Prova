# Projeto de Classificação de Iris com K-Nearest Neighbors (KNN)

Este repositório contém dois scripts para a classificação da base de dados Iris utilizando o algoritmo K-Nearest Neighbors (KNN). Ambos os scripts utilizam a biblioteca `scikit-learn` para realizar a normalização dos dados, treinamento do modelo, e avaliação da acurácia do classificador.

## Scripts

1. **Script 1: Treinamento e teste no mesmo conjunto de dados (com divisão interna)**
2. **Script 2: Treinamento e teste em conjuntos de dados separados**

### Script 1: Treinamento e Teste no Mesmo Conjunto de Dados

Este script carrega o dataset `iris2.data`, normaliza os dados, divide-os em conjuntos de treinamento e teste, treina o modelo KNN e avalia sua acurácia.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler

# Carregar os dados
data = pd.read_csv('iris2.data')
df = pd.DataFrame(data)

# Normalizar as características
scaler = MinMaxScaler()
personDataNormalized = scaler.fit_transform(data.drop(columns=['class']))

# Separar características e classe alvo
x = personDataNormalized
y = data['class']

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=7)

# Criar e treinar o classificador KNN
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Fazer previsões
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

# Imprimir a matriz de confusão e a acurácia
print("Matriz de confusão: ", confusion_matrix(y_test, y_pred))
print('Acurácia: %.2f%%' % accuracy)
```

### Script 2: Treinamento e Teste em Conjuntos de Dados Separados

Este script carrega datasets distintos para treino (`iris2.data`) e teste (`iris.data`), normaliza os dados, treina o modelo KNN no conjunto de treino e avalia sua acurácia no conjunto de teste.

```python
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler

# Carregar os dados
treino = pd.read_csv('iris2.data')
teste = pd.read_csv('iris.data')

# Normalizar os dados de treinamento
scaler = MinMaxScaler()
treinodataNormalized = scaler.fit_transform(treino.drop(columns="class"))

x_treino = pd.DataFrame(treinodataNormalized, columns=treino.drop(columns="class").columns)
y_treino = treino['class']

# Normalizar os dados de teste
testeDataNormalized = scaler.transform(teste.drop(columns="class"))
x_teste = pd.DataFrame(testeDataNormalized, columns=teste.drop(columns="class").columns)
y_teste = teste['class']

# Criar e treinar o classificador KNN
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_treino, y_treino)

# Fazer previsões
prev = knn.predict(x_teste)
accuracy = accuracy_score(y_teste, prev) * 100
matrix = confusion_matrix(y_teste, prev)

# Imprimir a matriz de confusão e a acurácia
print("Matriz de confusão:\n", matrix)
print('Acurácia: %.2f%%' % accuracy)
```

## Pré-requisitos

- Python 3.x
- Bibliotecas:
  - pandas
  - scikit-learn

Você pode instalar as bibliotecas necessárias utilizando o comando:

```bash
pip install pandas scikit-learn
```

## Execução

1. Certifique-se de que os arquivos `iris2.data` e `iris.data` estejam no mesmo diretório dos scripts.
2. Execute os scripts utilizando o Python:

```bash
python script1.py
python script2.py
```

Substitua `script1.py` e `script2.py` pelos nomes reais dos arquivos dos scripts.

## Resultados

Os scripts imprimirão a matriz de confusão e a acurácia do modelo KNN após a execução. Esses resultados ajudam a avaliar a performance do modelo na classificação das espécies de Iris.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests para melhorias ou correções.

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

---
