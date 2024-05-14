import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('iris2.data')
df = pd.DataFrame(data)

# Normalizar as características
scaler = MinMaxScaler()
personDataNormalized = scaler.fit_transform(data.drop(columns=['class']))  # Normalizar todas as colunas, exceto 'variety'
    
# Separar características e classe alvo
x = personDataNormalized
y = data['class']  # Manter a classe alvo no seu formato original
# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=7)

# Criar o classificador KNN
k = 3
knn = KNeighborsClassifier(n_neighbors=k)

# Treinar o classificador
knn.fit(X_train, y_train)

# Fazer previsões
y_pred = knn.predict(X_test)
accuraccy = accuracy_score(y_test, y_pred) * 100

# Imprimir a matriz de confusão e a acurácia
print("Matriz de confusão: ", confusion_matrix(y_test, y_pred))
print('Acurácia: %.2f%%' % accuraccy)