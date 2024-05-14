import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler

treino = pd.read_csv('iris2.data')
teste = pd.read_csv('iris.data')

scaler = MinMaxScaler()
treinodataNormalized = scaler.fit_transform(treino.drop(columns="class"))

x_treino = pd.DataFrame(treinodataNormalized, columns=treino.drop(columns="class").columns)
y_treino = treino['class']


testeDataNormalized = scaler.transform(teste.drop(columns="class"))
x_teste = pd.DataFrame(testeDataNormalized, columns=teste.drop(columns="class").columns)
y_teste = teste['class']

k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_treino, y_treino)

prev = knn.predict(x_teste)
accuracy = accuracy_score(y_teste, prev) * 100
matrix = confusion_matrix(y_teste, prev)

print("Matriz de confusão:\n", matrix)
print('Acurácia: %.2f%%' % accuracy)
