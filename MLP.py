import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Carregando os dados
data = pd.read_csv('marketplace.csv')

# Separando os dados em variáveis de entrada (características) e variável de saída (preço)
X = data[['sku_name','merchant_city', 'product_type', 'cod']]
y = data['price']

# Copiando o DataFrame original
data_normalized = data.copy()

# Codificando características categóricas usando one-hot encoding
X_encoded = pd.get_dummies(X)

# Normalizando as características
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_encoded)

# Criando um novo DataFrame com as características normalizadas
data_normalized.drop(['sku_name','merchant_city', 'product_type', 'cod'], axis=1, inplace=True)
data_normalized = pd.concat([data_normalized, pd.DataFrame(X_normalized)], axis=1)

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Criando o modelo MLP
model = MLPRegressor(hidden_layer_sizes=(10, 50, 80, 90), activation='relu', random_state=42, verbose=1)

# Treinando o modelo com os dados de treinamento
model.fit(X_train, y_train)

# Fazendo previsões com o modelo treinado
y_pred = model.predict(X_test)

# Avaliando o desempenho do modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print('Mean Squared Error:', mse)
print('R²:', r2)
print('Mean Absolute Error:', mae)

# Plotando um gráfico de dispersão dos valores reais versus os valores previstos
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')  # Linha de referência
plt.xlabel('Valor Real')
plt.ylabel('Valor Previsto')
plt.title('Comparação entre Valores Reais e Previstos')
plt.show()

# Exibindo os primeiros registros do DataFrame com as características normalizadas
print(data_normalized.head())
