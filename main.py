import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Carregando os dados
data = pd.read_csv('marketplace.csv')

# Separando os dados em variáveis de entrada (características) e variável de saída (preço)
X = data[['sku_name', 'product_type', 'merchant_city','cod']]
y = data['price']

# Codificando características categóricas usando one-hot encoding
X_encoded = pd.get_dummies(X)

# Normalizando as características
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_encoded)

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Criando o modelo de regressão linear
model = LinearRegression()

# Treinando o modelo com os dados de treinamento
model.fit(X_train, y_train)

# Fazendo previsões com o modelo treinado
y_pred = model.predict(X_test)

# Avaliando o desempenho do modelo
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
