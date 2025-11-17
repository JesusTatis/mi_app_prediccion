import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Cargar datos
DATA_PATH = "data/dataset_procesado.csv"
df = pd.read_csv(DATA_PATH)

# 2. Definir X e y
X = df.drop(columns=["GradeClass", "StudentID"])
y = df["GradeClass"]

# 3. ¡IMPORTANTE! Dividir los datos
# Usaremos 80% para entrenar y 20% para probar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Entrenar y Evaluar REGRESIÓN LINEAL ---
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred_lr = model_lr.predict(X_test)

# Calcular métricas
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("--- Resultados Regresión Lineal ---")
print(f"Error Cuadrático Medio (MSE): {mse_lr}")
print(f"Coeficiente de Determinación (R²): {r2_lr}")
print("\n")

# --- 5. Entrenar y Evaluar RANDOM FOREST ---
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred_rf = model_rf.predict(X_test)

# Calcular métricas
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("--- Resultados Random Forest ---")
print(f"Error Cuadrático Medio (MSE): {mse_rf}")
print(f"Coeficiente de Determinación (R²): {r2_rf}")