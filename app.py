import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from models.linear_regression_model import train_linear_regression, predict_linear
from models.random_forest_model import train_random_forest, predict_rf

# --- Configuración inicial ---
st.set_page_config(page_title="Predicción de Calificación Final", layout="wide")

# --- Cargar dataset ---
DATA_PATH = "data/dataset_procesado.csv"
df = pd.read_csv(DATA_PATH)

st.title("🎓 Predicción de Calificación Final (GradeClass)")
st.write("Esta aplicación permite predecir la calificación final de un estudiante en base a características demográficas y de preparación académica.")

# --- Verificación de columnas ---
if "StudentID" not in df.columns or "GradeClass" not in df.columns:
    st.error("El dataset debe tener las columnas 'StudentID' y 'GradeClass'.")
    st.stop()

# --- Sidebar ---
st.sidebar.header("⚙️ Configuración del modelo")
modelo = st.sidebar.selectbox("Selecciona el modelo de predicción", ["Regresión Lineal", "Random Forest"])
mostrar_graficas = st.sidebar.checkbox("Mostrar análisis visual", value=True)

# --- Seleccionar estudiante ---
estudiante_id = st.selectbox("Selecciona el ID del estudiante", df["StudentID"].unique())
fila_estudiante = df[df["StudentID"] == estudiante_id]

st.subheader("🧾 Datos del estudiante seleccionado")
st.dataframe(fila_estudiante)

# --- Preparar datos ---
X = df.drop(columns=["GradeClass", "StudentID"])
y = df["GradeClass"]

X_estudiante = fila_estudiante.drop(columns=["GradeClass", "StudentID"])

# --- Entrenar y predecir ---
if modelo == "Regresión Lineal":
    model = train_linear_regression(X, y)
    prediccion = predict_linear(model, X_estudiante)
else:
    model = train_random_forest(X, y)
    prediccion = predict_rf(model, X_estudiante)

valor_real = fila_estudiante["GradeClass"].values[0]

# --- Mostrar resultados ---
col1, col2 = st.columns(2)
with col1:
    st.metric("🎯 Predicción", f"{prediccion[0]:.2f}")
with col2:
    st.metric("📘 Valor real", f"{valor_real:.2f}")

# --- Gráfico de comparación ---
fig_pred = go.Figure()
fig_pred.add_trace(go.Bar(x=["Predicción"], y=[prediccion[0]], name="Predicción", marker_color="royalblue"))
fig_pred.add_trace(go.Bar(x=["Valor Real"], y=[valor_real], name="Valor Real", marker_color="lightgreen"))
fig_pred.update_layout(
    title=f"Comparación entre Predicción y Valor Real del Estudiante {estudiante_id}",
    yaxis_title="Calificación (GradeClass)",
    xaxis_title="",
    template="plotly_white"
)
st.plotly_chart(fig_pred, use_container_width=True)

# --- Análisis visual adicional ---
if mostrar_graficas:
    st.subheader("📊 Análisis visual del dataset")

    # Distribución de la variable objetivo
    fig_dist = px.histogram(df, x="GradeClass", nbins=20, title="Distribución de Calificaciones (GradeClass)",
                            color_discrete_sequence=["#636EFA"])
    st.plotly_chart(fig_dist, use_container_width=True)

# --- Información del modelo ---
st.markdown("---")
st.caption("📘 Proyecto desarrollado con Streamlit, Scikit-learn y Plotly para la visualización interactiva de modelos de predicción.")
