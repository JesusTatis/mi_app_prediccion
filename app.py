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

    # ======================================================
    # 1. DISTRIBUCIÓN DE LA EDAD
    # ======================================================
    fig_age = px.histogram(
        df,
        x="Age",
        nbins=15,
        title="Distribución de la Edad de los Estudiantes",
        color_discrete_sequence=["#EF553B"]
    )
    fig_age.update_layout(
        xaxis_title="Edad",
        yaxis_title="Cantidad de estudiantes"
    )
    st.plotly_chart(fig_age, use_container_width=True)

    # ======================================================
    # 2. DISTRIBUCIÓN DEL GÉNERO
    # ======================================================
    df_gender = df.copy()
    df_gender["GenderLabel"] = df_gender["Gender"].map({
        0: "0 - Male (Hombre)",
        1: "1 - Female (Mujer)"
    })

    fig_gender = px.histogram(
        df_gender,
        x="GenderLabel",
        title="Distribución por Género",
        color_discrete_sequence=["#00CC96"]
    )
    fig_gender.update_layout(
        xaxis_title="Género",
        yaxis_title="Cantidad de estudiantes"
    )
    st.plotly_chart(fig_gender, use_container_width=True)

    # ======================================================
    # 3. NIVEL EDUCATIVO DE LOS PADRES (ORDENADO 0 → 4)
    # ======================================================
    df_parentedu = df.copy()
    df_parentedu["ParentalEducationLabel"] = df_parentedu["ParentalEducation"].map({
        0: "0 - None",
        1: "1 - High School",
        2: "2 - Some College",
        3: "3 - Bachelor's",
        4: "4 - Higher"
    })

    orden_categorias = [
        "0 - None",
        "1 - High School",
        "2 - Some College",
        "3 - Bachelor's",
        "4 - Higher"
    ]

    fig_parentedu = px.histogram(
        df_parentedu,
        x="ParentalEducationLabel",
        category_orders={"ParentalEducationLabel": orden_categorias},
        title="Distribución del Nivel Educativo de los Padres",
        color_discrete_sequence=["#AB63FA"]
    )
    fig_parentedu.update_layout(
        xaxis_title="Nivel educativo de los padres",
        yaxis_title="Cantidad de estudiantes"
    )
    st.plotly_chart(fig_parentedu, use_container_width=True)

    # ======================================================
    # 4. DISTRIBUCIÓN DE LA VARIABLE OBJETIVO (GRADECLASS)
    # ======================================================
    fig_dist = px.histogram(
        df,
        x="GradeClass",
        nbins=20,
        title="Distribución de Calificaciones (GradeClass)",
        color_discrete_sequence=["#636EFA"]
    )
    fig_dist.update_layout(
        xaxis_title="Clase de calificación (GradeClass)",
        yaxis_title="Cantidad de estudiantes"
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# --- Información del modelo ---
st.markdown("---")
st.caption("📘 Proyecto desarrollado con Streamlit, Scikit-learn y Plotly para la visualización interactiva de modelos de predicción.")
