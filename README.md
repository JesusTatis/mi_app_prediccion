# 🎓 Predicción del Rendimiento Académico — *GradeClass Predictor*

## 📘 Descripción del Proyecto
Este proyecto utiliza **aprendizaje automático (Machine Learning)** para **predecir la clase de calificación final (`GradeClass`)** de los estudiantes, basándose en variables demográficas y académicas como:
- Género  
- Nivel educativo de los padres  
- Tipo de almuerzo  
- Curso de preparación previo  
- Entre otros factores  

La aplicación fue desarrollada con **Streamlit**, lo que permite una interfaz interactiva para explorar los datos, entrenar el modelo y hacer predicciones en tiempo real.

---

## 🧠 Objetivo
El objetivo principal es **predecir el rendimiento académico (GradeClass)** a partir de la información de un estudiante (identificado por su `StudentID`) y **entender cómo influyen las variables demográficas y de preparación en su desempeño**.

---

## ⚙️ Estructura del Proyecto

mi_proyecto_prediccion/
│
├── app.py
├── models/
│   ├── random_forest_model.py
│   └── linear_regression_model.py
│
└── data/
    └── dataset_procesado.csv


---

## 🧩 Funcionalidades Principales

1. **Carga del dataset (`dataset_procesado.csv`)**
   - El archivo contiene la información de los estudiantes, incluyendo su `StudentID` y la variable objetivo `GradeClass`.

2. **Selección individual por estudiante**
   - El usuario puede seleccionar un estudiante por su `StudentID` para visualizar su información y predecir su calificación final.

3. **Entrenamiento automático del modelo**
   - El sistema utiliza un modelo de clasificación (por ejemplo, *Random Forest* o *Logistic Regression*) entrenado con los datos cargados.

4. **Visualización de resultados**
   - Se muestran métricas del modelo como:
     - Exactitud (accuracy)
     - Matriz de confusión
     - Importancia de variables (gráficas de barras)
     - Comparación entre valores reales y predichos

5. **Interfaz interactiva**
   - Construida con **Streamlit**, permite explorar los datos, realizar predicciones y visualizar gráficamente el rendimiento del modelo.

---

## 📊 Ejemplo de Gráficas Incluidas
- Distribución de la variable objetivo (`GradeClass`)
- Importancia de las características predictoras
- Matriz de confusión del modelo
- Evolución de precisión en el entrenamiento

---

## 🧪 Tecnologías Utilizadas
- **Python 3.10+**
- **Pandas** – Procesamiento de datos  
- **Scikit-learn** – Modelado predictivo  
- **Matplotlib / Seaborn / Plotly** – Visualización de datos  
- **Streamlit** – Interfaz web interactiva  

---

## ⚙️ Instalación y Configuración del Entorno

A continuación se describen los pasos para preparar y ejecutar correctamente el proyecto.

---

### 🧩 1. Crear un entorno virtual

Se recomienda crear un entorno virtual de Python para aislar las dependencias del proyecto y evitar conflictos con otras instalaciones.

#### En Windows:
```bash
python -m venv venv
venv\Scripts\activate



