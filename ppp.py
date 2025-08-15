# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 16:39:29 2025

@author: Maria G
"""

## Importar librerías necesarias
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# --------------------------
# Cargar datos con cache
# --------------------------
@st.cache_data
def cargar_datos():
    return pd.read_csv("dataset_limpio.csv")

# --------------------------
# 1. Título y datos
# --------------------------
st.title("Predicción de Satisfacción de Vida")
ds = cargar_datos()
st.write("Vista previa de los datos")
st.dataframe(ds.head())

# --------------------------
# 2. Codificación de variables categóricas
# --------------------------
ds_encode = ds.copy()

# Codificadores separados por columna
le_genero = LabelEncoder()
le_nivel = LabelEncoder()

ds_encode['Genero'] = le_genero.fit_transform(ds_encode['Genero'])
ds_encode['Nivel_Educativo'] = le_nivel.fit_transform(ds_encode['Nivel_Educativo'])

# --------------------------
# 3. Definir X y y
# --------------------------
X = ds_encode.drop(columns=["Satisfaccion_Vida"])
y = ds_encode["Satisfaccion_Vida"]  # Asumimos que es numérica continua

# --------------------------
# 4. División de datos
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------
# 5. Visualización: Pairplot y Matriz de correlación
# --------------------------
st.header("📈 Modelo de Regresión Lineal Múltiple")

ds_numeric = ds_encode.drop(columns=["Genero", "Nivel_Educativo"])

st.subheader("Relaciones entre variables (solo cuantitativas)")
fig1 = sns.pairplot(ds_numeric)
st.pyplot(fig1)

st.subheader("Matriz de correlación (solo cuantitativas)")
fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.heatmap(ds_numeric.corr(), annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

# --------------------------
# 6. Entrenamiento del modelo
# --------------------------
modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# --------------------------
# 7. Evaluación del modelo
# --------------------------
st.subheader("📊 Evaluación del Modelo")
st.write("🔹 Error Cuadrático Medio (MSE):", round(mean_squared_error(y_test, y_pred), 2))
st.write("🔹 Coeficiente de Determinación (R²):", round(r2_score(y_test, y_pred), 2))

# --------------------------
# 8. Gráfico: Real vs. Predicho
# --------------------------
st.subheader("📉 Gráfico de Regresión: Valores Reales vs. Predichos")
fig3, ax3 = plt.subplots()
ax3.scatter(y_test, y_pred, color='blue', alpha=0.6)
ax3.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax3.set_xlabel("Satisfacción Real")
ax3.set_ylabel("Satisfacción Predicha")
ax3.set_title("Comparación: Real vs. Predicción")
st.pyplot(fig3)

# --------------------------
# 9. Formulario de predicción personalizada
# --------------------------
st.sidebar.header("🧾 Predicción personalizada")

# Variables numéricas restantes (excepto categóricas y y)
col_numericas = [col for col in X.columns if col not in ['Genero', 'Nivel_Educativo']]

inputs = {}

# Sliders para variables numéricas
for col in col_numericas:
    min_val = int(ds[col].min())
    max_val = int(ds[col].max())
    val = int(ds[col].mean())
    inputs[col] = st.sidebar.slider(f"{col}", min_val, max_val, val)

# Selectbox para Género
genero_opciones = le_genero.classes_
genero_seleccionado = st.sidebar.selectbox("Género", genero_opciones)
inputs['Genero'] = le_genero.transform([genero_seleccionado])[0]

# Selectbox para Nivel Educativo
nivel_opciones = le_nivel.classes_
nivel_seleccionado = st.sidebar.selectbox("Nivel Educativo", nivel_opciones)
inputs['Nivel_Educativo'] = le_nivel.transform([nivel_seleccionado])[0]

# Crear DataFrame con los inputs en el mismo orden que X
nuevo_input = pd.DataFrame([inputs])[X.columns]

# --------------------------
# 10. Predicción del modelo
# --------------------------
if st.sidebar.button("🔮 Predecir Satisfacción de Vida"):
    prediccion = modelo.predict(nuevo_input)
    st.sidebar.success(f"Satisfacción estimada: {prediccion[0]:.2f}")
# KNN

