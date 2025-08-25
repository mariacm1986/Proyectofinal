# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 19:23:39 2025

@author: Maria G
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats

# --------------------------
# Cargar datos con cache
# --------------------------
@st.cache_data
def cargar_datos():
    return pd.read_csv("dataset_limpio.csv")

# --------------------------
# Título y datos
# --------------------------
st.title("📊 Modelos: Regresión Lineal y Clasificación KNN")
ds = cargar_datos()
st.write("Vista previa de los datos")
st.dataframe(ds.head())

# --------------------------
# Codificación de variables categóricas
# --------------------------
ds_encode = ds.copy()
le_genero = LabelEncoder()
le_nivel = LabelEncoder()
ds_encode['Genero'] = le_genero.fit_transform(ds_encode['Genero'])
ds_encode['Nivel_Educativo'] = le_nivel.fit_transform(ds_encode['Nivel_Educativo'])

# --------------------------
# Variables para Regresión
# --------------------------
X = ds_encode.drop(columns=["Satisfaccion_Vida"])
y = ds_encode["Satisfaccion_Vida"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------
# Visualización: Pairplot y Correlación
# --------------------------
ds_numeric = ds_encode.drop(columns=["Genero", "Nivel_Educativo"])
st.subheader("Relaciones entre variables (solo cuantitativas)")
fig1 = sns.pairplot(ds_numeric)
st.pyplot(fig1.fig)

st.subheader("Matriz de correlación")
fig2, ax2 = plt.subplots(figsize=(6,4))
sns.heatmap(ds_numeric.corr(), annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

# --------------------------
# Modelo de Regresión Lineal
# --------------------------
st.header("📈 Regresión Lineal Múltiple")
modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

st.write("🔹 Error Cuadrático Medio (MSE):", round(mean_squared_error(y_test, y_pred), 2))
st.write("🔹 Coeficiente de Determinación (R²):", round(r2_score(y_test, y_pred), 2))

# Gráfico Real vs Predicho
fig3, ax3 = plt.subplots()
ax3.scatter(y_test, y_pred, color='blue', alpha=0.6)
ax3.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax3.set_xlabel("Satisfacción Real")
ax3.set_ylabel("Satisfacción Predicha")
ax3.set_title("Comparación: Real vs Predicción")
st.pyplot(fig3)

# --------------------------
# Formulario de predicción personalizada
# --------------------------
st.sidebar.header("🧾 Predicción personalizada Regresión")
col_numericas = [col for col in X.columns if col not in ['Genero', 'Nivel_Educativo']]
inputs = {}

for col in col_numericas:
    min_val = int(ds[col].min())
    max_val = int(ds[col].max())
    val = int(ds[col].mean())
    inputs[col] = st.sidebar.slider(f"{col}", min_val, max_val, val)

genero_opciones = le_genero.classes_
genero_seleccionado = st.sidebar.selectbox("Género", genero_opciones)
inputs['Genero'] = le_genero.transform([genero_seleccionado])[0]

nivel_opciones = le_nivel.classes_
nivel_seleccionado = st.sidebar.selectbox("Nivel Educativo", nivel_opciones)
inputs['Nivel_Educativo'] = le_nivel.transform([nivel_seleccionado])[0]

nuevo_input = pd.DataFrame([inputs])[X.columns]

if st.sidebar.button("🔮 Predecir Satisfacción de Vida"):
    prediccion = modelo.predict(nuevo_input)
    st.sidebar.success(f"Satisfacción estimada: {prediccion[0]:.2f}")

# --------------------------
# Modelo KNN
# --------------------------
st.header("🤖 Clasificación con KNN")

# Para KNN usamos solo variables numéricas (ejemplo: Edad, Ingreso)
X_knn = ds_numeric[['Edad', 'Ingreso_Mensual']]
y_knn = ds['Genero']  # ejemplo: predecir género (puedes cambiar a otra variable categórica)

X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_knn, y_knn, test_size=0.3, random_state=42)

scaler_knn = StandardScaler()
X_train_knn_scaled = scaler_knn.fit_transform(X_train_knn)
X_test_knn_scaled = scaler_knn.transform(X_test_knn)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_knn_scaled, y_train_knn)

y_pred_knn = knn.predict(X_test_knn_scaled)

# Métricas
cm = confusion_matrix(y_test_knn, y_pred_knn)
accuracy = accuracy_score(y_test_knn, y_pred_knn)

st.subheader("Matriz de Confusión")
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
ax_cm.set_xlabel("Predicho")
ax_cm.set_ylabel("Real")
st.pyplot(fig_cm)

st.write("🔹 Precisión del KNN:", round(accuracy, 2))

# Formulario KNN
st.sidebar.header("🧾 Predicción personalizada KNN")
edad_input = st.sidebar.slider("Edad", int(ds_numeric['Edad'].min()), int(ds_numeric['Edad'].max()), int(ds_numeric['Edad'].mean()))
ingreso_input = st.sidebar.slider("Ingreso Mensual", int(ds_numeric['Ingreso_Mensual'].min()), int(ds_numeric['Ingreso_Mensual'].max()), int(ds_numeric['Ingreso_Mensual'].mean()))

if st.sidebar.button("🔮 Predecir KNN"):
    nuevo_knn = scaler_knn.transform([[edad_input, ingreso_input]])
    pred_knn = knn.predict(nuevo_knn)
    st.sidebar.success(f"Predicción KNN: {pred_knn[0]}")

st.markdown("---")    
st.header("📊 Medidas descriptivas para variables cuantitativas")

medidas = pd.DataFrame({
    "Media": ds_numeric.mean(),
    "Mediana": ds_numeric.median(),
    "Moda": ds_numeric.mode().iloc[0],  # la primera moda
    "Asimetría": ds_numeric.skew(),
    "Curtosis": ds_numeric.kurt()
})

st.dataframe(medidas)
    
