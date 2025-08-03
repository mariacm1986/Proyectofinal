# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 22:33:26 2025

@author: Maria G
"""
# 📌 Requisitos:
# pip install streamlit scikit-learn pandas matplotlib seaborn



# Importar librerías necesarias
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Título
st.title("📊 Limpieza, Codificación y Descarga de Datos")

# 1. Cargar base de datos
dt = pd.read_csv("K:/OneDrive/Portatil-HP/Python-Practica/dataset_estadistica.csv")

# 2. Imputar datos numéricos (Ingreso_Mensual y Horas_Estudio_Semanal)
imputer_num = SimpleImputer(strategy="mean")
dt.iloc[:, 4:6] = imputer_num.fit_transform(dt.iloc[:, 4:6])

# 3. Eliminar columna ID_Persona
if 'ID_Persona' in dt.columns:
    dt.drop(columns=['ID_Persona'], inplace=True)

# 4. Imputar datos categóricos (Nivel_Educativo)
imputer_cat = SimpleImputer(strategy="most_frequent")
dt['Nivel_Educativo'] = imputer_cat.fit_transform(dt[['Nivel_Educativo']]).ravel()

# 5. Eliminar los últimos 10 registros
dt = dt.iloc[:-10]

# Mostrar preview de datos limpios (sin codificar)
st.subheader("📋 Vista previa del dataset limpio (sin codificar)")
st.dataframe(dt.head())

# Botón para descargar base limpia
st.download_button(
    label="📥 Descargar base limpia (sin codificar)",
    data=dt.to_csv(index=False),
    file_name="dataset_limpio.csv",
    mime="text/csv"
)

# 6. Separar X (predictoras) e y (variable dependiente)
X = dt.iloc[:, :-1]
y = dt.iloc[:, -1]  # ← esta es la columna 'Satisfacción_Vida'

# 7. Codificar 'Genero' y 'Nivel_Educativo' con LabelEncoder
le_genero = LabelEncoder()
le_nivel = LabelEncoder()
X['Genero'] = le_genero.fit_transform(X['Genero'])
X['Nivel_Educativo'] = le_nivel.fit_transform(X['Nivel_Educativo'])

# 8. Aplicar OneHotEncoder
ct = ColumnTransformer(
    transformers=[
        ('one_hot_encoder', OneHotEncoder(drop='first'), ['Genero', 'Nivel_Educativo'])
    ],
    remainder='passthrough'
)

X_enc = ct.fit_transform(X)

# 9. Construir columnas finales
col_ohe = ct.named_transformers_['one_hot_encoder'].get_feature_names_out(['Genero', 'Nivel_Educativo'])
col_restantes = [col for col in X.columns if col not in ['Genero', 'Nivel_Educativo']]
col_finales = list(col_ohe) + col_restantes

# 10. Crear DataFrame codificado
df_codificado = pd.DataFrame(X_enc, columns=col_finales)

# 11. Agregar la variable dependiente de nuevo
df_codificado['Satisfacción_Vida'] = y.values

# Mostrar preview codificada
st.subheader("📋 Vista previa del dataset codificado (para modelos)")
st.dataframe(df_codificado.head())

# Botón para descargar base codificada
st.download_button(
    label="📥 Descargar base codificada (lista para modelos)",
    data=df_codificado.to_csv(index=False),
    file_name="dataset_codificado.csv",
    mime="text/csv"
)

st.header("📈 Modelo de Regresión Lineal Múltiple")

# Visualización: Pairplot
st.subheader("Relaciones entre variables")
fig1 = sns.pairplot(df_codificado)
st.pyplot(fig1)

# Matriz de correlación
st.subheader("Matriz de correlación")
fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.heatmap(df_codificado.corr(), annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

# 12. Separar X e y del dataframe codificado
X_modelo = df_codificado.drop(columns=["Satisfacción_Vida"])
y_modelo = df_codificado["Satisfacción_Vida"]

# 13. División de datos
X_train, X_test, y_train, y_test = train_test_split(X_modelo, y_modelo, test_size=0.2, random_state=42)

# 14. Entrenamiento del modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# 15. Evaluación del modelo
st.subheader("📊 Evaluación del Modelo")
st.write("🔹 Error Cuadrático Medio (MSE):", round(mean_squared_error(y_test, y_pred), 2))
st.write("🔹 Coeficiente de Determinación (R²):", round(r2_score(y_test, y_pred), 2))



# 16. Gráfico: valores reales vs. predichos
st.subheader("📉 Gráfico de Regresión: Valores Reales vs. Predichos")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, color='blue', alpha=0.6)
ax.plot([0, 15], [0, 15], 'r--')
ax.set_xlabel("Satisfacción Real")
ax.set_ylabel("Satisfacción Predicha")
ax.set_title("Comparación: Real vs. Predicción")
st.pyplot(fig)



# 17. Formulario de predicción
st.sidebar.header("🧾 Predicción personalizada")

# Identificar columnas numéricas restantes (no categóricas)
col_input = [col for col in col_restantes if col not in ['Genero', 'Nivel_Educativo']]

# Crear sliders para columnas numéricas
inputs = {}
for col in col_input:
    min_val = int(dt[col].min())
    max_val = int(dt[col].max())
    val = int(dt[col].mean())
    inputs[col] = st.sidebar.slider(f"{col}", min_val, max_val, val)

# Para 'Genero'
genero_opciones = le_genero.classes_
genero_seleccionado = st.sidebar.selectbox("Género", genero_opciones)
inputs['Genero'] = le_genero.transform([genero_seleccionado])[0]

# Para 'Nivel_Educativo'
nivel_opciones = le_nivel.classes_
nivel_seleccionado = st.sidebar.selectbox("Nivel Educativo", nivel_opciones)
inputs['Nivel_Educativo'] = le_nivel.transform([nivel_seleccionado])[0]

# Crear DataFrame con los datos del formulario
nuevo_input = pd.DataFrame([inputs])

# Aplicar codificación one-hot al nuevo dato
nuevo_codificado = ct.transform(nuevo_input)

# Realizar predicción si se presiona el botón
if st.sidebar.button("🔮 Predecir Satisfacción de Vida"):
    prediccion = modelo.predict(nuevo_codificado)
    st.sidebar.success(f"Satisfacción estimada: {prediccion[0]:.2f}")