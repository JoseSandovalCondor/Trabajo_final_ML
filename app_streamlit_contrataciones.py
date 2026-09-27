# Debe direccionar VS Code a la carpeta con los archivos:
# 1.- Archivo
# 2.- Abrir carpeta. Debe dar click en la carpeta que contiene los archivos de interés
# 3.- A la izquierda, en el explorador deberá poder visualizar todos los archivos
#------------------------------------------------------------------------------------------------
# CÓDIGO STREAMLIT
# Ir a:   Ver/Terminal
# Crea un ambiente virtual (puedes usar otro nombre en lugar de 'venv'): coloca este código
# python -m venv venv
#---------------------------------------------------------------------------------------
# Luego de crear el ambiente virtual, lo activas
# .\venv\Scripts\activate   # En Windows
#---------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
# Cuando vuelva a iniciar sesión, debe volver a activar el ambiente virtual, ya no lo debe crear.
# En este caso debes abrir la carpeta con los archivos del caso.
#---------------------------------------------------------------------------------------------
# Instala las dependencias necesarias
# pip install streamlit pandas joblib numpy scikit-learn
#-------------------------------------------------------------------------------------------------
# Desde la segunda vez: hacer:
# Si da error, debes ir a PowerShell de Window y:
# Get-ExecutionPolicy                           Si es Restricted; ejecuta
# Set-ExecutionPolicy RemoteSigned              Colocar Sí
# En consola de VSC:  .\venv\Scripts\activate

import streamlit as st
import pandas as pd
from joblib import load
import numpy as np

#-------------------------PROCESO DE DESPLIEGUE------------------------------

# 01 - Load the model (ruta relativa: funciona en local y en la nube)
clf = load('modelo_contrataciones.joblib')

#-------------------------CARGA DE DATOS PARA FILTRADO DINÁMICO------------------------------

@st.cache_data
def cargar_datos():
    """Carga el CSV y lo cachea para no leerlo en cada interacción.
    Usa ruta relativa para funcionar tanto en local como en Streamlit Cloud."""
    """return pd.read_csv('contrataciones.csv', sep=';')"""
    return pd.read_csv('data/contrataciones.csv', sep=';')

# Cargar datos (solo se lee una vez gracias al caché)
df_datos = cargar_datos()

# Obtener listas únicas ordenadas directamente de la data de entrenamiento
departamento_options = sorted(df_datos['DEPARTAMENTO'].dropna().unique().tolist())
metodo_options = sorted(df_datos['METODO DE CONTRATACION'].dropna().unique().tolist())
objeto_options = sorted(df_datos['OBJETO'].dropna().unique().tolist())

# Valores por defecto
departamento_default = departamento_options[0] if departamento_options else None
metodo_default = metodo_options[0] if metodo_options else None
objeto_default = objeto_options[0] if objeto_options else None

# 03 - Reseteo - Flag to track error
error_flag = False

# Reset inputs function
def reset_inputs():
    global error_flag
    error_flag = False

# Inicializar variables
reset_inputs()

#-----------------------------------------------------------------------------------------------
#------------------------Título centrado-------------------------------------------------
st.title("🏛️ Modelo Predictivo de Nivel de Competencia en Contratación Pública")
st.markdown("Este modelo predice el **Nivel de Competencia** (concurrencia de postores) en procesos de contratación pública en Perú, en base a características del proceso.")
st.markdown("Elaborado por: **José Sandoval Santamaría**")
st.markdown("---")

#----------------------- Función para validar los campos del formulario----------------------------
def validate_inputs():
    global error_flag
    if cuantia < 0:
        st.error("La cuantía no puede ser negativa. Por favor, ingrese un valor válido.")
        error_flag = True
    else:
        error_flag = False

#------------------------------------ Formulario en dos columnas------------------------------------
with st.form("contrataciones_form"):
    col1, col2 = st.columns(2)
    
    # Input fields en la primera columna
    with col1:
        # SELECTBOX DE DEPARTAMENTO (lista fija de 25)
        departamento = st.selectbox(
            "**📍 Departamento**", 
            options=departamento_options,
            index=departamento_options.index(departamento_default) if departamento_default in departamento_options else 0
        )
        
        # SELECTBOX DE PROVINCIA (dinámico, filtrado por departamento)
        # Filtramos las provincias que pertenecen al departamento seleccionado
        provincias_del_departamento = sorted(
            df_datos[df_datos['DEPARTAMENTO'] == departamento]['PROVINCIA'].dropna().unique().tolist()
        )
        
        provincia_default = provincias_del_departamento[0] if provincias_del_departamento else None
        
        provincia = st.selectbox(
            "**🏙️ Provincia**", 
            options=provincias_del_departamento,
            index=provincias_del_departamento.index(provincia_default) if provincia_default in provincias_del_departamento else 0
        )
        
        metodo = st.selectbox(
            "**📋 Método de Contratación**", 
            options=metodo_options,
            index=metodo_options.index(metodo_default) if metodo_default in metodo_options else 0
        )
    
    # Input fields en la segunda columna
    with col2:
        objeto = st.selectbox(
            "** Objeto de Contratación**", 
            options=objeto_options,
            index=objeto_options.index(objeto_default) if objeto_default in objeto_options else 0
        )
        cuantia = st.number_input(
            "**💰 Cuantía (PEN)**", 
            min_value=0.0, 
            value=100000.0, 
            step=1000.0,
            help="Presupuesto, valor estimado o valor referencial del proceso"
        )
    
    # ----------------------------------------- Boton de Predecir-------------------------------------------------
    predict_button = st.form_submit_button("🔮 Predecir Nivel de Competencia")
    
    # Validar que no haya valores negativos en los campos cuando se presiona el botón
    if predict_button and error_flag:
        st.stop()
    
    if predict_button and not error_flag:
        # Crear DataFrame con los inputs
        data = {
            'DEPARTAMENTO': [departamento],
            'PROVINCIA': [provincia],
            'METODO DE CONTRATACION': [metodo],
            'OBJETO': [objeto],
            'CUANTIA': [cuantia]
        }
        df = pd.DataFrame(data)
        
        # Realizar predicción
        probabilities_classes = clf.predict_proba(df)[0]
        
        # Obtener la clase con la mayor probabilidad
        class_predicted = np.argmax(probabilities_classes)
        
        # Obtener las clases del modelo (dinámicamente)
        clases = clf.best_estimator_.classes_
        outcome = clases[class_predicted]
        probability_outcome = probabilities_classes[class_predicted]
        
        # Asignar estilo según la clase predicha
        if outcome == 'Alta competencia':
            style_result = 'background-color: lightgreen; font-size: larger; padding: 15px; border-radius: 5px;'
        elif outcome == 'Mediana competencia':
            style_result = 'background-color: lightyellow; font-size: larger; padding: 15px; border-radius: 5px;'
        else:  # Baja competencia
            style_result = 'background-color: lightcoral; font-size: larger; padding: 15px; border-radius: 5px;'
        
        # Mostrar resultado con estilo personalizado
        result_html = f"<div style='{style_result}'>🎯 La predicción fue de clase: <strong>'{outcome}'</strong> con una probabilidad de <strong>{round(float(probability_outcome), 4)}</strong></div>"
        st.markdown(result_html, unsafe_allow_html=True)
        
        # Mostrar todas las probabilidades
        st.markdown("---")
        st.markdown("### 📊 Probabilidades por clase:")
        for i, clase in enumerate(clases):
            st.write(f"- **{clase}**: {round(float(probabilities_classes[i]), 4)}")
        
        # Mostrar resumen del proceso
        st.markdown("---")
        st.markdown("### 📝 Resumen del Proceso:")
        st.write(f"- **Departamento**: {departamento}")
        st.write(f"- **Provincia**: {provincia}")
        st.write(f"- **Método de Contratación**: {metodo}")
        st.write(f"- **Objeto**: {objeto}")
        st.write(f"- **Cuantía**: S/ {cuantia:,.2f}")

#--------------------------- Boton de Resetear-------------------------------------
if st.button("🔄 Resetear"):
    # Resetear inputs
    reset_inputs()
    st.rerun()

# Instrucciones para ejecutar
# streamlit run app_streamlit_contrataciones.py
# pip freeze > requirements.txt
