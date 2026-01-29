Este proyecto implementa el modelo de Deep Learning de redes neuronales recurrentes LSTM para predecir el comportamiento de activos financieros, El objetivo fue aprender y comparar enfoques univariados vs. multivariados en la prediccion de series de tiempo volatiles.

## Modelos Implementados
* Modelo Bitcoin y Peso chileno Univariado:Prediccion basada exclusivamente en el precio historico de cierre,funciona como linea base
* Modelo Peso Chileno Multivariado:Modelo que incorpora otros indicadores para mejorar la precision
    * Inputs: Precio CLP, indice Dolar (DXY), Precio del Cobre y Yuan Chino(se omitio su uso)
    * Ventaja: Corrigio el retraso del modelo simple y se obtuvo una predidccion mas precisa al incorporar variables correlacionadas

## Herramientas
* Data: Descarga automatica desde Yahoo Finance 
* Analisis: Notebooks con un Analisis Exploratorio de Datos simple y validación de modelos.
* Dashboard: Interfaz web interactiva en Streamlit para visualizar predicciones y datos historicos,actualmente solo funciona con los modelos univariados

## Crear entorno
* conda env create -f environment.yml
* conda activate bitcoin_env

## Abrir streamlit
* streamlit run src/05_dashboard_v2.py

## Estructura del Proyecto

```text
BITCOIN_PREDICTOR_LSTM/
├── data/                   # Datasets CSV de Bitcoin, CLP y CLP multivariables
├── models/                 # Carpeta modelos
│   ├── *.h5                # Modelos entrenados (clp_multi_lstm.h5)
│   └── *.gz                # Scalers guardados para normalizar datos nuevos
├── notebooks/              # Jupyter Notebooks para pruebas e inferencia rapida
├── src/                    # Codigo fuente principal
│   ├── 01_get_data.py      # Descarga datos univariados (BTC/CLP)
│   ├── get_data_multi.py   # Descarga datos multivariados (CLP,Cobre, DXY, Yuan).
│   ├── train_multi.py      # Script de entrenamiento del modelo multivariable
│   ├── 05_dashboard.py     # Aplicacion web Streamlit
│   └── ...                 # Otros scripts de preprocesamiento,train y prediccion
└── environment.yml         # Archivo de dependencias para Conda