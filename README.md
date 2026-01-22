Este pequeño proyecto utiliza una red neuronal recurente (LSTM) de forma simple y univariable para predecir el precio de cierre del bitcoin y peso chileno  
El modelo se entrena con datos historicos y utiliza una ventana de observación de 80 dias para proyectar el valor del dia siguiente
Incluye un dashboard  en Streamlit para visualizar datos histricos 

El proyecto esta diseñado para ejecutarse en conda,`bitcoin_env`

conda env create -f environment.yml
conda activate bitcoin_env

streamlit run src/05_dashboard_v2.py