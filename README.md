Este pequeño proyecto utiliza una red neuronal recurente (LSTM) simple y univariado para predecir el precio de cierre del bitcoin y peso chileno  
El modelo se entrena con datos de yahoo finance y utiliza ventanas de observacion de diferentes dias para las dos moneas con tal de proyectar el valor del dia siguiente
Incluye un dashboard  en Streamlit para visualizar datos histricos 

El proyecto esta diseñado para ejecutarse en conda,`bitcoin_env`

conda env create -f environment.yml
conda activate bitcoin_env

streamlit run src/05_dashboard_v2.py