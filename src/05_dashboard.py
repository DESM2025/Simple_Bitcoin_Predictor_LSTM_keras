import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from datetime import timedelta

st.set_page_config(
    page_title="Prediccion simple",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')

ASSETS_CONFIG = {
    "Bitcoin (USD)": {
        "data_file": "bitcoin_data.csv",
        "model_file": "bitcoin_lstm.h5",
        "scaler_file": "scaler.gz",
        "symbol": "$",
        "color_main": "#17becf", 
        "color_pred": "#e377c2", 
        "pd_days": 80 
    },
    "Peso Chileno (CLP)": {
        "data_file": "clp_data.csv",
        "model_file": "clp_lstm.h5",      
        "scaler_file": "scaler_clp.gz",   
        "symbol": "CLP",
        "color_main": "#ff7f0e", 
        "color_pred": "#2ca02c", 
        "pd_days": 14 
    }
}

# cargar
@st.cache_resource
def load_assets(model_path, scaler_path):
    """Carga modelo y scaler segun las rutas"""
    if not os.path.exists(model_path):
        return None, None
    
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def load_data(data_path):
    """Carga el CSV especifico"""
    if not os.path.exists(data_path):
        return None
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    return df

def main():
    with st.sidebar:
        st.title("configuracion")
        #seleccionar moneda
        selected_asset = st.selectbox(
            "Selecciona el Activo:",
            list(ASSETS_CONFIG.keys()) # ["Bitcoin (USD)", "Peso Chileno (CLP)"]
        )
        
        # Recuperar la configuracion de la moneda elegida
        config = ASSETS_CONFIG[selected_asset]
        
        st.write("---")
        days_to_show = st.slider("historial a mostrar en dias", 30, 365, 90)
        
    path_data = os.path.join(DATA_DIR, config["data_file"])
    path_model = os.path.join(MODEL_DIR, config["model_file"])
    path_scaler = os.path.join(MODEL_DIR, config["scaler_file"])

    # cargar recursos
    model, scaler = load_assets(path_model, path_scaler)
    df = load_data(path_data)

    # Validaciones de seguridad 
    if model is None or scaler is None:
        st.error(f"No se encontro el modelo o el scaler para {selected_asset}.")
        st.warning(f"Buscando en: {path_model}")
        return
    if df is None:
        st.error(f"No se encontro el archivo de datos: {config['data_file']}")
        return

    #Prediccion 
    PD = config["pd_days"]
    
    # Tomar ultimos dias
    try:
        last_days = df['Close'].values[-PD:]
        last_scaled = scaler.transform(last_days.reshape(-1, 1))
        X_input = last_scaled.reshape(1, PD, 1)
        
        pred_scalar = model.predict(X_input, verbose=0)

        pred_scalar_reshaped = pred_scalar.reshape(-1, 1)
        pred_prices_list = scaler.inverse_transform(pred_scalar_reshaped)
        pred_price_d1 = pred_prices_list[0][0]
    except Exception as e:
        st.error(f"Error en la prediccion")
        st.error(str(e))
        return

    # Calculos para KPI
    last_real_price = df['Close'].iloc[-1]
    last_date = df.index[-1]
    delta = pred_price_d1 - last_real_price
    delta_percent = (delta / last_real_price) * 100

    # visualizacion
    st.title(f"Prediccion para {selected_asset}")
    st.markdown(f"Proyeccion para el cierre del **{last_date + timedelta(days=1):%Y-%m-%d}**")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Precio Actual", 
            value=f"{config['symbol']} {last_real_price:,.2f}"
        )
    
    with col2:
        st.metric(
            label="Prediccion para mañana", 
            value=f"{config['symbol']} {pred_price_d1:,.2f}",
            delta=f"{delta:,.2f} ({delta_percent:.2f}%)"
        )
    
    with col3:
        trend = "subir" if delta > 0 else "bajar"
        st.metric(label="Tendencia Esperada", value=trend)

    st.markdown("---")

    tab1, tab2 = st.tabs(["grafico de analisis", "tabla de datos"])

    with tab1:
        plot_df = df.iloc[-days_to_show:]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=plot_df.index, 
            y=plot_df['Close'],
            mode='lines',
            name='valor historico real',
            line=dict(color=config["color_main"], width=2)
        ))
        
        # Prediccion
        future_dates = [last_date + timedelta(days=i) for i in range(1, 3)]
        future_prices = [p[0] for p in pred_prices_list]

        fig.add_trace(go.Scatter(
            x=[last_date] + future_dates,
            y=[last_real_price] + future_prices,
            mode='lines+markers',
            name='Prediccion del modelo',
            line=dict(color=config["color_pred"], width=2, dash='dot'),
            marker=dict(size=8)
        ))

        fig.update_layout(
            title=f"Evolucion del valor {selected_asset}",
            xaxis_title="Fecha",
            yaxis_title=f"Precio ({config['symbol']})",
            template="plotly_dark",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.dataframe(df.tail(10).sort_index(ascending=False), use_container_width=True)

if __name__ == "__main__":
    main()