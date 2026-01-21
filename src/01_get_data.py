import yfinance as yf
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
#comprobar o crear carpeta 
os.makedirs(DATA_DIR, exist_ok=True)

def download_data():
    print("descargando el historal de precio del bitcon desde yahoo")
    #periodo el maximo posible y el valor en intervalos de 1 dia para evitar ruido,todo esto con yahoo
    df = yf.download('BTC-USD', period='max', interval='1d')

    if len(df) > 0:
       print(f"se descargaron en total {len(df)}")
    else:
        print("no se dascargaron datos")
        return
    
    #limpieza, dejar solo la columna con el precio de cierre close
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close']
    #comprobar que el df este limpio con una columna close
    df = pd.DataFrame(df)
    df.columns = ['Close']
    #guardar en un csv
    file_path = os.path.join(DATA_DIR, 'bitcoin_data.csv')
    df.to_csv(file_path)
    print(f"los datos descargars se guardaron en {file_path}")

def download_data2():
    print("descargando el historal de precio del peso chileno desde yahoo")
    #periodo el maximo posible y el valor en intervalos de 1 dia para evitar ruido,todo esto con yahoo
    df = yf.download('USD-CLP', period='max', interval='1d')

    if len(df) > 0:
       print(f"se descargaron en total {len(df)}")
    else:
        print("no se dascargaron datos")
        return
    
    #limpieza, dejar solo la columna con el precio de cierre close
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close']
    #comprobar que el df este limpio con una columna close
    df = pd.DataFrame(df)
    df.columns = ['Close']
    #guardar en un csv
    file_path = os.path.join(DATA_DIR, 'bitcoin_data.csv')
    df.to_csv(file_path)
    print(f"los datos descargars se guardaron en {file_path}")

if __name__ == "__main__":
    download_data()

