import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'bitcoin_data.csv')

def prepare_data():
    #cargar csv donce index_col=0 usa fecha como indice y parse_dates para que entienda que son fechas
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    data = df.values

    #escalado entre 0y1 para evitar numeros grandes
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)

    #crear las ventanas 
    prediction_days = 100
    x_train = []
    y_train = []
    #cada xtrain tiene los dias anteriores  e ytrain el valor a predecir
    for i in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[i-prediction_days:i,0]) #toma desde i-prediccion hasta i dias
        y_train.append(scaled_data[i, 0])  #toma el i dia exacto a predecir
    
    x_train, y_train = np.array(x_train), np.array(y_train) #convertir a numpy
    #reshape [Muestras(samples, los datos- prediction), Pasos de Tiempo(timestep dias), Features(1 Close)]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    print(f"entrada x_train {x_train.shape}")
    print(f" {x_train.shape[0]} ejemplos cada uno de {x_train.shape[1]} dias")
    print(f"salida y_train {y_train.shape}")

    uj_x = x_train[-1] #ultimos ejemplos 100
    uv_y = y_train[-1] #ultimo valor precio real

    plt.figure(figsize=(10,5))
    plt.plot(uj_x, label="valor en 100 dias")
    plt.plot(100, uv_y, 'ro', label="a predecir el dia 101")
    plt.title("ventana de entrenamiento")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    prepare_data()