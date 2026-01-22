import numpy as np
import pandas as pd
import os
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'clp_data.csv')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

def train_model():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    data = df.values

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    scaler_path = os.path.join(MODEL_DIR, 'scaler_clp.gz')
    joblib.dump(scaler, scaler_path)
    
    #ventanas
    PD = 90 #dias
    x_train = []
    y_train = []

    for i in range(PD, len(scaled_data)):
        x_train.append(scaled_data[i-PD:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    #reshape [Muestras(samples), Pasos de Tiempo(timestep dias), Features(1 Close)]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    #LSTM 3 capas
    model = Sequential()
    #capa 1
    model.add(LSTM(units=70, return_sequences=True, input_shape=(x_train.shape[1], 1))) # neuronas en este caso al ser clp, sigueinte capa recurente= true
    
    model.add(Dropout(0.35)) #dropout, aqui sirvio un dropout mayor que el del bitcoin

    #capa 2
    model.add(LSTM(units=60, return_sequences=False)) #ultima capa recurente=false
    model.add(Dropout(0.35))

    #capa 3 densa
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=1)) #1 precio

    #compilar/optimizador y loss/checkpoint mejor modelo y metrica mae
    #opt = Adam(learning_rate=0.0008)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])  #model.compile(optimizer='adam', loss='mean_squared_error')
    best_model_path = os.path.join(MODEL_DIR, 'clp_lstm.h5')
    checkpoint = ModelCheckpoint(
        filepath=best_model_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    print("entrenando modelo")
    
    #early stoping
    early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=11,        # espera n epocas
    restore_best_weights=True
    )
    
    #guardar entrenamiento en history/0 epocas en batch de n , activar checkpoint
    history = model.fit(x_train, y_train, epochs=100, batch_size=30, validation_split=0.1, callbacks=[checkpoint, early_stopping] )
    #model.fit(x_train, y_train, epochs=200, batch_size=31,validation_split=0.1,callbacks=[checkpoint], shuffle=False)
    #bajar el batch a 31 dio mejor resultado 
    #guardar ultimo modelo
    final_model_path = os.path.join(MODEL_DIR, 'clp_lstm_final.h5')
    model.save(final_model_path)
    print(f"el modelo final se guardo en {{final_model_path}}")
    print(f"el mejor modelo se guardo en {{best_model_path}}")
    print(f"el entrenamiento se detuvo en la epoca {len(history.epoch)}")

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss (Entrenamiento)', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss (Validación)', color='orange')
    plt.title('Curva de Aprendizaje: Loss vs Val_Loss')
    plt.xlabel('Epocas')
    plt.ylabel('Error (Loss)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_model()