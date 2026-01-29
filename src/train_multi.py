import pandas as pd
import numpy as np
import random
import tensorflow as tf
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)          
os.environ['TF_DETERMINISTIC_OPS'] = '1'          
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'clp_multivariable.csv')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

def train_model_multi():

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    
    # definir features, elimine el yuan para que el modelo funcione mejor
    features = ['CLP', 'Dolar_Index', 'Cobre']
    data = df[features].values # array de n_filas y columnas

    # scalers separados x para features e y para el target clp
    scaler_x = MinMaxScaler(feature_range=(0,1))
    scaled_data_x = scaler_x.fit_transform(data) # escala las 3 columnas
    scaler_y = MinMaxScaler(feature_range=(0,1))
    scaler_y.fit(df[['CLP']].values) # escala el target y guardar su scaler 

    # guardar scalers
    joblib.dump(scaler_x, os.path.join(MODEL_DIR, 'scaler_multi_X.gz'))
    joblib.dump(scaler_y, os.path.join(MODEL_DIR, 'scaler_multi_Y.gz'))
    
    PD = 70  # subido a 70 dias para mas contexto, funciona mejor que en el univariado donde funcionaba mejor uno pequeño
    PDC = 7  # tambien subi los dias a predecir, ya que el modelo al tener mas features en teoria puede manejar mejor la prediccion a mas dias
        
    x_train = []
    y_train = []
    
    #ventanas
    for i in range(PD, len(scaled_data_x) - PDC + 1):
        x_train.append(scaled_data_x[i-PD:i, :])       # input todas las columnas menos el clp
        y_train.append(scaled_data_x[i:i+PDC, 0])      # target solo columna 0 CLP

    x_train, y_train = np.array(x_train), np.array(y_train) # reshape  a x(samples, pasos_tiempo, features=3), y (samples, PDC)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], len(features)))

    #separacion de datos train/val/test a diferencia de los dos modelos univariados que use validation_split
    total_samples = len(x_train)
    train_end = int(total_samples * 0.8) #80% train
    val_end = int(total_samples * 0.9)   #10% validar y el 10% resante test 

    X_train = x_train[:train_end] #80% 
    y_train_split = y_train[:train_end]

    X_val = x_train[train_end:val_end] #10%
    y_val = y_train[train_end:val_end]

    X_test = x_train[val_end:] #10%
    y_test = y_train[val_end:]

    print(f"\nmuestras totales: {total_samples:,}")
    print(f"train Set: {len(X_train):,} muestras ({len(X_train)/total_samples*100:.1f}%)")
    print(f"validation Set: {len(X_val):,} muestras ({len(X_val)/total_samples*100:.1f}%)")
    print(f"test Set : {len(X_test):,} muestras ({len(X_test)/total_samples*100:.1f}%)\n")

    #guardar test set par la evaluacion 
    np.save(os.path.join(MODEL_DIR, 'X_test_multi.npy'), X_test)
    np.save(os.path.join(MODEL_DIR, 'y_test_multi.npy'), y_test)

    # LSTM basado en el univariado pero adaptado a multivariable cambiando parametros, en general mas pequeño que el univariado para evitar overfitting
    model = Sequential()
    # capa 1, agregar regularización L2 no mejoro   
    model.add(LSTM(units=45, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2]))) #reduci las neuronas por que tiende a sobreajustarse si uso las des univariado
    model.add(Dropout(0.20)) #menos dropout por lo mismo(20), con 25 tambien va bien
                      
    # capa 2 eliminada por ahora, despues de estar probando configuraciones me dio mejor resultado sin ella reduciendo el overfitting
    #model.add(LSTM(units=20, return_sequences=False)) 
    #model.add(Dropout(0.3))

    # capa 3 densa y salida 
    model.add(Dense(units=20, activation='relu')) #neurona dense y activacion relu,lo subo a 20 ya que el modelo trabaja con mas features que en el univariado
    model.add(Dropout(0.15)) #ligero dropout en la capa densa, ayudo bastante 
    model.add(Dense(units=PDC)) 
    
    # compilar, dejo casi todo igual que en el univariado
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    best_model_path = os.path.join(MODEL_DIR, 'clp_multi_lstm.h5')
    
    checkpoint = ModelCheckpoint(
        filepath=best_model_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20, # mayor espera para que converja mejor
        restore_best_weights=True
    )
    
    #reducir learning rate no me funciono
    #reduce_lr = ReduceLROnPlateau(
    #    monitor='val_loss',
    #    factor=0.5,
    #    patience=10,
    #    min_lr=0.00001,
    #    verbose=1
    #)

    # entrenamiento con uso de validation data en vez del split, mejor para monitorear overfitting, sobreajuste
    history = model.fit(X_train, y_train_split, epochs=250, batch_size=64, validation_data=(X_val, y_val), callbacks=[checkpoint, early_stopping],verbose=1)
    #mas epocas y batch mayor para estabilidad


    # info final
    print(f"\nmejor modelo guardado en {best_model_path}")
    print(f"epocas totales: {len(history.epoch)}\n")

    #cargar mejor modelo
    best_model = load_model(best_model_path)
    #evaluar en el test set y prediccion para ver como se comporta con datos nuevos
    test_loss, test_mae_scaled = best_model.evaluate(X_test, y_test, verbose=0) 
    y_pred_scaled = best_model.predict(X_test, verbose=0)

    #desnormalizar usando el escaler de y 
    y_pred = scaler_y.inverse_transform(y_pred_scaled) #desnormalizar predidccion 
    y_true = scaler_y.inverse_transform(y_test) #desnormalizar valores reales

    #calcular metricas de error en clp
    mae_clp = mean_absolute_error(y_true, y_pred)
    rmse_clp = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # baseline que es una prediccion simple que predice el siguiente valor con el actual,copia, asumir hoy=mañana
    last_known_values = X_test[:, -1, 0:1]
    last_known_clp = scaler_y.inverse_transform(last_known_values)
    baseline_pred = np.repeat(last_known_clp, PDC, axis=1)
    baseline_mae = mean_absolute_error(y_true, baseline_pred)
    improvement = ((baseline_mae - mae_clp) / baseline_mae * 100)

    print(f"\nmetricas en Pesos Chilenos pen el Test Set\n")
    print(f"MAE: {mae_clp:.2f} CLP")  #error absoluto medio
    print(f"RMSE: {rmse_clp:.2f} CLP") #raiz error cuadratico
    print(f"MAPE(porcentaje de error): {mape:.2f}%\n")  #error 

    print(f"comparacion con baseline\n")
    print(f"mejora sobre baseline: {improvement:.2f}%") 
    print(f"mejora del LSTM multivariable: {mae_clp:.2f}%")
    print(f"diferencia ${baseline_mae - mae_clp:.2f} CLP")

    print(f"\nejemplos de prediccion en test Set\n")
    for i in range(min(3, len(y_true))):
        print(f"muestra {i+1}:")
        print(f"Real: ${y_true[i][0]:.2f} | Pred: ${y_pred[i][0]:.2f} | Error: ${abs(y_true[i][0]-y_pred[i][0]):.2f}")

    # grafico 
    plt.figure(figsize=(14, 5))
    
    # grafico 1 para Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    plt.axhline(y=test_loss, color='red', linestyle='--', label=f'Test Loss ({test_loss:.4f})', linewidth=2)
    plt.title('Curva de aprendizaje multivariable')
    plt.xlabel('epocas', fontsize=12)
    plt.ylabel('Loss MSE', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)


    plt.title(f'Validacion del modelo en 180 dias(Ventana de {PD} dias)')
    

    # grafico 2 para MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE', color='blue', linewidth=2)
    plt.plot(history.history['val_mae'], label='Validation MAE', color='orange', linewidth=2)
    plt.title('Mean Absolute Error')
    plt.xlabel('epocas', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, 'training_history_multi.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nGrafico guardado en: {plot_path}\n")
    plt.show()

if __name__ == "__main__":
    train_model_multi()