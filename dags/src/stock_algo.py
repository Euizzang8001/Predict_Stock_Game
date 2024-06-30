import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pykrx import stock
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pendulum

def get_today():
    kst = pendulum.timezone('Asia/Seoul')
    current_time = datetime.now().astimezone(kst)
    dt_now = str(current_time.date())
    print(f'{dt_now} 기준')
    dt_now = ''.join(c for c in dt_now if c not in '-')
    # dt_now = "2024-06-28"
    return dt_now

def get_nexon():
    dt_now = get_today()
    nexon = stock.get_market_ohlcv("20150925", dt_now, "225570")
    nexon.to_csv(f'./{dt_now}_nexon_stock.csv', index=True)

def nexon_xy():
    dt_now = get_today()
    nexon = pd.read_csv(f'./{dt_now}_nexon_stock.csv', index_col=0)
    scaler = MinMaxScaler()
    scaler2 = MinMaxScaler()
    scale_cols_for_x = ['시가', '고가', '저가', '거래량']
    scale_cols_for_y = ['종가']
    test = scaler.fit_transform(nexon[scale_cols_for_x])
    test2 = scaler2.fit_transform(nexon[scale_cols_for_y])
    np.save(f'./{dt_now}_test.npy', test)
    np.save(f'./{dt_now}_test2.npy', test2)
    
       
def lstm_nexon(today_info):
    dt_now = get_today()
    test = np.load(f'./{dt_now}_test.npy')
    test2 = np.load(f'./{dt_now}_test2.npy')
    X = np.array([test[i:i+1] for i in range(test.shape[0]-1 )])
    y = np.array([test2[i+1] for i in range(test2.shape[0]-1)])
    X_model_input = tf.keras.layers.Input(shape=(X.shape[1], X.shape[2]))
    X_model_output = tf.keras.layers.LSTM(32, activation='relu', return_sequences=False)(X_model_input)
    X_model_output = tf.keras.layers.Dense(1, activation='linear')(X_model_output)
    X_model_output = tf.keras.layers.Dense(1)(X_model_output)
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2) 
    result_model = tf.keras.Model(inputs=X_model_input, outputs=X_model_output)
    result_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    result_model.fit(x_train, y_train, batch_size = 1, epochs=100, validation_data=(x_valid, y_valid), callbacks=[early_stop])                
    pred = result_model.predict(np.array([today_info]))
    return pred

def predict_or_check():
    dt_now = get_today()
    nexon = pd.read_csv(f'./{dt_now}_nexon_stock.csv', index_col=0)
    scaler = MinMaxScaler()
    scaler2 = MinMaxScaler()
    scale_cols_for_x = ['시가', '고가', '저가', '거래량']
    scale_cols_for_y = ['종가']
    scaler.fit_transform(nexon[scale_cols_for_x])
    scaler2.fit_transform(nexon[scale_cols_for_y])

    kst = pendulum.timezone('Asia/Seoul')
    current_time = datetime.now().astimezone(kst)

    today_nexon = stock.get_market_ohlcv(dt_now, dt_now, "225570")
    today_info = scaler.transform([[today_nexon['시가'].values[0], today_nexon['고가'].values[0], today_nexon['저가'].values[0], today_nexon['거래량'].values[0]]])
    test = lstm_nexon(today_info=today_info)
    today_pred = scaler2.inverse_transform(test)
    today_pred = "{:.2f}".format(today_pred[0][0])
    if current_time.hour >= 15:
        today_close = today_nexon['종가'].values[0]
        print(f'넥슨!\n{dt_now}의 예측 종가는?:\n{today_pred}원\n오늘의 진짜 종가!:\n{today_close}')
    else:
         print(f'넥슨!\n{dt_now}의 예측 종가는?:\n{today_pred}원')