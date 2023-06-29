import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import pandas_datareader as data 
import streamlit as st 
import yfinance as yf

st.title('Stock Trend Prediction')


@st.cache(allow_output_mutation=True,show_spinner=True)
def head(user_input):
  
  #ticker = request.args['ticker']
  dff = yf.Ticker(user_input).history(period='max', # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
                                  interval='1d', # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
                                  actions=False)
  return dff                                

user_input_h = st.text_input('Enter Stock Ticker','SBIN.NS')
df = head(user_input_h)




#Describing Data 
st.subheader('Data Description')
st.write(df.describe())


#Vishualisation
st.subheader('Closing Prise vs Time Chart')
fig = plt.figure(figsize= (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize= (12,6))
plt.plot(ma100, 'r')
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 200MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize= (12,6))
plt.plot(ma100 , 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close)
st.pyplot(fig)




#splitting for test and train

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

#feature scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

#getting data ready
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])
x_test, y_test = np.array(x_test), np.array(y_test)

#upscale
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_test = y_test * scale_factor




#all model representation
#function for plotting the graph
def plot_the_graph(a,b):
  fig2 = plt.figure(figsize=(12,6))
  plt.plot(a, label = 'Original Price')
  plt.plot(b, 'r', label = 'Predicted Price')
  plt.xlabel('Time')
  plt.ylabel('Price')
  plt.legend()
  st.pyplot(fig2)


st.subheader('Model_10 (Epochs - 10 , activation function - relc) :- Prediction vs Original')
model10 = load_model('Models/Keras_model_lstm_10.h5')
y_predicted_10 = model10.predict(x_test)
y_predicted_10 = y_predicted_10 * scale_factor 
plot_the_graph(y_test,y_predicted_10)
  
st.subheader('Model_20 (Epochs - 20 , activation function - sigmoid) :- Prediction vs Original')
model20 = load_model('Models/Keras_model_lstm_20.h5')
y_predicted_20 = model20.predict(x_test)
y_predicted_20 = y_predicted_20 * scale_factor 
plot_the_graph(y_test,y_predicted_20)

st.subheader('Model_30 (Epochs - 30 , activation function - relc):- Prediction vs Original')
model30 = load_model('Models/Keras_model_lstm_30.h5')
y_predicted_30 = model30.predict(x_test)
y_predicted_30 = y_predicted_30 * scale_factor 
plot_the_graph(y_test,y_predicted_30)

st.subheader('Model_40 (Epochs - 40 , activation function - relc):- Prediction vs Original')
model40 = load_model('Models/Keras_model_lstm_40.h5')
y_predicted_40 = model40.predict(x_test)
y_predicted_40 = y_predicted_40 * scale_factor 
plot_the_graph(y_test,y_predicted_40)

st.subheader('Model_50 (Epochs - 50 , activation function - relc):- Prediction vs Original')
model50 = load_model('Models/Keras_model_lstm_50.h5')
y_predicted_50 = model50.predict(x_test)
y_predicted_50 = y_predicted_50 * scale_factor 
plot_the_graph(y_test,y_predicted_50)






















