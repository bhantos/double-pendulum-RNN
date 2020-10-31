# -*- coding: utf-8 -*-
from generate_data import *
from time import time
import os
import glob

dt = 0.01
t = np.arange(0.0, 100, dt)
sampNum = 300

def init():
        line.set_data([], [])
        line2.set_data([],[])
        time_text.set_text('')
        return line, line2, time_text
    
def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]
    
    thisx_t = [0, x1_t[i], x2_t[i]]
    thisy_t = [0, y1_t[i], y2_t[i]]

    line.set_data(thisx, thisy)
    line2.set_data(thisx_t, thisy_t)
    time_text.set_text(time_template % (i*dt))
    return line, line2, time_text

#%%
#Generate data
time1 = time()

try:
    os.remove(r"C:\Users\balin\Desktop\7. félév\compsim\proj1\data\training\init.txt")
except FileNotFoundError:
    print("File not found")

for i in range(1,sampNum+1):
    solve(i)
    
print("Training data generated with {} samples".format(sampNum))
print("{} s".format(round(time()-time1,2)))
#%%
#libraries for modeling and training
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dropout, LSTM, SimpleRNNCell
from keras.utils import plot_model
from sklearn.preprocessing import StandardScaler
import keras.backend as kb

import matplotlib.pyplot as plt
from matplotlib import animation

#%%
#loading generated data
t1 = time()
data = np.array([np.loadtxt(file,delimiter=",") for file in 
                 sorted(glob.glob(r"C:\Users\balin\Desktop\7. félév\compsim\proj1\data\training\[!i]*.txt"))])

inputs = np.loadtxt(r"C:\Users\balin\Desktop\7. félév\compsim\proj1\data\training\init.txt",delimiter=",")
print("Loading took {} s.".format(round(time()-t1,3)))
#%%
#Process the data for teacher forcing
v_squares_sim = data[:,:,6:]
x = data[:,:9999,:6]
y = data[:,1:10000,:6]

#print(sum(sum(sum(x[:,1:,:]-y[:,:1998,:])))) #=0, thus data is ready for teacher forcing

#%%
#Train-test-validation split 80%-10%-10%
x_train = x[0:int(sampNum*0.8),:,:]
x_valid = x[int(sampNum*0.8):int(sampNum*0.9),:,:]
x_test =  x[int(sampNum*0.9):,:,:]

y_train = y[0:int(sampNum*0.8),:,:]
y_valid = y[int(sampNum*0.8):int(sampNum*0.9),:,:]
y_test =  y[int(sampNum*0.9):,:,:]

#%%
def train_model(EPOCHS = 10, BATCH_SIZE= 10, LOSS = "mean_absolute_error"):
    time1 = time()
    model = Sequential()
    model.add(SimpleRNN(6, input_shape=x.shape[1:],
                   activation="tanh",
                   return_sequences=True))
    
    
    
    
    opt = tf.keras.optimizers.Adam(lr=0.01,decay=1e-2)
    model.compile(loss=LOSS, optimizer=opt ,metrics=['mae'],)
    
    model.fit(x_train,
              y_train,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              validation_data = (x_valid,y_valid))
    print("Training took ",round(time()-time1,3)," s")
    return model

def loss_fixed_len(alpha,beta):
    def loss(y_true, y_pred):
        loss_value = kb.mean(kb.abs(y_pred  - y_true) ) +  10**(-3)*(alpha * kb.abs(kb.sqrt(kb.square(y_pred[0]) + beta * kb.square(y_pred[1])) - 1 )+ kb.abs(kb.sqrt(kb.square(y_pred[3]) + kb.square(y_pred[4])) - 1))
        return loss_value
    
    return loss

#%%
model = train_model(LOSS = "mean_absolute_error",EPOCHS = 10, BATCH_SIZE = 150) #loss_fixed_len(0.0002,0.001)
#%%

model.evaluate(x_test,y_test)

y_predict = model.predict(x_test)
a = 9 #choose a-th generated time series 
#%%
x1 = y_predict[a,:, 0] #x coordinate of a-th time series
y1 = y_predict[a,:, 1]

x2 = y_predict[a,:, 3]
y2 = y_predict[a,:, 4]

x1_t = y_test[a,:, 0] 
y1_t = y_test[a,:, 1]

x2_t = y_test[a,:, 3]
y2_t = y_test[a,:, 4]


#create figure, axes, plots, animation object
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.grid()

line, = ax.plot([], [], 'o-', lw=1)
line2, = ax.plot([], [], marker= "o", color="red", lw = 1)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y1)),
                              interval=15, blit=True, init_func=init)

plt.show()

#%%

#%%