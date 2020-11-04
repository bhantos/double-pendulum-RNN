# -*- coding: utf-8 -*-
from generate_data import *
from time import time, strftime
import os
import glob
import math

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
from tensorflow.keras.layers import SimpleRNN, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorboard.plugins.hparams import api as hp
from keras.utils import plot_model
from sklearn.preprocessing import StandardScaler
import keras.backend as kb
from matplotlib.colors import LogNorm

import matplotlib.pyplot as plt
from matplotlib import animation

custom_no = 0

#%%
#loading generated data
t1 = time()
data = np.array([np.loadtxt(file,delimiter=",") for file in 
                 sorted(glob.glob(r"C:\Users\balin\Desktop\7. félév\compsim\proj1\data\training\[!i]*.txt"))])

inputs = np.loadtxt(r"C:\Users\balin\Desktop\7. félév\compsim\proj1\data\training\init.txt",delimiter=",")
print("Loading took {} s.".format(round(time()-t1,3)))
#%%
def loss_fixed_len(alpha,beta,coef):
    def loss(y_true, y_pred):
        loss_value = kb.mean(kb.abs(y_pred  - y_true))  \
            +  coef*(alpha * kb.square(kb.sqrt(kb.square(y_pred[0]) +  kb.square(y_pred[1])) - 1 ) \
            + beta * kb.square(kb.sqrt(kb.square(y_pred[3]) + kb.square(y_pred[4])) - 1))
        return loss_value
    
    return loss

HP_NUM_LAYERS = [12]
HP_LOSS =["mean_absolute_error","mean_squared_error", loss_fixed_len(0.2,1,10**(-1)),
          loss_fixed_len(0.2, 1, 10**(-2)), loss_fixed_len(1, 1, 10**(-2))]
HP_LAYER_TYPE = ["VRNN","LSTM"]
HP_ACT_FUN = ["tanh","relu","sigmoid"]

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

def train_test_model(hparams):
    time1 = time()
    
    model = Sequential()
    if hparams["LAYER_TYPE"] == "LSTM":
        for i in range(hparams["NUM_LAYERS"]):
            model.add(LSTM(6, input_shape=x.shape[1:],
                                   activation=hparams["ACT_FUN"],
                                   return_sequences=True))
    if hparams["LAYER_TYPE"] == "VRNN":
        for i in range(hparams["NUM_LAYERS"]):
            model.add(SimpleRNN(6, input_shape=x.shape[1:],
                                   activation=hparams["ACT_FUN"],
                                   return_sequences=True))
    
    opt = tf.keras.optimizers.Adam(lr=0.1,decay=1e-2)
    model.compile(loss=hparams["LOSS"], optimizer=opt ,metrics=['mae'])
    
    if hparams["NUM_LAYERS"] == 2:
        fit_model = model.fit(x_train,
                  y_train,
                  epochs=EPOCH2,
                  batch_size=150,
                  validation_data = (x_valid,y_valid))
        
    else:
        fit_model = model.fit(x_train,
                  y_train,
                  epochs=EPOCH1,
                  batch_size=150,
                  validation_data = (x_valid,y_valid))
    _, mae = model.evaluate(x_test,y_test)
    print("Training took ",round(time()-time1,3)," s")
    
    if hparams["LOSS"] in ["mean_squared_error","mean_absolute_error"]:
        model.save(r"C:\Users\balin\Desktop\7. félév\compsim\proj1\models\{}_{}_loss{}.model".format(hparams["LAYER_TYPE"],
                                                                                    hparams["NUM_LAYERS"],
                                                                                    hparams["LOSS"]))
    else:
        global custom_no
        custom_no += 1
        model.save(r"C:\Users\balin\Desktop\7. félév\compsim\proj1\models\{}_{}_loss_{}_n{}.model".format(hparams["LAYER_TYPE"],
                                                                                    hparams["NUM_LAYERS"],"custom",custom_no))
        
        
    return mae

def train_model(hparams):
    time1 = time()
    
    model = Sequential()
    if hparams["LAYER_TYPE"] == "LSTM":
        for i in range(hparams["NUM_LAYERS"]):
            model.add(LSTM(6, input_shape=x.shape[1:],
                                   activation=hparams["ACT_FUN"],
                                   return_sequences=True))
    if hparams["LAYER_TYPE"] == "VRNN":
        for i in range(hparams["NUM_LAYERS"]):
            model.add(SimpleRNN(6, input_shape=x.shape[1:],
                                   activation=hparams["ACT_FUN"],
                                   return_sequences=True))
    
    opt = tf.keras.optimizers.Adam(lr=0.1,decay=1e-2)
    model.compile(loss=hparams["LOSS"], optimizer=opt ,metrics=['mae'])
    
    if hparams["NUM_LAYERS"] == 2:
        fit_model = model.fit(x_train,
                  y_train,
                  epochs=EPOCH2,
                  batch_size=150,
                  validation_data = (x_valid,y_valid))
        
    else:
        fit_model = model.fit(x_train,
                  y_train,
                  epochs=EPOCH1,
                  batch_size=150,
                  validation_data = (x_valid,y_valid))
    print("Training took ",round(time()-time1,3)," s")
    
    return model
    
        
#%%
session_num = 0

time1 = time()
maes= {}
EPOCH1 = 1
EPOCH2 = 1
for num_layers in HP_NUM_LAYERS:
    for loss in HP_LOSS:
        for act_fun in HP_ACT_FUN:
            for layer_type in HP_LAYER_TYPE:
                time2 = time()
                hparams = { "NUM_LAYERS": num_layers,
                            "LOSS": loss,
                            "ACT_FUN": act_fun,
                            "LAYER_TYPE": layer_type}
                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                maes[train_test_model(hparams)] = [hparams, time()- time2]
                session_num += 1

print(time()-time1)


#%%
filtered_dict = {}
for k in maes.keys():
    if not (math.isnan(k)):
        filtered_dict[k] = maes[k]
        
#%%

#filtered_maes = {}
EPOCH1 = 30
EPOCH2 = 15

for i,_ in filtered_dict.values():
    pass
    #filtered_maes[train_test_model(i)] = i
    
#%%
models = [filtered_maes[ sorted(filtered_maes.keys())[i] ] for i in range(1,6)] #ignore first(it's nan), 
#get the best five MAE-scored modelparameters

#%%
trained_models = [train_model(models[i]) for i in range(len(models))]
#reinstate previous models with stored hyperparameters

#%%
y_predict = trained_models[4].predict(x_test)
a = 0 #choose a-th generated time series
#%%
x1 = y_predict[a,:, 0] #x coordinate of a-th time series
y1 = y_predict[a,:, 1]

x2 = y_predict[a,:, 3]
y2 = y_predict[a,:, 4]

x1_t = y_test[a,:, 0] 
y1_t = y_test[a,:, 1]

x2_t = y_test[a,:, 3]
y2_t = y_test[a,:, 4]

#%%


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

#plt.hist2d(x1,y_test[a,:,0], norm = LogNorm(),bins=[1000,1000])

#%%

def plot_confusion_hist(model,title):
    y_predict = model.predict(x_test) 
    x1_cc = np.array([])
    x2_cc = np.array([])
    x1_tcc = np.array([])
    x2_tcc = np.array([])

    y1_cc = np.array([])
    y2_cc = np.array([])
    y1_tcc = np.array([])
    y2_tcc = np.array([])
    for a in range(y_predict.shape[0]):
        x1 = y_predict[a,:, 0] #x coordinate of a-th time series
        y1 = y_predict[a,:, 1]
        
        x2 = y_predict[a,:, 3]
        y2 = y_predict[a,:, 4]

        x1_cc = np.concatenate([x1_cc,x1])
        x2_cc = np.concatenate([x2_cc,x2])
        y1_cc = np.concatenate([y1_cc,x1])
        y2_cc = np.concatenate([y2_cc,x1])
        
    plt.figure(figsize=[14,10])
    plt.suptitle("Model number {}".format(title),fontsize=22)
    plt.subplot(221)
    plt.hist2d(x1_cc,y_test[:,:,0].flatten(), norm = LogNorm(),bins=[100,100])
    plt.xlabel("$x_1$ predicted",size=14)
    plt.ylabel("$x_1$ true",size=14)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tick_params("both",length=12,width=2)
    plt.tick_params("both",which="minor",length=8,width=2)
    
    plt.subplot(222)
    plt.hist2d(x2_cc,y_test[:,:,1].flatten(), norm = LogNorm(),bins=[100,100])
    plt.xlabel("$x_2$ predicted",size=14)
    plt.ylabel("$x_2$ true",size=14)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tick_params("both",length=12,width=2)
    plt.tick_params("both",which="minor",length=8,width=2)
    
    plt.subplot(223)
    plt.hist2d(y1_cc,y_test[:,:,3].flatten(), norm = LogNorm(),bins=[100,100])
    plt.xlabel("$y_1$ predicted",size=14)
    plt.ylabel("$y_1$ true",size=14)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tick_params("both",length=12,width=2)
    plt.tick_params("both",which="minor",length=8,width=2)
    
    plt.subplot(224)
    plt.hist2d(y2_cc,y_test[:,:,4].flatten(), norm = LogNorm(),bins=[100,100])
    plt.xlabel("$y_2$ predicted",size=14)
    plt.ylabel("$y_2$ true",size=14)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tick_params("both",length=12,width=2)
    plt.tick_params("both",which="minor",length=8,width=2)
    
    plt.tight_layout()
#plt.plot(y1,y1_t)

#%%
plot_confusion_hist(trained_models[4],"5")
plt.savefig(r"C:\Users\balin\Desktop\7. félév\compsim\doc\visualization\model_no{}.png".format("5"))