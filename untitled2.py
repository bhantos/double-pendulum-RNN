# -*- coding: utf-8 -*-
from generate_data import *
from time import time, strftime
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
from tensorflow.keras.layers import SimpleRNN, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorboard.plugins.hparams import api as hp
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
HP_NUM_LAYERS = hp.HPARAM("num_layers", hp.Discrete([1,2]))
HP_LOSS = hp.HPARAM("loss", hp.Discrete(["mean_absolute_error","mean_squared_error", loss_fixed_len(0.2,1,10**(-1)),
                                         loss_fixed_len(0.2, 1, 10**(-2)), loss_fixed_len(1, 1, 10**(-2))]))
HP_LAYER_TYPE = hp.HPARAM("layer_type", hp.Discrete(["VRNN","LSTM"]))
HP_ACT_FUN = hp.HPARAM("act_fun", hp.Discrete(["tanh","relu","sigmoid"]))

METRIC = "mae"

with tf.summary.create_file_writer("logs\hparam_tuning").as_default():
  hp.hparams_config(hparams=[HP_NUM_LAYERS, HP_LOSS,HP_LAYER_TYPE, HP_ACT_FUN], 
                    metrics=[hp.Metric(METRIC, display_name="mae")])

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
def train_test_model(hparams):#NUM_LAYERS, LAYER_TYPE, EPOCHS = 10, BATCH_SIZE= 150, LOSS = "mean_absolute_error"
    time1 = time()
    
    name = "DoublePendulumRNN_{}".format(strftime("%m%d-%H%M%S"))
    logdir = "logs\{}".format(name)
    
    model = Sequential()
    if hparams[LAYER_TYPE] == "LSTM":
        for i in range(hparams[NUM_LAYERS]):
            model.add(LSTM(6, input_shape=x.shape[1:],
                                   activation=hparams[HP_ACT_FUN],
                                   return_sequences=True))
    if LAYER_TYPE == "VRNN":
        for i in range(NUM_LAYERS):
            model.add(SimpleRNN(6, input_shape=x.shape[1:],
                                   activation=hparams[HP_ACT_FUN],
                                   return_sequences=True))
    
    opt = tf.keras.optimizers.Adam(lr=0.1,decay=1e-2)
    model.compile(loss=hparams[HP_LOSS], optimizer=opt ,metrics=['mae'],)
    
    if hparams[NUM_LAYERS] == 2:
        fit_model = model.fit(x_train,
                  y_train,
                  epochs=1,
                  batch_size=150,
                  validation_data = (x_valid,y_valid))
        
    else:
        fit_model = model.fit(x_train,
                  y_train,
                  epochs=1,
                  batch_size=100,
                  validation_data = (x_valid,y_valid))
    _, mae = model.evaluate(x_test,y_test)
    print("Training took ",round(time()-time1,3)," s")
    model.save(r"C:\Users\balin\Desktop\7. félév\compsim\proj1\models\{}_{}_loss{}.model".format(hparams[HP_LAYER_TYPE],
                                                                                    hparams[HP_NUM_LAYERS],
                                                                                    hparams[LOSS]))
    return mae

def loss_fixed_len(alpha,beta,coef):
    def loss(y_true, y_pred):
        loss_value = kb.mean(kb.abs(y_pred  - y_true))  \
            +  coef*(alpha * kb.square(kb.sqrt(kb.square(y_pred[0]) +  kb.square(y_pred[1])) - 1 ) \
            + beta * kb.square(kb.sqrt(kb.square(y_pred[3]) + kb.square(y_pred[4])) - 1))
        return loss_value
    
    return loss

#%%
def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        mae = train_test_model(hparams)
        tf.summary.scalar(METRIC, mae,step=1)
        
#%%
session_num = 0

time1 = time()

for num_layers in HP_NUM_LAYERS.domain.values:
    for loss in HP_LOSS.domain.values:
        for act_fun in HP_ACT_FUN.domain.values:
            for layer_type in HP_LAYER_TYPE.domain.values:
                
                hparams = { HP_NUM_LAYERS: num_layers,
                            HP_LOSS: loss,
                            HP_ACT_FUN: act_fun,
                            HP_LAYER_TYPE: layer_type}
                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                run(r"C:\Users\balin\Desktop\7. félév\compsim\proj1\logs" + run_name, hparams)
                session_num += 1

print(time()-time1)



#%%
#model = train_model(LOSS = "mean_absolute_error", EPOCHS = 50, BATCH_SIZE = 150) #loss_fixed_len(0.0002,0.001)
model = train_model( 1,"VRNN",LOSS = "mean_absolute_error",EPOCHS = 20, BATCH_SIZE = 150)


#%%

model.evaluate(x_test,y_test)

y_predict = model.predict(x_test)
a = 1 #choose a-th generated time series 
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