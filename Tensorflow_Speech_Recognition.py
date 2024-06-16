import tensorflow as tf
from tensorflow import keras, train
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler
from sklearn.model_selection import train_test_split
import IPython
from IPython import display
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix,multilabel_confusion_matrix

path = os.getcwd() + '\mini_speech_commands'

# print(os.listdir(path))

path = Path(path)


labels = []

for i in path.iterdir():
    
    labels.append(i.name)
        
labels.pop(0)

print(labels)

# labels = np.array(labels)

# print(labels)


train_data,val_data = keras.utils.audio_dataset_from_directory(path,batch_size=32,validation_split=0.2,seed=42,subset='both',output_sequence_length=16000)


one =  next(train_data.as_numpy_iterator())

print(one[0][2])

print(one[1][2])

print(one[0].shape)
print("Train Shape")

label_names = np.array(train_data.class_names)

print(label_names)


print(train_data.element_spec)

def squeeze(audio,labels):
    aud = tf.squeeze(audio,axis=-1)
    return aud,labels


train_data = train_data.map(squeeze,tf.data.AUTOTUNE)
val_data = val_data.map(squeeze,tf.data.AUTOTUNE)




test_data = val_data.shard(2,index=0)
val_data = val_data.shard(2,index=1)

example_features,example_labels = next(test_data.as_numpy_iterator())

print(example_features)
print(example_labels)

test_fea = tf.concat(list(test_data.map(lambda fea,lab:fea)),axis=0)


# for i in range(9):
#     plt.subplot(3,3,i+1)
#     plt.plot(example_features[i])
#     plt.title(label_names[example_labels[i]])
#     plt.ylim([-1.1, 1.1])
#     plt.yticks(np.arange(-1.2, 1.2, 0.2))
#     plt.xticks([0,7500,15000])
#     plt.tight_layout()
    
# plt.show()
    
def spectrogram(waveform):
    spect = tf.signal.stft(waveform,frame_length=255,frame_step=128)
    
    spect = tf.abs(spect)
    spect = spect[..., tf.newaxis]
    
    return spect


spect_exa = example_features[2]

   
# plt.imshow((spectrogram(spect_exa)))

# plt.show()

for i in range(3):
    labels = label_names[example_labels[i]]
    
    waveform = example_features[i]
    print("Label:",labels)
    display.display(display.Audio(waveform,rate=16000,autoplay=True))
    
def plot_spectorgram(spectrogram,axis):
    if len(spectrogram.shape) > 2 :
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram,axis=-1)
        log_spec = np.log(spectrogram.T + np.finfo(float).eps)
        
    
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    axis.pcolormesh(X, Y, log_spec)
    


    
# fig,ax = plt.subplots(2,figsize=(12,9))
# ax[0].plot(example_features[5])
# ax[0].set_title(label_names[example_labels[5]])
# ax[0].set_xlim([0,16000])

# plot_spectorgram(spectrogram(example_features[5]),ax[1])
# ax[1].set_title('Spectrogram')
# plt.suptitle(label_names[example_labels[5]])
    
    
# plt.show()

def make_spec_ds(ds):
     
    return ds.map(lambda feature,label:(spectrogram(feature),label),num_parallel_calls=tf.data.AUTOTUNE)
    
train_spect_ds = make_spec_ds(train_data)
test_spect_ds = make_spec_ds(test_data)
val_spect_ds = make_spec_ds(val_data)


features_spect,labels_spect = next(train_spect_ds.as_numpy_iterator())


fig,axis = plt.subplots(3,3,figsize=(12,9))

for i in range(9):
    
    r = i // 3

    c = i % 3    

    ax = axis[r][c]

    plot_spectorgram(features_spect[i],ax)

    ax.set_title(label_names[labels_spect[i]])

    plt.tight_layout()

plt.show()

owntest_path = os.getcwd() + '\Test'

owntest_data = keras.utils.audio_dataset_from_directory(owntest_path,batch_size=32,seed=42,output_sequence_length=16000,shuffle=False)

owntest_data = owntest_data.map(squeeze,tf.data.AUTOTUNE)

owntest_spect_ds = make_spec_ds(owntest_data)

print(owntest_spect_ds)

print(train_spect_ds)


train_spect_ds = train_spect_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
test_spect_ds = test_spect_ds.cache().prefetch(tf.data.AUTOTUNE)
val_spect_ds = val_spect_ds.cache().prefetch(tf.data.AUTOTUNE)
owntest_spect_ds = owntest_spect_ds.cache().prefetch(tf.data.AUTOTUNE)

## load_model 
model = keras.models.load_model('Model\my_model.h5')

print(model.summary())

# num_lables = 8

# input_shape = features_spect.shape[1:]

# norm_layer = keras.layers.Normalization()

# norm_layer.adapt(data=train_spect_ds.map(map_func=lambda spec, label: spec))

# model = keras.Sequential([keras.layers.Input(shape=input_shape),
#                           keras.layers.Resizing(32,32),
#                           norm_layer,
#                           keras.layers.Conv2D(32,3,activation='relu'),
#                           keras.layers.Conv2D(64,3,activation='relu'),
#                           keras.layers.MaxPooling2D(),
#                           keras.layers.Dropout(0.25),
#                           keras.layers.Flatten(),
#                           keras.layers.Dense(128, activation='relu'),
#                           keras.layers.Dropout(0.5),
#                           keras.layers.Dense(num_lables,activation='softmax')

#                           ])

# print(model.summary())


# model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

# his = model.fit(train_spect_ds,epochs=50,callbacks=keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,verbose=1),validation_data=val_spect_ds)

# metrics = his.history

# print(metrics)

# loss = his.history['loss']

# val_loss = his.history['val_loss']

# accuracy = his.history['accuracy']

# val_acc = his.history['val_accuracy']

# plt.plot(loss,c='r')
# plt.plot(val_loss,c='b')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')

# plt.legend(['loss','val_loss'])

# plt.show()

# plt.plot(accuracy,c='g')
# plt.plot(val_acc,c='orange')

# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.legend(['Accuracy','val_accuracy'])

# plt.show()

y_pred_logit = model.predict(test_spect_ds)

y_pred = np.argmax(y_pred_logit,axis=1)
print(y_pred)

y_true = []

for fea,lab in test_spect_ds.as_numpy_iterator():
    y_true.append(lab)


y_true = np.array(y_true)

y_true = y_true.flatten()

print(y_true)

test_acc = np.equal(y_true,y_pred).sum()/len(y_true)

print(test_acc * 100)

conf = confusion_matrix(y_true,y_pred,labels=np.arange(1,9,1))

ConfusionMatrixDisplay(conf,display_labels=label_names).plot()

plt.show()

# test_fea = tf.concat(list(test_spect_ds.map(lambda fe,la:fe)),axis=0)


random = np.random.randint(1,800,9)


print(random)
# print(test_fea.shape)
# print(test_fea[2])

for index,i in enumerate(random):
    
    plt.subplot(3,3,index+1)
    plt.bar(label_names,y_pred_logit[i])
    plt.title(f"Prediction :{label_names[y_pred[i]]} Actual :{label_names[y_true[i]]}",c='green' if label_names[y_true[i]] == label_names[y_pred[i]] else 'red')
    
    display.display(display.Audio(test_fea[i],rate=16000,autoplay=True))
    plt.tight_layout()

    
plt.show()


## test

test_pred = model.predict(owntest_spect_ds)

print(label_names[np.argmax(test_pred,axis=1)])

plt.bar(label_names,test_pred[0])

plt.show()

# model.save('Model\my_model.h5')




features_spect_own,labels_spect_own = next(owntest_spect_ds.as_numpy_iterator())

fig,axe = plt.subplots(2,2,figsize=(12,9))

for i in range(4):
    
    r = i // 2

    c = i % 2   

    ax = axe[r][c]

    plot_spectorgram(features_spect_own[i],ax)

    plt.tight_layout()

plt.show()

