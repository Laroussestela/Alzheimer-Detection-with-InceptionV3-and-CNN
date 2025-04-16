from keras.applications import InceptionV3
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import BatchNormalization

def model():
    inc=InceptionV3(input_shape=(128,128,3),weights='imagenet',include_top=False)
    inc.summary()

    for i in inc.layers:
    i.trainable = True

    for layer in inc.layers:
    if layer.name == 'conv2d_80':
        break
    layer.trainable = False
    print('Capa ' + layer.name + ' congelada...')


    model_fine_tuning = Sequential()
    model_fine_tuning.add(inc)
    model_fine_tuning.add(layers.Flatten())
    model_fine_tuning.add(layers.Dense(128, activation='relu'))
    model_fine_tuning.add(BatchNormalization())
    model_fine_tuning.add(layers.Dropout(0.4)) 
    model_fine_tuning.add(layers.Dense(64, activation='relu'))
    model_fine_tuning.add(BatchNormalization())
    model_fine_tuning.add(layers.Dropout(0.4)) 
    model_fine_tuning.add(layers.Dense(4, activation='softmax'))

    model_fine_tuning.summary()

    return model_fine_tuning


