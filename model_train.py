import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from data_loader import df
from Models.CNN_model import model 
from metrics import precision, recall, specificity, f1_score, accuracy

custom_model = model()
train_images, val_images, test_images = df()

EPOCHS = 500
OPT = tf.keras.optimizers.Adam(learning_rate=0.001)

custom_model.compile(optimizer='adam',
                loss=tf.losses.CategoricalCrossentropy(),
                metrics=[accuracy, precision, recall, specificity, f1_score])

earlystopping = EarlyStopping(
                              monitor = 'val_f1_score', 
                              mode = 'max',
                              patience = 10,
                              restore_best_weights=True,
                              verbose = 1)

filepath = './best_weights_model_custom.hdf5'

checkpoint    = ModelCheckpoint(filepath,
                                monitor = 'val_f1_score', 
                                mode = 'max',
                                save_best_only=True, 
                                verbose = 1)

callback_list = [earlystopping, checkpoint]

custom_model_history = custom_model.fit(train_images,
                                        validation_data=val_images, 
                                        callbacks=callback_list, 
                                        epochs=EPOCHS)
