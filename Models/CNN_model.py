from tensorflow.keras import Sequential, Input, BatchNormalization, MaxPool2D, Conv2D, Flatten, Dense, Dropout

IMAGE_SIZE = [128, 128]

custom_model = Sequential([
    Input(shape=(*IMAGE_SIZE, 3)),
    Conv2D(16, 3, activation='relu', padding='same'),
    Dropout(0.1),
    Conv2D(16, 3, activation='relu', padding='same'),
    MaxPool2D((2, 2)),

    Conv2D(32, 3, activation='relu', padding='same'),
    Dropout(0.1),
    Conv2D(32, 3, activation='relu', padding='same'),
    MaxPool2D((2, 2)),

    Conv2D(64, 3, activation='relu', padding='same'),
    Dropout(0.1),
    Conv2D(64, 3, activation='relu', padding='same'),
    MaxPool2D((2, 2)),

    Conv2D(128, 3, activation='relu', padding='same'),
    Dropout(0.1),
    Conv2D(128, 3, activation='relu', padding='same'),
    MaxPool2D((2, 2)),

    Conv2D(256, 3, activation='relu', padding='same'),
    Conv2D(256, 3, activation='relu', padding='same', name = 'last_conv_layer'),
    MaxPool2D((2, 2)),

    Flatten(),
    Dropout(0.4),   # 0.4
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),   # 0.5
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),   # 0.3
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),   # 0.3
    Dense(4, activation='softmax')        
], name = "cnn_model")

custom_model.summary(line_length=100)
