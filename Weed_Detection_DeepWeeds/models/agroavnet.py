from tensorflow.keras.layers import Input, Conv2D 
from tensorflow.keras.layers import MaxPool2D, Flatten, Dense 
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential

def AgroAVNET():
    input = Input(shape=(224, 224, 3))
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='truncated_normal')(input)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='truncated_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='truncated_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='truncated_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='truncated_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='truncated_normal')(x)
    x = Flatten()(x) 
    outputs = Dense(9, activation='sigmoid', name='fc9')(x)
    model = Model(inputs=input, outputs=outputs)
    model.summary()
    print(outputs.shape)
    return model

AgroAVNET()

