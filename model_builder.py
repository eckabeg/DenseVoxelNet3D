from keras import layers, models, regularizers
import config as CONFIG
from residual_connections_builder import ResidualConnectionsBuilder

class ModelBuilder:
    
    def OwnNet(num_classes, input_shape, grouping):
        if(grouping > 1):
            input_shape = input_shape + (grouping,)
        model = models.Sequential()
        model.add(layers.Input(input_shape))
        model.add(layers.Conv3D(16, (5, 5, 5), strides=(1, 1, 1), padding="same", activation='relu', kernel_regularizer=regularizers.l2(CONFIG.CONV_REGULARIZERS)))  # Input: 3D grid
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling3D((2, 2, 2)))

        model.add(layers.Conv3D(32, (3, 3, 3), padding="same", activation='relu', kernel_regularizer=regularizers.l2(CONFIG.CONV_REGULARIZERS)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling3D((2, 2, 2)))

        model.add(layers.Conv3D(64, (3, 3, 3), strides=(2, 2, 2), padding="same", activation='relu', kernel_regularizer=regularizers.l2(CONFIG.CONV_REGULARIZERS)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling3D((2, 2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(num_classes, activation='softmax'))
        return model

    def AlexNet(num_classes, input_shape, grouping):
        if(grouping > 1):
            input_shape = input_shape + (grouping,)
        model = models.Sequential()
        model.add(layers.Input(input_shape))
        model.add(layers.Conv3D(96, (5, 5, 5), (2, 2, 2), padding="same"))  # Input: 3D grid
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling3D((3, 3, 3), (2, 2, 2), padding="valid"))

        model.add(layers.Conv3D(256, (3, 3, 3), padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling3D((3, 3, 3), (2, 2, 2), padding="same"))

        model.add(layers.Conv3D(384, (3, 3, 3), padding="same", activation='relu'))
        model.add(layers.Conv3D(384, (3, 3, 3), padding="same", activation='relu'))
        model.add(layers.Conv3D(256, (3, 3, 3), padding="same", activation='relu'))
        model.add(layers.MaxPooling3D((3, 3, 3), (2, 2, 2), padding="same"))

        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dense(num_classes, activation='softmax'))
        return model
    
    def ResNet(num_classes, input_shape, grouping):
        if grouping > 1:
            input_shape = input_shape + (grouping,)

        inputs = layers.Input(shape=input_shape)

        # Initial Convolution and Pooling
        x = layers.Conv3D(64, 7, strides=2, padding="same", use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling3D(3, strides=2, padding='same')(x)

        # Residual Blocks
        num_blocks = [2, 2, 2, 2]
        filters = 64
        for i, num_block in enumerate(num_blocks):
            for j in range(num_block):
                stride = 1 if j > 0 else 2
                x = ResidualConnectionsBuilder.residual_block(x, filters, stride=stride)
            filters *= 2  # Double filters at each stage

        # Final Layers
        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Dense(512, activation='relu')(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        # Create Model
        model = models.Model(inputs, outputs)
        return model

    def TemporalNet(num_classes, input_shape, grouping):
        if(grouping > 1):
            input_shape = (grouping,) + input_shape
        model = models.Sequential()
        model.add(layers.Input(input_shape))
        model.add(layers.Conv3D(64, (3, 3, 3), activation='relu', padding="same"))
        model.add(layers.MaxPooling3D((2, 2, 2), padding="same"))
        model.add(layers.Conv3D(128, (3, 3, 3), activation='relu', padding="same"))
        model.add(layers.MaxPooling3D((2, 2, 2), padding="same"))
        model.add(layers.Conv3D(256, (3, 3, 3), activation='relu', padding="same"))
        model.add(layers.MaxPooling3D((2, 2, 2), padding="same"))
        model.add(layers.Conv3D(128, (1, 3, 3), activation='relu', padding="same"))
        model.add(layers.Conv3D(256, (1, 3, 3), activation='relu', padding="same"))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(num_classes, activation='softmax'))
        return model