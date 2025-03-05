from keras import layers, models, Model
from residual_connections_builder import ResidualConnectionsBuilder


class ModelBuilder:

    def channel_attention(input_tensor, reduction=16):
        filters = input_tensor.shape[-1]
        se = layers.GlobalAveragePooling3D()(input_tensor)
        se = layers.Dense(filters // reduction, activation='relu')(se)
        se = layers.Dense(filters, activation='sigmoid')(se)
        se = layers.Reshape((1, 1, 1, filters))(se)
        return layers.Multiply()([input_tensor, se])

    def spatial_attention(input_tensor):
        se = layers.Conv3D(1, (7, 7, 7), padding="same", activation='sigmoid')(input_tensor)
        return layers.Multiply()([input_tensor, se])

    def cbam_block(input_tensor):
        x = ModelBuilder.channel_attention(input_tensor)  # Apply channel attention
        x = ModelBuilder.spatial_attention(x)  # Apply spatial attention
        return x

    def se_block(input_tensor, reduction=16):
        """ Squeeze-and-Excitation block for 3D Convolutional Networks """
        filters = input_tensor.shape[-1]  # Get the number of channels
        se = layers.GlobalAveragePooling3D()(input_tensor)  # Squeeze: Compute global context
        se = layers.Dense(filters // reduction, activation='relu')(se)  # Excitation: Learn channel weights
        se = layers.Dense(filters, activation='sigmoid')(se)
        se = layers.Reshape((1, 1, 1, filters))(se)  # Reshape to apply channel-wise
        return layers.Multiply()([input_tensor, se])  # Scale input features

    def ModtionNet(num_classes, input_shape, conv_regularizer, dropout, activation_function):
        inputs = layers.Input(input_shape)

        x = layers.Conv3D(64, (3, 3, 3), activation=activation_function, kernel_regularizer=conv_regularizer)(inputs)
        x = layers.BatchNormalization()(x)

        x = layers.Conv3D(128, (3, 3, 3), activation=activation_function, kernel_regularizer=conv_regularizer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D((2, 2, 2))(x)

        x = layers.Conv3D(256, (3, 3, 3), activation=activation_function, kernel_regularizer=conv_regularizer)(x)
        x = layers.BatchNormalization()(x)

        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Dropout(dropout[0])(x)
        x = layers.Dense(512, activation=activation_function)(x)
        x = layers.Dropout(dropout[1])(x)
        x = layers.Dense(256, activation=activation_function)(x)
        x = layers.Dropout(dropout[2])(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        model = Model(inputs, outputs, name="MotionNet")
        return model

    def AlexNet(num_classes, input_shape, conv_regularizer, dropout, activation_function):
        model = models.Sequential()
        model.add(layers.Input(input_shape))
        model.add(layers.Conv3D(96, (5, 5, 5), (2, 2, 2), padding="same", kernel_regularizer=conv_regularizer))  # Input: 3D grid
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(activation_function))
        model.add(layers.MaxPooling3D((3, 3, 3), (2, 2, 2), padding="valid"))

        model.add(layers.Conv3D(256, (3, 3, 3), padding="same", kernel_regularizer=conv_regularizer))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(activation_function))
        model.add(layers.MaxPooling3D((3, 3, 3), (2, 2, 2), padding="same"))

        model.add(layers.Conv3D(384, (3, 3, 3), padding="same", activation=activation_function, kernel_regularizer=conv_regularizer))
        model.add(layers.Conv3D(384, (3, 3, 3), padding="same", activation=activation_function, kernel_regularizer=conv_regularizer))
        model.add(layers.Conv3D(256, (3, 3, 3), padding="same", kernel_regularizer=conv_regularizer))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation(activation_function))
        model.add(layers.MaxPooling3D((3, 3, 3), (2, 2, 2), padding="same"))

        model.add(layers.Flatten())
        model.add(layers.Dropout(dropout[0]))
        model.add(layers.Dense(2048, activation=activation_function))
        model.add(layers.Dropout(dropout[1]))
        model.add(layers.Dense(2048, activation=activation_function))
        model.add(layers.Dropout(dropout[2]))
        model.add(layers.Dense(num_classes, activation='softmax'))
        return model
    
    def ResNet(num_classes, input_shape, conv_regularizer, dropout, activation_function):
        inputs = layers.Input(shape=input_shape)

        # Initial Convolution and Pooling
        x = layers.Conv3D(64, 7, strides=2, padding="same", use_bias=False, kernel_regularizer=conv_regularizer)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling3D(3, strides=2, padding='same')(x)

        # Residual Blocks
        num_blocks = [2, 2, 2]
        filters = 64
        for i, num_block in enumerate(num_blocks):
            for j in range(num_block):
                stride = 1 if j > 0 else 2
                x = ResidualConnectionsBuilder.residual_block(x, filters, conv_regularizer, stride=stride)
            filters *= 2  # Double filters at each stage

        # Final Layers
        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Dropout(dropout[0])(x)
        x = layers.Dense(512, activation=activation_function)(x)
        x = layers.Dropout(dropout[1])(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        # Create Model
        model = models.Model(inputs, outputs)
        return model