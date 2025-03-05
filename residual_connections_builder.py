from keras import layers, regularizers

class ResidualConnectionsBuilder:

    def residual_block(x, filters, conv_regularizer, kernel_size=3, stride=1):
        shortcut = x  # Save input for skip connection
    
        # First convolution
        x = layers.Conv3D(filters, kernel_size, strides=stride, padding='same', use_bias=False, kernel_regularizer=conv_regularizer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Second convolution
        x = layers.Conv3D(filters, kernel_size, strides=1, padding='same', use_bias=False, kernel_regularizer=conv_regularizer)(x)
        x = layers.BatchNormalization()(x)
        
        # Adjust shortcut dimensions if needed
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv3D(filters, 1, strides=stride, use_bias=False, kernel_regularizer=conv_regularizer)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        # Add skip connection
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        
        return x  # Return the updated tensor
