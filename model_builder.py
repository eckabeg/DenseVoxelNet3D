from keras import layers, models

class ModelBuilder:
    
    def AlexNet(classes):
        model = models.Sequential()
        model.add(layers.Input((2, 64, 64, 64)))
        model.add(layers.Conv3D(96, (7, 7, 7), (2, 2, 2), padding="same", activation='relu'))  # Input: 3D grid
        model.add(layers.MaxPooling3D((3, 3, 3), (2, 2, 2), padding="same"))

        model.add(layers.Conv3D(256, (5, 5, 5), padding="same", activation='relu'))
        model.add(layers.MaxPooling3D((3, 3, 3), (2, 2, 2), padding="same"))

        model.add(layers.Conv3D(384, (3, 3, 3), padding="same", activation='relu'))
        model.add(layers.Conv3D(384, (3, 3, 3), padding="same", activation='relu'))
        model.add(layers.Conv3D(256, (3, 3, 3), padding="same", activation='relu'))
        model.add(layers.MaxPooling3D((3, 3, 3), (2, 2, 2), padding="same"))

        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(len(classes), activation='softmax'))
        return model

    def TemporalNet(classes):
        model = models.Sequential()
        model.add(layers.Input((2, 64, 64, 64)))
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
        model.add(layers.Dense(len(classes), activation='softmax'))
        return model
