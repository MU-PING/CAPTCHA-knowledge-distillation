import tensorflow as tf


class Teacher(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.sequential = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(60, 200, 3)),
                tf.keras.layers.Conv2D(512, 3, activation='relu'),
                tf.keras.layers.Conv2D(512, 3, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.2),

                tf.keras.layers.Conv2D(256, 3, activation='relu'),
                tf.keras.layers.Conv2D(256, 3, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.2),

                tf.keras.layers.Conv2D(128, 3, activation='relu'),
                tf.keras.layers.Conv2D(128, 3, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Flatten()
            ],
        )
        self.additionLayers = [AdditionLayer() for _ in range(6)]
        
        self.sequential.summary()

    def call(self, input):
        x = self.sequential(input)

        outputs = []
        for additionLayer in self.additionLayers:
            outputs.append(additionLayer(x))

        return outputs

class Student(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.sequential = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(60, 200, 3)),
                tf.keras.layers.Conv2D(256, 3, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.2),

                tf.keras.layers.Conv2D(128, 3, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.2),

                tf.keras.layers.Conv2D(64, 3, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Flatten()
            ],
        )
        self.additionLayers = [AdditionLayer() for _ in range(6)]

    def call(self, input):
        x = self.sequential(input)

        outputs = []
        for additionLayer in self.additionLayers:
            outputs.append(additionLayer(x))

        return x

class AdditionLayer(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense = tf.keras.layers.Dense(64, activation='relu')
        self.outputs = tf.keras.layers.Dense(36)

    def call(self, input):
        x = self.dense(input)
        x = self.dropout(x)
        output = self.outputs(x)
        return output

class Distiller(tf.keras.Model):

    def __init__(self, teacher, student):
        super().__init__()
        self.teacher = teacher
        self.student = student

teacher = Teacher()
inputlayer = tf.keras.Input((60, 200, 3))
teacher.build(input_shape=(None, 60, 200, 3))
teacher.summary()