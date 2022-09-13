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

    def call(self, input, training):
        x = self.sequential(input, training)

        outputs = []
        for additionLayer in self.additionLayers:
            outputs.append(additionLayer(x, training))

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

    def call(self, input, training):
        x = self.sequential(input, training)

        outputs = []
        for additionLayer in self.additionLayers:
            outputs.append(additionLayer(x, training))

        return x

class AdditionLayer(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense = tf.keras.layers.Dense(64, activation='relu')
        self.outputs = tf.keras.layers.Dense(36)

    def call(self, input, training):
        x = self.dense(input)
        x = self.dropout(x, training)
        output = self.outputs(x)
        return output

class Distiller(tf.keras.Model):

    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(self, optimizer, metrics, student_loss_fn, distiller_loss_fn, alpha=0.1, temperature=3):

        """ Configure the distiller.
        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """

        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(x_train, y_train):

        # Forward pass of teacher
        teacher_predictions = self.teacher(x_train, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x_train, training=True)
            
teacher = Teacher()
inputlayer = tf.keras.Input((60, 200, 3))
teacher.build(input_shape=(None, 60, 200, 3))
teacher.summary()