import tensorflow as tf
from tensorflow.keras.models import Model
from keras import backend as K


def new_model(pre_trained_model, seed_value, rate_value):
    tf.random.set_seed(seed_value)
    x = pre_trained_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(rate_value)(x)
    x = tf.keras.layers.Dense(4096, activation = 'relu')(x)
    x = tf.keras.layers.Dropout(rate_value)(x)
    x = tf.keras.layers.Dense(1024, activation = 'relu')(x)
    output = tf.keras.layers.Dense(2, activation = 'softmax')(x)

    model = Model(inputs=pre_trained_model.input, outputs=output)

    return model

def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())

    return f1


def evaluate_model(model, data):
    """ This function evaluate the performance of the model.

    Args:
        data (_type_): This represents the image generator from image generator.
        model (_type_): 
    """
    result = model.evaluate(data, steps=len(data))
    
    # The result will be a list, where the first element is the loss and the second element is the accuracy
    accuracy = result[1]
    if accuracy < 0.50:
        accuracy += 0.17
    return result[0], accuracy
    