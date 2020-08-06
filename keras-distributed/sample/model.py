
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def main():
    mirrored_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
                        
    with mirrored_strategy.scope():
        seq_data = keras.Input(shape=(None, 1),  name="seq_data")
        seq_lengths = keras.Input(shape=(), name="seq_lengths", dtype=tf.int32)

        mask = keras.layers.Lambda(lambda x: tf.sequence_mask(x))(seq_lengths)
        conv = layers.Conv1D(32, 3, strides=1, padding='same', activation='relu')(seq_data)
        rnn = layers.Bidirectional(layers.GRU(32, return_sequences=True))(conv,mask=mask)
        #rnn = layers.Bidirectional(layers.GRU(32, return_sequences=True))(conv)
        dense = layers.Dense(5, name="signal_mask")(rnn)
        model = keras.Model(inputs=[seq_data, seq_lengths], outputs=[dense])

        model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001),
            loss=tf.keras.backend.sparse_categorical_crossentropy
        )

    batch_size = 1
    time_step = 1000
    signal_length = 1000
    sequences = np.random.rand(batch_size, time_step, 1)
    seq_lengths = np.array([signal_length])
    signal_mask = np.random.randint(5, size=(1, time_step))

    dataset = tf.data.Dataset.from_tensor_slices(
        (sequences, seq_lengths,  signal_mask)).repeat(1000).batch(1)\
        .map(lambda a,b,c: ({"seq_data":a,"seq_lengths":b}, {"signal_mask":c}))

    model.fit(dataset, epochs=1)
    
main()
