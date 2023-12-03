import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras import Model, callbacks, layers, losses
import time
import os

BATCH_SIZE = 64
BUFFER_SIZE = 10000
EMBEDDING_DIM = 256
EPOCHS = 20
RNN_UNITS = 1024
SEQ_LENGTH = 100

class TextGenerator(Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.gru = layers.GRU(rnn_units, return_sequences=True, return_state=True)
        self.dense = layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)
        if return_state:
            return x, states
        else:
            return x

class OneStep(Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars
        self.temperature = temperature
        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(values=[-float('inf')] * len(skip_ids), indices=skip_ids, dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate(self, inputs, states=None):
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()
        predicted_logits, states = self.model(inputs=input_ids, states=states, return_state=True)
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits /= self.temperature
        predicted_logits += self.prediction_mask
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)
        predicted_chars = self.chars_from_ids(predicted_ids)
        return predicted_chars, states

def get_file_text(path):
    try:
        with open(path, 'rb') as f:
            text = f.read().decode(encoding='utf-8')
            f.close()
        return text
    except IOException as e:
        print(f'fail to open file: {path}, {e}')
        return ''

def vectorize(text):
    vocab = sorted(set(text))
    example_texts = ['abcdefg', 'xyz']
    chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')
    ids_from_chars = layers.StringLookup(vocabulary=list(vocab), mask_token=None)
    chars_from_ids = layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), mask_token=None, invert=True)
    return ids_from_chars, chars_from_ids

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('need file path input')
        sys.exit(1)
    path = sys.argv[1]
    msg = sys.argv[2]
    text = get_file_text(path)
    ids_from_chars, chars_from_ids = vectorize(text)
    all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
    sequences = ids_dataset.batch(SEQ_LENGTH+ 1, drop_remainder=True)
    dataset = sequences.map(split_input_target)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    vocab_size = len(ids_from_chars.get_vocabulary())
    loss = losses.SparseCategoricalCrossentropy(from_logits=True)
    model = TextGenerator(vocab_size, EMBEDDING_DIM, RNN_UNITS)
    model.compile(optimizer='adam', loss=loss)
    checkpoint = os.path.join(''), 'chkpt_{epoch}'
    checkpoint_callback = callbacks.ModelCheckpoint(filepath=checkpoint, save_weights_only=True)
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
    start = time.time()
    states = None
    next_char = tf.constant([msg])
    result = [next_char]
    for n in range(1000):
        next_char, states = one_step_model.generate(next_char, states=states)
        result.append(next_char)
    result = tf.strings.join(result)
    end = time.time()
    print(result[0].numpy().decode('utf-8'), '\n\n' + '_' * 80)
    print('\n runtime: ', end - start)
    tf.saved_model.save(one_step_model, 'one_step')
