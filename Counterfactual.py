import tensorflow as tf


class Counterfactual():
    def __init__(self, params, vocab):
        for key in params:
            setattr(self, key, params[key])
        self.vocab = vocab

    def build(self):
        tf.reset_default_graph()
        self.keep_prob = tf.placeholder(tf.float32)
        self.X = tf.placeholder(tf.int32,
                                [None, None],
                                name="inputs")
        # Counterfactuals of the inputs. If the input doesn't have a SGT
        # the counterfactual is the sentence itself, otherwise its one of
        # its counterfactuals
        self.cf = tf.placeholder(tf.int32,
                                [None, None],
                                name="counterfactuals")

        # Counterfactuals have the same length as the sentence
        # so we don't need to redefine it
        self.X_len = tf.placeholder(tf.int32,
                                    [None],
                                    name="sequence_len")

        self.y_hate = tf.placeholder(tf.int64, [None], name="hate_labels")
        self.weights = tf.placeholder(tf.float32, [None], name="weights")
        embedding_W = tf.Variable(tf.constant(0.0,
                                              shape=[len(self.vocab), 300]),
                                  trainable=False, name="Embed")

        self.embedding_placeholder = tf.placeholder(tf.float32,
                                                    shape=[len(self.vocab), 300])

        self.embedding_init = embedding_W.assign(self.embedding_placeholder)

        # [batch_size, sent_length, emb_size]
        self.X_embed = tf.nn.embedding_lookup(self.embedding_placeholder,
                                                          self.X)
        self.cf_embed = tf.nn.embedding_lookup(self.embedding_placeholder,
                                                          self.cf)

        # encoder

        fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.h_size, name="ForwardRNNCell",
                                          state_is_tuple=False)
        bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.h_size, name="BackwardRNNCell",
                                          state_is_tuple=False, reuse=False)
        _, self.states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                                         self.X_embed,
                                                         dtype=tf.float32,
                                                         sequence_length=self.X_len)
        self.H = tf.concat(self.states, 1)

        _, self.counter_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                                         self.cf_embed,
                                                         dtype=tf.float32,
                                                         sequence_length=self.X_len)
        self.cf_H = tf.concat(self.states, 1)

        X_logits = tf.layers.dense(self.H, 2, name="hate", reuse=True)
        cf_logits = tf.layers.dense(self.cf_H, 2, name="hate", reuse=True)
        logit_weights = tf.gather(self.weights, self.y_hate)

        xentropy = tf.losses.sparse_softmax_cross_entropy(
            labels=self.y_hate,
            logits=X_logits,
            weights=logit_weights
        )
        loss = tf.reduce_mean(xentropy)
        loss += tf.reduce_mean(tf.abs(tf.subtract(X_logits, cf_logits)))
        predicted = tf.argmax(X_logits, 1)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(predicted, self.y_hate), tf.float32))




    def train(self):


    def predict(self):