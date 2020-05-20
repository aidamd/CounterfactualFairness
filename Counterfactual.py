import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np


class Counterfactual():
    def __init__(self, params, vocab, hate_w):
        for key in params:
            setattr(self, key, params[key])
        self.vocab = vocab
        self.hate_weights = hate_w

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
        self.drop_out =  tf.placeholder(tf.float32)
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
        self.H = tf.nn.dropout(tf.concat(self.states, 1), rate=self.drop_out)

        _, self.counter_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                                         self.cf_embed,
                                                         dtype=tf.float32,
                                                         sequence_length=self.X_len)
        self.cf_H = tf.nn.dropout(tf.concat(self.states, 1), rate=self.drop_out)

        X_logits = tf.layers.dense(self.H, 2, name="hate", reuse=True)
        cf_logits = tf.layers.dense(self.cf_H, 2, name="hate", reuse=True)
        logit_weights = tf.gather(self.weights, self.y_hate)

        xentropy = tf.losses.sparse_softmax_cross_entropy(
            labels=self.y_hate,
            logits=X_logits,
            weights=logit_weights
        )
        self.loss = tf.reduce_mean(xentropy)
        self.loss += tf.reduce_mean(tf.abs(tf.subtract(X_logits, cf_logits)))
        self.predicted = tf.argmax(X_logits, 1)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(self.predicted, self.y_hate), tf.float32))
        self.oprimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate) \
            .minimize(self.loss)

    def feed_dict(self, batch, test=False, predict=False):
        feed_dict = {
            self.X: [t["input"] for t in batch],
            self.cf: [t["counter"] for t in batch],
            self.X_len: [t["length"] for t in batch],
            self.keep_prob: 1 if test else self.drop_ratio,
            self.embedding_placeholder: self.embeddings,
            self.weights: self.hate_weights
            }
        if not predict:
            feed_dict[self.y_hate] = [t["hate"] for t in batch]
        return feed_dict

    def train(self, batches):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as self.sess:
            init.run()
            epoch = 1
            while True:
                train_loss, train_acc = 0, 0
                test_acc = 0
                train_idx, test_idx = train_test_split(np.arange(len(batches)),
                                                       test_size=0.15, shuffle=True)
                train_batches = [batches[i] for i in train_idx]
                test_batches = [batches[i] for i in test_idx]

                for batch in train_batches:
                    _, loss, acc = self.sess.run(
                        [self.oprimizer, self.loss, self.accuracy],
                        feed_dict=self.feed_dict(batch))
                    train_loss += loss
                    train_acc += acc

                for batch in test_batches:
                    acc = self.sess.run(
                        [self.accuracy],
                        feed_dict=self.feed_dict(batch, True))
                    test_acc += acc

                print("Epoch: %d, loss: %.4f, train: %.4f, test: %.4f" %
                          (epoch, train_loss / len(train_batches),
                           train_acc / len(train_batches),
                           test_acc / len(test_batches)))
                epoch += 1
                if epoch == self.epochs:
                    saver.save(self.sess, "saved_model/counter")
                    break

    def predict(self, batches):
        saver = tf.train.Saver()
        predicted = list()
        with tf.Session() as self.sess:
            saver.restore(self.sess, "saved_model/adversary/hate")
            for batch in batches:
                predicted.extend(list(self.sess.run(
                    self.predicted,
                    feed_dict=self.feed_dict(batch, True, True))))
        return predicted