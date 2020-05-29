import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
from utils import *
from random import *


class Counterfactual():
    def __init__(self, params, train, test, counter):
        for key in params:
            setattr(self, key, params[key])
        self.preprocess(train, test, counter)
        self.embeddings = load_embedding(self.vocab,
                                         "Data/humility_embeddings.txt",
                                         300)
        self.CV()
        self.test_model()

    def preprocess(self, train, test, counter):
        train = preprocess(train)
        counter = preprocess(counter)
        test = preprocess(test)

        self.train, self.test = dict(), dict()

        self.train["text"] = train["text"].values.tolist()
        self.train["ids"] = train["Tweet ID"].values.tolist()
        self.train["labels"] = train["hate"].values.tolist()
        self.train["perplex"] = {self.train["ids"][i]: train["perplexity"].tolist()[i]
                                 for i in range(train.shape[0])}

        self.vocab = learn_vocab(self.train["text"], self.vocab_size)

        self.train["tokens"] = tokens_to_ids(self.train["text"], self.vocab)

        self.test["text"] = test["text"].values.tolist()
        self.test["ids"] = test["Tweet ID"].values.tolist()
        self.test["labels"] = test["hate"].values.tolist()
        self.test["perplex"] = test["perplexity"].values.tolist()

        self.test["tokens"] = tokens_to_ids(self.test["text"], self.vocab)

        self.counter = dict()
        for name, group in counter.groupby(["Tweet ID"]):
            if name in self.train["perplex"]:
                counter = self.asymmetrics(name, group,
                                           self.train["perplex"][name],
                                           self.train["labels"][self.train["ids"].index(name)])
                self.counter[name] = tokens_to_ids(counter.reset_index()["text"].tolist(),
                                                   self.vocab,
                                                   )

        self.hate_weights = [1, 5]


    def asymmetrics(self, tweet, counters, perplex, hate):
        if self.asym:
            diffs = [abs(perplex - counters["perplexity"].tolist()[i])
                     for i in range(counters.shape[0])]
            diffs.sort()
            thresh = np.argmax([diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)])
            return counters.iloc[[i for i in range(counters.shape[0]) if
                                 abs(perplex - counters["perplexity"].tolist()[i]) <= diffs[thresh]]]
        else:
            return [] if hate else counters

    def CV(self):
        kfold = StratifiedKFold(n_splits=5)
        results = dict()
        i = 1
        for t_idx, v_idx in kfold.split(np.arange(len(self.train["tokens"])),
                                        self.train["labels"]):
            print("CV:", i)
            i += 1
            t_tokens, v_tokens = [self.train["tokens"][t] for t in t_idx], [self.train["tokens"][t] for t in v_idx]
            t_index, v_index = [self.train["ids"][t] for t in t_idx], [self.train["ids"][t] for t in v_idx]
            t_labels, v_labels = [self.train["labels"][t] for t in t_idx], [self.train["labels"][t] for t in v_idx]
            train_batches = get_batches(t_tokens,
                                        t_index,
                                        self.batch_size,
                                        self.vocab.index("<pad>"),
                                        hate=t_labels,
                                        counter=self.counter)
            val_batches = get_batches(v_tokens,
                                      v_index,
                                      self.batch_size,
                                      self.vocab.index("<pad>"),
                                      hate=v_labels,
                                      counter=self.counter)
            self.build()
            self.train_model(train_batches, val_batches)
            v_predictions = self.predict(val_batches)
            res = prediction_results(v_labels, v_predictions)
            for m in res:
                results.get(m, list()).append(res[m])
        print(results)
        for m in results:
            print(m, ":", sum(results[m]) / len(results[m]))

    def test_model(self):
        batches = get_batches(self.test["tokens"],
                              self.test["ids"],
                              self.batch_size,
                              self.vocab.index("<pad>"))

        test_predictions = self.predict(batches)
        _ = prediction_results(self.test["labels"],
                               test_predictions)


    def build(self):
        tf.reset_default_graph()
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
        self.drop_ratio =  tf.placeholder(tf.float32, name="drop")
        embedding_W = tf.Variable(tf.constant(0.0,
                                              shape=[len(self.vocab), 300]),
                                  trainable=False, name="Embed")

        self.embedding_placeholder = tf.placeholder(tf.float32,
                                                    shape=[len(self.vocab), 300])

        self.embedding_init = embedding_W.assign(self.embedding_placeholder)

        # [batch_size, sent_length, emb_size]
        self.X_embed = tf.nn.dropout(tf.nn.embedding_lookup(self.embedding_placeholder,
                                                          self.X),
                                     keep_prob=self.drop_ratio)
        self.cf_embed = tf.nn.dropout(tf.nn.embedding_lookup(self.embedding_placeholder,
                                                          self.cf),
                                      keep_prob=self.drop_ratio)

        # encoder

        fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.h_size, name="ForwardRNNCell",
                                          state_is_tuple=False)
        bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.h_size, name="BackwardRNNCell",
                                          state_is_tuple=False, reuse=False)
        _, self.states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                                         self.X_embed,
                                                         dtype=tf.float32,
                                                         sequence_length=self.X_len)
        #self.H = tf.nn.dropout(tf.concat(self.states, 1), keep_prob=self.drop_ratio)
        self.H = tf.concat(self.states, 1)

        _, self.counter_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                                         self.cf_embed,
                                                         dtype=tf.float32,
                                                         sequence_length=self.X_len)
        #self.cf_H = tf.nn.dropout(tf.concat(self.states, 1), keep_prob=self.drop_ratio)
        self.cf_H = tf.concat(self.states, 1)

        X_logits = tf.layers.dense(self.H, 2, name="hate")
        cf_logits = tf.layers.dense(self.cf_H, 2, name="hate", reuse=True)
        logit_weights = tf.gather(self.weights, self.y_hate)

        xentropy = tf.losses.sparse_softmax_cross_entropy(
            labels=self.y_hate,
            logits=X_logits,
            weights=logit_weights
        )
        self.loss = tf.reduce_mean(xentropy)
        self.diff = tf.reduce_mean(tf.abs(tf.subtract(X_logits, cf_logits)))
        self.loss += self.diff
        self.predicted = tf.argmax(X_logits, 1)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(self.predicted, self.y_hate), tf.float32))
        self.oprimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate) \
            .minimize(self.loss)

    def feed_dict(self, batch, test=False, predict=False):
        feed_dict = {
            self.X: np.array([t["input"] for t in batch]),
            self.cf: np.vstack([t["counter"][randrange(len(t["counter"]) - 1)]
                                if len(t["counter"]) > 1 else t["counter"][0]
                                for t in batch]),
            self.X_len: np.array([t["length"] for t in batch]),
            self.drop_ratio: 1 if test else self.drop_rate,
            self.embedding_placeholder: self.embeddings,
            self.weights: np.array(self.hate_weights)
            }

        if not predict:
            feed_dict[self.y_hate] = np.array([t["hate"] for t in batch])
        return feed_dict

    def train_model(self, train_batches, val_batches):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as self.sess:
            init.run()
            epoch = 1
            while True:
                train_loss, train_acc = 0, 0
                val_acc = 0

                for batch in train_batches:
                    _, loss, acc, diff = self.sess.run(
                        [self.oprimizer, self.loss, self.accuracy, self.diff],
                        feed_dict=self.feed_dict(batch))
                    print(diff)
                    train_loss += loss
                    train_acc += acc

                for batch in val_batches:
                    acc, pred = self.sess.run([self.accuracy, self.predicted],
                        feed_dict=self.feed_dict(batch, True))
                    val_acc += acc

                print("Epoch: %d, loss: %.4f, train: %.4f, test: %.4f" %
                          (epoch, train_loss / len(train_batches),
                           train_acc / len(train_batches),
                           val_acc / len(val_batches)))
                epoch += 1
                if epoch == self.epochs:
                    saver.save(self.sess, "saved_model/counter")
                    break


    def predict(self, batches):
        saver = tf.train.Saver()
        predicted = list()
        with tf.Session() as self.sess:
            saver.restore(self.sess, "saved_model/counter")
            for batch in batches:
                predicted.extend(list(self.sess.run(
                    self.predicted,
                    feed_dict=self.feed_dict(batch, True, True))))
        return predicted
