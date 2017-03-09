import os
import logging
import sys
import time
import math
from datetime import datetime

import tensorflow as tf
import numpy as np

from text_encoder import TextEncoder
from text_encoder import  QuestionEncoder
import coattention_layer
from bi_lstm_encoder import BidirectionalLSTMEncoder
from quora_classifier import QuoraClassifier
from utils.padder import pad_sequences
from utils.util import ConfusionMatrix, Progbar, minibatches
from utils.defs import LABELS
from utils.data_loader import load_quora_data, load_embeddings

from model import Model

logger = logging.getLogger("209_project")


class QuoraModel(Model):
    """
    Implements a recurrent neural network with an embedding layer and
    single hidden layer.
    This network will predict a sequence of labels (e.g. PER) for a
    given token (e.g. Henry) using a featurized window around the token.
    """
    def __init__(self, config, pretrained_embeddings, train_size, report=None):
        self.config = config
        self.report = report
        self.pretrained_embeddings = pretrained_embeddings
        self.train_size = train_size
        self.learning_rate = self.config.learning_rate
        self.grad_norm = None

        # Defining placeholders.
        self.input_placeholder1 = None
        self.input_placeholder2 = None
        self.labels_placeholder = None
        self.mask_placeholder1 = None
        self.mask_placeholder2 = None
        self.dropout_placeholder = None

        self.build()

###################### Build the model ##############################

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, self.max_length, n_features), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape (None, self.max_length), type tf.int32
        mask_placeholder:  Mask placeholder tensor of shape (None, self.max_length), type tf.bool
        dropout_placeholder: Dropout value placeholder (scalar), type tf.float32

        TODO: Add these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
            self.mask_placeholder
            self.dropout_placeholder

        HINTS:
            - Remember to use self.max_length NOT Config.max_length

        (Don't change the variable names)
        """
        ### YOUR CODE HERE (~4-6 lines)
        self.input_placeholder1 = tf.placeholder(tf.int32, shape=(None, self.config.max_timestep))
        self.input_placeholder2 = tf.placeholder(tf.int32, shape=(None, self.config.max_timestep))
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, ))
        self.mask_placeholder1 = tf.placeholder(tf.bool, shape=(None, self.config.max_timestep))
        self.mask_placeholder2 = tf.placeholder(tf.bool, shape=(None, self.config.max_timestep))
        self.dropout_placeholder = tf.placeholder(tf.float32)

        ### END YOUR CODE

    def create_feed_dict(self, text1_batch, text2_batch, mask1_batch, mask2_batch, labels_batch=None, dropout=1):
        """Creates the feed_dict for the dependency parser.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.
        Hint: When an argument is None, don't add it to the feed_dict.

        Args:
            inputs_batch: A batch of input data.
            mask_batch:   A batch of mask data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        ### YOUR CODE (~6-10 lines)
        feed_dict = {self.input_placeholder1: text1_batch,
                     self.input_placeholder2: text2_batch,
                     self.mask_placeholder1: mask1_batch,
                     self.mask_placeholder2: mask2_batch,
                     self.dropout_placeholder: dropout}
        if not labels_batch is None:
            feed_dict[self.labels_placeholder] = labels_batch

        ### END YOUR CODE
        return feed_dict

    def add_embedding(self, inputs):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:

        TODO:
            - Create an embedding tensor and initialize it with self.pretrained_embeddings.
            - Use the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, max_length, n_features, embed_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, max_length, n_features * embed_size).

        HINTS:
            - You might find tf.nn.embedding_lookup useful.
            - You can use tf.reshape to concatenate the vectors. See
              following link to understand what -1 in a shape means.
              https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#reshape.

        Returns:
            embeddings: tf.Tensor of shape (None, max_length, n_features*embed_size)
        """
        ### YOUR CODE HERE (~4-6 lines)
        embeddings = tf.nn.embedding_lookup(self.pretrained_embeddings, inputs)

        ### END YOUR CODE
        return embeddings

    def add_logit_op(self):
        """Adds the unrolled RNN:
            h_0 = 0
            for t in 1 to T:
                o_t, h_t = cell(x_t, h_{t-1})
                o_drop_t = Dropout(o_t, dropout_rate)
                y_t = o_drop_t U + b_2

        TODO: There a quite a few things you'll need to do in this function:
            - Define the variables U, b_2.
            - Define the vector h as a constant and inititalize it with
              zeros. See tf.zeros and tf.shape for information on how
              to initialize this variable to be of the right shape.
              https://www.tensorflow.org/api_docs/python/constant_op/constant_value_tensors#zeros
              https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#shape
            - In a for loop, begin to unroll the RNN sequence. Collect
              the predictions in a list.
            - When unrolling the loop, from the second iteration
              onwards, you will HAVE to call
              tf.get_variable_scope().reuse_variables() so that you do
              not create new variables in the RNN cell.
              See https://www.tensorflow.org/versions/master/how_tos/variable_scope/
            - Concatenate and reshape the predictions into a predictions
              tensor.
        Hint: You will find the function tf.pack (similar to np.asarray)
              useful to assemble a list of tensors into a larger tensor.
              https://www.tensorflow.org/api_docs/python/array_ops/slicing_and_joining#pack
        Hint: You will find the function tf.transpose and the perms
              argument useful to shuffle the indices of the tensor.
              https://www.tensorflow.org/api_docs/python/array_ops/slicing_and_joining#transpose

        Remember:
            * Use the xavier initilization for matrices.
            * Note that tf.nn.dropout takes the keep probability (1 - p_drop) as an argument.
            The keep probability should be set to the value of self.dropout_placeholder

        Returns:
            pred: tf.Tensor of shape (batch_size, max_length, n_classes)
        """

        with tf.variable_scope("QuoraModel"):

            text1 = self.add_embedding(self.input_placeholder1)
            text2 = self.add_embedding(self.input_placeholder2)
            dropout_rate = self.dropout_placeholder

            # step 1: Text encoder
            textEncoder = TextEncoder(self.config)
            with tf.variable_scope("TextEncoder"):
                text1_preds = textEncoder.add_prediction_op(text1)
                tf.get_variable_scope().reuse_variables()
                text2_preds = textEncoder.add_prediction_op(text2)

            # apply mask to each encoding output
            #text1_encoding = tf.boolean_mask(text1_preds, self.mask_placeholder1)
            #text2_encoding = tf.boolean_mask(text2_preds, self.mask_placeholder2)
            text1_encoding = text1_preds * tf.expand_dims(tf.to_float(self.mask_placeholder1), -1)
            text2_encoding = text2_preds * tf.expand_dims(tf.to_float(self.mask_placeholder2), -1)

            # take last timestep as encoding
            #text1_encoding = text1_preds[:, -1]
            #text2_encoding = text2_preds[:, -1]

            # # # step 2: concatenate two encodings
            # text1_encoding = tf.reduce_sum(text1_encoding, 1)
            # text2_encoding = tf.reduce_sum(text2_encoding, 1)
            # encodings = tf.concat(1, [text1_encoding, text2_encoding])


            # more complex alternative
            # step 2: Co-attention layer
            # non-linearity on one question encoding
            question_encoder = QuestionEncoder(self.config)
            text1_encoding = question_encoder.add_prediction_op(text1_encoding)
            encodings = coattention_layer.encode(text1_encoding, text2_encoding)

            # step 3: Bi-LSTM
            biLSTMEncoder = BidirectionalLSTMEncoder(self.config)
            encodings = biLSTMEncoder.add_prediction_op(encodings)
            _shape = encodings.get_shape()[1] * encodings.get_shape()[2].value
            encodings = tf.reshape(encodings, [-1, _shape.value]) # flatten

            # step 4: Decoder/Quora classifier
            classifier = QuoraClassifier(self.config)
            logits = classifier.add_prediction_op(encodings)

        assert logits.get_shape().as_list() == [None, self.config.n_classes_quora], \
            "predictions are not of the right shape. Expected {}, got {}"\
                .format([None, self.config.n_classes_quora],
                        logits.get_shape().as_list())
        return logits

    def add_prediction_op(self, logits):
        preds = tf.argmax(logits, axis=1)
        return preds

    def add_loss_op(self, logits):
        """Adds Ops for the loss function to the computational graph.

        TODO: Compute averaged cross entropy loss for the predictions.
        Importantly, you must ignore the loss for any masked tokens.

        Hint: You might find tf.boolean_mask useful to mask the losses on masked tokens.
        Hint: You can use tf.nn.sparse_softmax_cross_entropy_with_logits to simplify your
                    implementation. You might find tf.reduce_mean useful.
        Args:
            pred: A tensor of shape (batch_size, max_length, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        ### YOUR CODE HERE (~2-4 lines)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels_placeholder)
        loss = tf.reduce_mean(cross_entropy)

        ### END YOUR CODE
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        ### YOUR CODE HERE (~1-2 lines)
        batch = tf.Variable(0, trainable=False)
        n_batches = self.train_size / self.config.batch_size + 1
        self.learning_rate = tf.train.exponential_decay(self.config.learning_rate, batch, n_batches, self.config.lr_decay,
                                                   staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        grads_vars = optimizer.compute_gradients(loss)
        grads = [pair[0] for pair in grads_vars]
        self.grad_norm = tf.global_norm(grads)
        train_op = optimizer.apply_gradients(grads_vars, global_step=batch)

        ### END YOUR CODE
        return train_op


################## Run training and evaluation #########################

    def preprocess_sequence_data(self, input_list):
        """Preprocess sequence data for the model. For example, apply padding.

        Args:
            input_list: A list of tuples (intput_text1, intput_text2, label) that are turned into indices.
        Returns:
            A new list of vectorized input tuples plus the masks. i.e. (input_text1, input_text2, label, mask1, mask2)
        """

        input_text1_list, input_text2_list, labels = zip(*input_list) # this will unzip the list of tuples
        padded_input1_and_masks = pad_sequences(input_text1_list, self.config.max_timestep)
        text1_padded, mask1_list = zip(*padded_input1_and_masks)
        padded_input2_and_masks = pad_sequences(input_text2_list, self.config.max_timestep)
        text2_padded, mask2_list = zip(*padded_input2_and_masks)
        labels = np.array(labels)
        # labels = np.array(labels).reshape((-1, 1)) # tf requires labels to have shape (batch_size, 1)
        result = zip(text1_padded, text2_padded, labels, mask1_list, mask2_list)
        return result

    def consolidate_predictions(self, data_raw, data, preds):
        """
        apply masking to predictions
        Then group the input_text_2, input_text_2, labels, predicted_labels, in that order
        Note here the inputs and labels are strings, instead of indices

        Args:
            data_raw: raw text data read from file, a list of tuples (input1, input2, label)
            data: processed data that has been turned into indices. a list of tuples (input1, input2, label)
            preds: predictions made by the model. a list of indices
        """
        assert len(data_raw) == len(data)
        assert len(data_raw) == len(preds)

        ret = []
        for i, (input_text1, input_text2) in enumerate(data_raw):
            labels_predicted = preds[i]
            label = data[i][2] # 3rd element is label
            # make sure types are int
            assert labels_predicted.shape == ()
            assert label.shape == ()
            # assert type(label) == int
            ret.append([input_text1, input_text2, label, labels_predicted])
        return ret

    def predict_on_batch(self, sess, input_text1_batch, input_text2_batch, mask1_batch, mask2_batch):
        feed = self.create_feed_dict(text1_batch=input_text1_batch,
                                     text2_batch=input_text2_batch,
                                     mask1_batch=mask1_batch,
                                     mask2_batch=mask2_batch)
        predictions = sess.run(self.pred, feed_dict=feed)  # pick the class that has highest probability
        return predictions

    def train_on_batch(self, sess, input_text1_batch, input_text2_batch, labels_batch, mask1_batch, mask2_batch):
        feed = self.create_feed_dict(text1_batch=input_text1_batch,
                                     text2_batch=input_text2_batch,
                                     mask1_batch=mask1_batch,
                                     mask2_batch=mask2_batch,
                                     labels_batch=labels_batch,
                                     dropout=self.config.dropout)
        _, loss, grad_norm, learning_rate = sess.run([self.train_op, self.loss, self.grad_norm, self.learning_rate], feed_dict=feed)
        reversed_feed = self.create_feed_dict(text1_batch=input_text2_batch,
                                     text2_batch=input_text1_batch,
                                     mask1_batch=mask2_batch,
                                     mask2_batch=mask1_batch,
                                     labels_batch=labels_batch,
                                     dropout=self.config.dropout)
        _, loss_r, grad_norm_r, learning_rate_r = sess.run([self.train_op, self.loss, self.grad_norm, self.learning_rate], feed_dict=reversed_feed)
        return (loss+loss_r)/2.0, (grad_norm+grad_norm_r)/2, (learning_rate+learning_rate_r)/2


    def output(self, sess, inputs_raw, inputs):
        """
        Reports the output of the model on examples (uses helper to featurize each example).
        """
        # always require valid inputs arg
        # if inputs is None:
        #     inputs = self.preprocess_sequence_data(self.helper.vectorize(inputs_raw))

        preds = []
        prog = Progbar(target=1 + int(len(inputs) / self.config.batch_size))
        for i, batch in enumerate(minibatches(inputs, self.config.batch_size, shuffle=False)):
            # Ignore predict
            batch = batch[:2] + batch[3:]
            preds_ = self.predict_on_batch(sess, *batch)
            preds += list(preds_)
            prog.update(i + 1, [])
        return self.consolidate_predictions(inputs_raw, inputs, preds)


    def evaluate(self, sess, examples, examples_raw, best_score):
        """Evaluates model performance on @examples.

        This function uses the model to predict labels for @examples and constructs a confusion matrix.

        Args:
            sess: the current TensorFlow session.
            examples: A list of vectorized input/output pairs.
            examples: A list of the original input/output sequence pairs.
        Returns:
            The F1 score for predicting tokens as named entities.
        """
        confusion_matrix = ConfusionMatrix(labels=LABELS)

        true_positives, false_negatives, false_positives = 0., 0., 0.
        output = self.output(sess, examples_raw, examples)
        for _, _, label, label_predicted in output:
            confusion_matrix.update(label, label_predicted)
            if label_predicted == 1:
                if label == 1:
                    true_positives += 1
                else:
                    false_positives += 1
            else: # label_predicted == 0
                if label == 1:
                    false_negatives += 1

        p = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        r = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        # print predicted results
        if (f1 > best_score):
            logger.info('best new score: f1 = ' + str(f1))
            for q1_raw, q2_raw, label, label_predicted in output:
                logger.debug('\n' + q1_raw + q2_raw + \
                      ' correct: ' + str(label) + \
                      ', predicted: ' + str(label_predicted))

        logger.debug("Confusion matrix:\n" + confusion_matrix.as_table())
        logger.debug("Scores:\n" + confusion_matrix.summary())
        # logger.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)
        logger.info('f1: ' + str(f1))

        return f1



def run(config):
    # Set up some parameters.

    train, dev, train_raw, dev_raw = load_quora_data(data_path='data/quora', tiny=config.debug)
    embeddings = load_embeddings(data_path='data/quora', wordvec_dim=config.embed_size)

    report = None  # Report(Config.eval_output)

    with tf.Graph().as_default():
        logger.info("Building model...", )
        start = time.time()
        model = QuoraModel(config, embeddings, len(train))
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)

        with tf.Session() as session:
            if config.model_path:
                logger.info("restoring model: " + config.model_path)
                start = time.time()
                saver.restore(session, config.model_path)
                logger.info("took %.2f seconds", time.time() - start)
            else:
                session.run(init)
            model.fit(session, saver, train, dev, train_raw, dev_raw)