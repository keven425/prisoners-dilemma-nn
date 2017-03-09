import tensorflow as tf
from SubModel import SubModel

class QuoraClassifier(SubModel):
    def add_prediction_op(self, inputs, output_size=None):
        """
        Takes in encoder outputs and use the last time-step to create an output that's of size num_classes (i.e. 2)

        Here we basically do a matrix multiplication to change to dimensions.

        Args:
            inputs: A tensor of shape (batch_size, max_time, n_classes_input)
            output_size: ignored
        Returns:
            outputs: A tensor of shape (batch_size, n_classes_quora)
        """
        with tf.variable_scope("QuoraClassifier"):
            input_size = inputs.get_shape()[-1]
            # create variables for W_Cl, b_Cl
            W_Cl = tf.get_variable("W_Cl", shape=(input_size, self.config.n_classes_quora), dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
            b_Cl = tf.get_variable("b_Cl", shape=(self.config.n_classes_quora,), dtype=tf.float32)

            outputs = tf.matmul(inputs, W_Cl) + b_Cl

        return outputs
