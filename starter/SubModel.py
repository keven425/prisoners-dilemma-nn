class SubModel(object):
    """Abstracts a sub-model, like encoders, for example
    """
    def __init__(self, config):
        self.config = config

    def add_prediction_op(self, inputs, output_size=None):
        """Implements the core of the model that transforms a batch of input data into predictions.

        Args:
            inputs: A tensor of shape (batch_size, max_time, n_classes_input)
            output_size: n_classes_output if you want it to be different from n_classes_input
        Returns:
            outputs: A tensor of shape (batch_size, max_time, n_classes_output)
        """
        raise NotImplementedError("Each Model must re-implement this method.")


    #TODO: we need a way to incorporate pre-trained weights/initial state for models



