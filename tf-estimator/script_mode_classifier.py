import numpy as np
import os
import tensorflow as tf

INPUT_TENSOR_NAME = 'inputs'


def estimator_fn(run_config, params):
    feature_columns = [tf.feature_column.numeric_column(INPUT_TENSOR_NAME, shape=[29])]
    return tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                      hidden_units=[10, 20, 10],
                                      n_classes=2,
                                      config=run_config)


def serving_input_fn(params):
    feature_spec = {INPUT_TENSOR_NAME: tf.FixedLenFeature(dtype=tf.float32, shape=[29])}
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)()


def train_input_fn(training_dir, params):
    """Returns input function that would feed the model during training"""
    return _generate_input_fn(training_dir, 'train_data.csv')


def eval_input_fn(training_dir, params):
    """Returns input function that would feed the model during evaluation"""
    return _generate_input_fn(training_dir, 'test_data.csv')


def _generate_input_fn(training_dir, training_filename):
    training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=os.path.join(training_dir, training_filename),
        target_dtype=np.int,
        features_dtype=np.float32)

    return tf.estimator.inputs.numpy_input_fn(
        x={INPUT_TENSOR_NAME: np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True)()



def _parse_args():

    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()
    
     # Create the Estimator
    # mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=args.model_dir)
    feature_columns = [tf.feature_column.numeric_column(INPUT_TENSOR_NAME, shape=[29])]
    my_classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                      hidden_units=[10, 20, 10],
                                      n_classes=2,
                                      config=run_config)

    
    train_spec = tf.estimator.TrainSpec(train_input_fn, training_dir = args.train, max_steps=20000)
    eval_spec = tf.estimator.EvalSpec(eval_input_fn, training_dir = args.train)
    tf.estimator.train_and_evaluate(mnist_classifier, train_spec, eval_spec)
