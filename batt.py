"""Train a neural network using tenserflow.

The classifier is a standard neural net work with hidden units.
The --model_type option specify the number of hidden_units.
The data must be battleline machine data.
"""
import tensorflow as tf
import sys
import argparse
import datetime
import math
import posreader as batt
# Ignore warnings
tf.logging.set_verbosity(tf.logging.INFO)

# Define flds
FLDS = batt.createFlds()
ENC_ONE_HOT = 1
ENC_EMBEDDING = 2
FLAGS = None


def build_estimator(model_dir, model_type, encoding):
    """Creates a estimater

    Args:
    model_dir: The model directory if it contain a model the model is used for retraining.
    model_type: the specific model.
    encoding: The encoding type for class features ENC_ONE_HOT or ENC_EMBEDDING

    Returns:
    A tensorflow DNNClassifier
    """
    features = []
    for i, fld in enumerate(FLDS):
        if hasattr(fld, 'values'):
            f = tf.contrib.layers.sparse_column_with_keys(
                column_name=fld.name, keys=fld.values, dtype=tf.int64)
            if encoding == ENC_ONE_HOT:
                f = tf.contrib.layers.one_hot_column(f)
            else:
                no = len(fld.values)
                d = int(round(math.log2(no)))
                f = tf.contrib.layers.embedding_column(f, dimension=d)
        else:
            scale = fld.scale
            f = tf.contrib.layers.real_valued_column(
                fld.name, normalizer=lambda x: x / scale)

        features.append(f)
    hidden_units = [100, 100]
    if model_type == 2:
        hidden_units = [200, 200]
    elif model_type == 3:
        hidden_units = [1000, 500, 250]

    m = tf.contrib.learn.DNNClassifier(
        model_dir=model_dir,
        feature_columns=features,
        hidden_units=hidden_units)
    return m, hidden_units


def predict_fn(x):
    """A data feeder function for making prediction.

    Arg:
    x: a np matrix with one column per feature. The order
    of features should match FLDS.

    Returns:
    feature_cols: A map of feature names and tensor.(python :variables python-enable-yapf-format-on-save t)
    lable: always None
    """
    lable = None
    feature_cols = {}
    size = x.shape[0]
    for i, fld in enumerate(FLDS):
        if hasattr(fld, 'values'):
            feature_cols[fld.name] = tf.SparseTensor(
                indices=[[j, 0] for j in range(size)],
                values=x[:, i],
                dense_shape=[size, 1])
        else:
            feature_cols[fld.name] = tf.constant(
                x[:, i], dtype=tf.int64, shape=[size, 1])
    return feature_cols, lable


def input_fn(batch_size, file_pathern, no_epochs):
    """A data feeder function for training and evaluation.

    Arg:
    batch_size: The batch size the chunk size of data to deliver on every call.
    file_pathern: The file pathern of the data files.
    no_epochs: The number of epochs. The number of times the whole data should
    be procesed.

    Returns:
    feature_cols: A map of feature names and tensor.
    lable: A tensor of lables.
    """
    defaults = [tf.constant([-1], dtype=tf.int64)] * (len(FLDS) + 1)
    examples_op = tf.contrib.learn.read_batch_examples(
        file_pathern,
        batch_size=batch_size,
        reader=tf.TextLineReader,
        num_epochs=no_epochs,
        parse_fn=lambda x: tf.decode_csv(x, defaults))

    label = examples_op[:, len(FLDS)]
    size = tf.to_int64(tf.shape(examples_op)[0])
    indices = tf.transpose(
        [tf.range(size),
         tf.zeros_like(examples_op[:, 0], dtype=tf.int64)])
    feature_cols = {}
    for i, fld in enumerate(FLDS):
        if hasattr(fld, 'values'):
            feature_cols[fld.name] = dense_to_sparse(examples_op[:, i],
                                                     indices, size)
        else:
            #size=tf.to_int64(tf.shape(examples_op[:, i])[0])
            feature_cols[fld.name] = tf.reshape(examples_op[:, i], [-1, 1])

    return feature_cols, label


def dense_to_sparse(dense_tensor, indices, size):
    """Convert a dense tensor to a sparse tensor.

    This is just how class features is stored basicly a
    normal tensor in a sparse tensor. Later it is transformed.

    Args:
    dense_tensor: The tenser to convert.
    indices: The index tensor.
    size: The number of rows.

    Returns: A sparse tensor.
    """
    values = dense_tensor
    shape = [size, 1]
    return tf.SparseTensor(indices=indices, values=values, dense_shape=shape)


def train_and_eval(model_dir, model_type, batch_size, file_pathern_learn,
                   file_pathern_test, no_epochs, encoding):
    """Train and evaluate a model.

    Args:
    model_dir: The model directory.
    model_type: The model type.
    file_pathern_learn: The train file pathern.
    file_pathern_test: The evaluate file pathern.
    no_epochs: The number of epochs.
    encoding: The class feature encoding. One hot (ENC_ONE_HOT)
    or embedding (ENC_EMBEDDING).

    """
    startTs = datetime.datetime.today()
    print("Start at %s" % startTs)
    print("Model directory: %s" % model_dir)

    m, hidden_units = build_estimator(model_dir, model_type, encoding)
    print("Number of fields: {}".format(len(FLDS)))
    print("Hidden units: {}".format(hidden_units))
    print("Batch size: {}".format(batch_size))
    if encoding == 1:
        print("Type feature encoding one hot")
    else:
        print("Type feature encoding embedding")

    print("Epochs: {}".format(no_epochs))
    m.fit(input_fn=lambda: input_fn(batch_size, file_pathern_learn, no_epochs))
    trainTs = datetime.datetime.today()
    results = m.evaluate(
        input_fn=lambda: input_fn(batch_size, file_pathern_test, 1))
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))

    endTs = datetime.datetime.today()

    print("Train time: {}".format(str(trainTs - startTs)))
    print("Eval time: {}".format(str(endTs - trainTs)))
    print("Total time: {}".format(str(endTs - startTs)))


def main(_):
    train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.batch_size,
                   FLAGS.file_pathern_learn, FLAGS.file_pathern_cv,
                   FLAGS.no_epochs, FLAGS.feature_encoding)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage="Train a battleline tensorflow model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./model",
        help="Base directory for output models.")
    parser.add_argument(
        "--feature_encoding",
        type=int,
        default=1,
        help="Valid model types: {'1) one hot', '2) embedding'}.")
    parser.add_argument(
        "--model_type",
        type=int,
        default=3,
        help="Valid model types: {'1', '2','3'}.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Number of records in training steps.")
    parser.add_argument(
        "--file_pathern_learn",
        type=str,
        default="/home/rho/BoltDb/tf/big/move.cvs.t1*",  # "./move.cvs.learn",
        help="Pathern to the training data files.")
    parser.add_argument(
        "--file_pathern_cv",
        type=str,
        default="/home/rho/BoltDb/tf/big/move.cvs.t1*",  #"./move.cvs.cv",
        help="Pathern to the test data file(s).")
    parser.add_argument(
        "--no_epochs", type=int, default=2, help="Number of epochs")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
