"""Starts a battleline bot server

The server loads a tensorflow model and uses it to evaluate
battleline moves requested by clients via zmq.
"""
import numpy as np
import tensorflow as tf
import sys
import argparse
import batt as batttrain
import zmq
import signal
import time

tf.logging.set_verbosity(tf.logging.ERROR)

noFeatureOld = 92
noFeatureNew = 19 * 18


def loadModel(model_dir, model_type, encoding):
    """Loads the tensorflow model

    Args:
    model_dir: The model directory if it contain a model the model is used for retraining.
    model_type: the specific model.
    encoding: The encoding type for class features ENC_ONE_HOT or ENC_EMBEDDING

    Returns:
    A tensorflow DNNClassifier
    """
    flds = batttrain.createFlds(model_type)
    m, hidden_units = batttrain.build_estimator(model_dir, model_type,
                                                encoding, flds)
    return m, flds


def bytesToMposs(buf):
    """Convert bytes to a np matrix of battleline machine positions(old)."""
    v = np.frombuffer(buf, dtype=np.uint8)
    size = len(v)
    if size % noFeatureOld != 0:
        raise NameError("Incomplete data")
    m = np.reshape(v, (-1, noFeatureOld))
    m = m.astype(np.int64)
    return m


def bytesToMFloats(buf):
    """Convert bytes to a np matrix of battleline machine 2 flag analysis floats"""
    v = np.frombuffer(buf, dtype=np.float32)
    size = len(v)
    if size % noFeatureNew != 0:
        raise NameError("Incomplete data")
    m = np.reshape(v, (-1, noFeatureNew))
    return m


def stop(context):
    """Stop the zmq listner

    Args:
    context: A zmq context
    """
    print("User stop program")
    context.close()


def main(_):
    print("Current libzmq version is %s" % zmq.zmq_version())
    print("Current  pyzmq version is %s" % zmq.__version__)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:" + str(FLAGS.port))
    model, flds = loadModel(FLAGS.model_dir, FLAGS.model_type,
                            FLAGS.feature_encoding)
    signal.signal(signal.SIGINT, lambda: stop(context))
    signal.signal(signal.SIGTERM, lambda: stop(context))
    while True:
        try:
            print("wait for request")
            sys.stdout.flush()
            b = socket.recv()
            if FLAGS.model_type < 4:
                m = bytesToMposs(b)
            else:
                m = bytesToMFloats(b)

            print("recieved moves:", str(m.shape))
            print(str(m))
            print(m)
            sys.stdout.flush()
            t = time.time()
            res = model.predict_proba(
                input_fn=lambda: batttrain.predict_fn(m, flds))
            print("Time to make move ", str(time.time() - t))
            proba = []
            for v in res:
                proba.append(v[1])
            print("sends: ", str(proba), str(len(proba)))
            proba_bytes = np.array(proba, dtype=np.float64).tobytes()
            socket.send(proba_bytes)
            print("send bytes ", str(len(proba_bytes)))
            sys.stdout.flush()
        except zmq.ZMQError as e:
            print(e.msg)
            sys.stdout.flush()
            break

    socket.close()
    if not context.closed:
        context.close()


if __name__ == "__main__":
    usage = """Start a battleline bot server that uses a trained tensorflow \
model to evaluete battleline moves."""
    parser = argparse.ArgumentParser(
        usage=usage, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./model",
        help="Directory for model.")
    parser.add_argument(
        "--feature_encoding",
        type=int,
        default=1,
        help="Valid model types: {'1) one hot', '2) embedding'}.")
    parser.add_argument(
        "--model_type",
        type=int,
        default=4,
        help="Valid model types: {'1', '2','3','4'}.")
    parser.add_argument("--port", type=int, default=5555, help="server port")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
