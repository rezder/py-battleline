import numpy as np
import tensorflow as tf
import sys
import argparse
import batt as batttrain
import zmq
import signal
import time

tf.logging.set_verbosity(tf.logging.ERROR)

noFeature=88

def loadModel(model_dir,model_type,encoding):
    m,hidden_units = batttrain.build_estimator(model_dir,model_type,encoding)
    return m

def bytesToMposs(buf):
    v=np.frombuffer(buf,dtype=np.uint8)
    size=len(v)
    if size%noFeature!=0:
        raise NameError("Incomplete data")
    m=np.reshape(v,(-1,noFeature))
    m=m.astype(np.int64)
    return m

def stop(context):
    print("User stop program")
    context.close()


def main(_):
    print("Current libzmq version is %s" % zmq.zmq_version())
    print("Current  pyzmq version is %s" % zmq.__version__)
    context=zmq.Context()
    socket=context.socket(zmq.REP)
    socket.bind("tcp://*:"+str(FLAGS.port))
    model=loadModel(FLAGS.model_dir,FLAGS.model_type,FLAGS.feature_encoding)
    signal.signal(signal.SIGINT,lambda: stop(context))
    signal.signal(signal.SIGTERM,lambda: stop(context))
    while True:
        try:
            print("wait for request")
            b=socket.recv()
            m=bytesToMposs(b)
            print("recieved moves:",str(m.shape))
            print(str(m))
            t=time.time()
            res=model.predict_proba(input_fn=lambda: batttrain.predict_fn(m))
            print("Time to make move ",str(time.time()-t))
            proba=[]
            for v in res:
                proba.append(v[1])
            print("sends: ",str(proba),str(len(proba)))
            proba_bytes=np.array(proba,dtype=np.float64).tobytes()
            socket.send(proba_bytes)
            print("send bytes ",str(len(proba_bytes)))
        except zmq.ZMQError as e:
            print(e.msg)
            break

    socket.close()
    if not context.closed:
        context.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./model",
        help="Directory for model."
    )
    parser.add_argument(
        "--feature_encoding",
        type=int,
        default=1,
        help="Valid model types: {'1) one hot', '2) embedding'}."
    )
    parser.add_argument(
        "--model_type",
        type=int,
        default=3,
        help="Valid model types: {'1', '2','3'}."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="server port"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
