import time
import datetime
import argparse

if __name__ == "__main__":
    startTs=datetime.datetime.today()
    time.sleep(1)
    trainTs=datetime.datetime.today()
    time.sleep(2)
    endTs=datetime.datetime.today()

    print("Train time: {}".format(str(trainTs-startTs)))
    print("Eval time: {}".format(str(endTs-trainTs)))
    print("Total time: {}".format(str(endTs-startTs)))
    hidden_units=[300,333,12]
    print("Hidden units: {}".format(hidden_units))

    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
      "--model_dir",
      type=str,
      default="./model",
      help="Base directory for output models."
    )
    parser.add_argument(
        "--bb",
        type="bool",
        default= False,
        help="Base directory for output models."
    )
    FLAGS, up=parser.parse_known_args()
    print(FLAGS.model_dir)
    print(FLAGS.bb)
