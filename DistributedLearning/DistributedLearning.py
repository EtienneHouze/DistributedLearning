from __future__ import print_function

import tensorflow as tf
import sys
import time
import argparse
from src import *

FLAGS = None

def main():
    worker = int(sys.argv[1])


    cluster = tf.train.ClusterSpec({"ps": ["10.229.34.21:2222"],"worker" : ["10.229.47.42:2222"]})
    if (worker == 1):
        server = tf.train.Server(cluster, job_name = "worker", task_index = 0)
    else:
        server = tf.train.Server(cluster, job_name = "local", task_index = 0)

    print ("starting server #{}".format(task_number))

    server.start()
    server.join()

if __name__ == "__main__":
    main()