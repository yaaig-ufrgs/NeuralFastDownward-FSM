import sys
import tensorflow as tf
from tensorflow.python.platform import gfile

# $ python load_mode.py 'trained_model_filename.pb'

GRAPH_PB_PATH = str(sys.argv[1])

graph = tf.GraphDef()
File = open(GRAPH_PB_PATH,"rb")
graph.ParseFromString(File.read())

for layer in graph.node:
    print(layer.name)
