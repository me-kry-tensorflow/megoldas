import tensorflow as tf
import matplotlib as mpl
import os
import numpy

with open("versions.txt", "w") as the_file:
    the_file.write("tensorflow:" + tf.version.VERSION + os.linesep)
    the_file.write("matplotlib:" + mpl.__version__ + os.linesep)
    the_file.write("numpy:" + numpy.version.version + os.linesep)
