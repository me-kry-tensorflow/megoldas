import tensorflow as tf
import datetime


@tf.function
def my_func():
    a = tf.constant(1, name='a')
    b = tf.constant(2, name='b')
    c = tf.constant(3, name='c')
    d = tf.constant(4, name='d')
    result = tf.add(tf.multiply(a, b, name='multiply_a_b'), tf.add(c, d))


# Set up logging.
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = 'logs/func/%s' % stamp

writer = tf.summary.create_file_writer(logdir)

tf.summary.trace_on()

my_func()

with writer.as_default():
    tf.summary.trace_export(
      name="my_func_trace",
      step=0,
      profiler_outdir=logdir)
