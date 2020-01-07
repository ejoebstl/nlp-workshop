import tensorflow as tf
import os

def get_tpu_strategy():
    TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']
    tf.keras.backend.clear_session()
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=TPU_WORKER)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    return tf.distribute.experimental.TPUStrategy(resolver)