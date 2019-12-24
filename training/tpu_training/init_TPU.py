import tensorflow as tf


def connect_to_tpu(tpu_address):
    return tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address)


def init_tpu(tpu_ip):
    print("Trying to connect to a TPU node")

    print(("\n!!!MAKE SURE THE TPU ADDRESS IS CORRECT!!\n"
           "1.ip must be correct\n"
           "2.tpu must be turned on\n"
           "3.version must be 'nightly-2.x'\n"
           "4.tpu must be reachable (check with gce networking/connectivity test)\n"
           "if not this will hang!\n"), flush=True)

    tpu_address = 'grpc://' + tpu_ip + ':8470'
    print("Trying to connect to:", tpu_address, flush=True)
    resolver = connect_to_tpu(tpu_address)

    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
    return strategy, resolver
