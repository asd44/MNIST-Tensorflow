import tensorflow as tf


def freeze_model(logdir, name):
    tf.train.write_graph(
        tf.get_default_graph(),
        logdir,
        name,
        False
    )


def load_model(frozen_model_path):
    with tf.gfile.GFile(frozen_model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
