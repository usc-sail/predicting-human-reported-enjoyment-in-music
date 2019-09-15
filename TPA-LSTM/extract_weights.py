import os
import tensorflow as tf

from lib.setup import params_setup, logging_config_setup, config_setup
from lib.model_utils import create_graph, load_weights, print_num_of_trainable_parameters


def main():
    para = params_setup()
    logging_config_setup(para)

    graph, model, data_generator = create_graph(para)

    with tf.Session(config=config_setup(), graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        load_weights(para, sess, model)
        print_num_of_trainable_parameters()

        try:
            # EXTRACT WEIGHTS HERE
            for variable in tf.global_variables(): # tf.trainable_variables():
                print(variable)

            # VIEW FETCHABLE OPS
            graph = tf.get_default_graph()
            print([op for op in parent_ops if graph.is_fetchable(op)])

        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        finally:
            print('Weights extracted. Stop.')


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
