import os
import tensorflow as tf

from lib.setup import params_setup, logging_config_setup, config_setup
from lib.model_utils import create_graph, load_weights, print_num_of_trainable_parameters
from lib.train import train
from lib.test import test


def main():
    para = params_setup()
    logging_config_setup(para)

    print("Creating graph...")
    graph, model, data_generator = create_graph(para)
    print("Done creating graph.")

    with tf.Session(config=config_setup(), graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        print("Loading weights...")
        load_weights(para, sess, model)
        print_num_of_trainable_parameters()

        # PRINT NAMES OF TENSORS THAT ARE ALPHAS
        # example name: "model/rnn/cond/rnn/multi_rnn_cell/cell_0/cell_0/temporal_pattern_attention_cell_wrapper/attention/Sigmoid:0"
        # for item in [n.name for n in tf.get_default_graph().as_graph_def().node
        #              if (n.name.find("temporal_pattern_attention_cell_wrapper/attention")!=-1 and
        #                  n.name.find("Sigmoid")!=-1)]:
        #     print(item)

        # Print names of ops
        # for op in tf.get_default_graph().get_operations():
        #     if(op.name.find("ben_multiply")!=-1):
        #         print(str(op.name))

        # PRINT REG KERNEL AND BIAS
        # reg_weights = [v for v in tf.global_variables() if v.name == "model/dense_2/kernel:0"][0]
        # reg_bias = [v for v in tf.global_variables() if v.name == "model/dense_2/bias:0"][0]
        # print("Reg Weights:", sess.run(reg_weights))
        # print("Reg Bias:", sess.run(reg_bias) * data_generator.scale[0])

        try:
            if para.mode == 'train':
                train(para, sess, model, data_generator)
            elif para.mode == 'test':
                print("Evaluating model...")
                test(para, sess, model, data_generator)

        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        finally:
            print('Stop')


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
