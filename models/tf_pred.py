import numpy as np
import tensorflow as tf
import cv2

class Predictor():

    def __init__(self, img_path, input_sz, ckpt_path):
        self.image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        self.image = self.image.reshape(-1, input_sz)
        self.checkpoints = ckpt_path

    def predict_with_tf(self):
        saver = tf.train.import_meta_graph(self.checkpoints+'.meta')
        with tf.Session() as sess:
            saver.restore(sess, self.checkpoints)
            graph = tf.get_default_graph()
            X = graph.get_tensor_by_name('input/X:0')
            # Because logits is a subgraph, so if you want to get its output,
            # you need to get its tensor by the name of its last operation in that subgraph.
            logits = graph.get_tensor_by_name('output/output/BiasAdd:0')
            Z = logits.eval(feed_dict={X: self.image})
            y_pred = np.argmax(Z, axis=1)
            print(y_pred)

if __name__ == '__main__':
    predictor = Predictor('../datasets/mnist3.jpeg', 28*28, './tf_tmp_checkpoints/mnist_cnn_model')
    predictor.predict_with_tf()




