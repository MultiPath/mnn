import os
import tensorflow as tf


class NeuralNetwork():

    def train(self):
        print("TODO: implement proper training")

    def test(self):
        print("TODO: implement proper testing")

    def load(self):
        ckpt = tf.train.get_checkpoint_state(self.models_folder)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.models_folder, ckpt_name))
            return True
        else:
            return False

    def save(self, step):
        model_name = "model_" + str(step) + ".model"
        self.saver.save(self.sess, os.path.join(self.models_folder,
                                                model_name), global_step=step)
