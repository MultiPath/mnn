import os
import tensorflow as tf
from time import gmtime, strftime
import re

import mnn.config as conf


class NeuralNetwork():

    def __init__(self):
        self.static_config = conf.Configuration()
        self.static_config.load("static_config.json")

        self.dynamic_config = conf.Configuration()
        self.dynamic_config.load("dynamic_config.json")

    def train(self):
        print("TODO: implement proper training")

    def test(self):
        print("TODO: implement proper testing")

    def createDefaultFolders(self):
        self.log_folder = "logs/" + strftime("%d-%b-%Y-%H_%M_%S", gmtime())
        os.mkdir(self.log_folder)
        self.samples_folder = "samples/" +\
            strftime("%d-%b-%Y-%H_%M_%S", gmtime())
        os.mkdir(self.samples_folder)
        self.models_folder = "models/" +\
            strftime("%d-%b-%Y-%H_%M_%S", gmtime())
        os.mkdir(self.models_folder)

    def saveInitialConfig(self):
        self.static_config.save(
            self.models_folder + "/" + "static_config.json")
        self.dynamic_config.save(
            self.models_folder + "/" + "dynamic_config.json")

    def load(self):
        ckpt = tf.train.get_checkpoint_state(
            "models" + "/" + self.static_config.get("snapshot"))
        
        #TODO a dirty trick to get the iteration number
        num_iter = int(re.match('.*-(\d*)$',ckpt.model_checkpoint_path).group(1))

        if ckpt and ckpt.model_checkpoint_path:
            print("loading snapshot: %s , iter %d" % (ckpt.model_checkpoint_path, num_iter))
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True, num_iter
        else:
            return False, num_iter

    def save(self, step):
        model_name = "model_" + str(step) + ".model"
        self.saver.save(
            self.sess, os.path.join(
                self.models_folder, model_name), global_step=step)
