import importlib
import getopt
import os
import tensorflow as tf


experiments_dir = \
   "/misc/lmbraid17/tatarchm/projects/cars_project/experiments/models"


class MnnApplication:

    def __init__(self, cmd_arguments):

        self.net_def = None
        self.mode = None

        print(cmd_arguments)

        opts, args = getopt.getopt(cmd_arguments, "d:m:",
                                   ["def", "mode"])
        for o, a in opts:
            if o in ("-d", "--def"):
                self.net_def = str(a)
            if o in ("-m", "--mode"):
                self.mode = str(a)

    def start(self):
        os.chdir(experiments_dir + "/" + self.net_def)

        module = importlib.import_module("models." + self.net_def + ".model")

        with tf.Session() as sess:
            self.network = module.Model(sess)

        if self.mode == "train":
            self.network.train()
        elif self.mode == "test":
            self.network.test()
