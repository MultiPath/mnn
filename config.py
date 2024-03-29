import json


class Configuration():

    def __init__(self):
        self.config = None

    def load(self, fname):
        with open(fname, 'r') as f:
            self.config = json.load(f)

    def save(self, fname):
        with open(fname, 'w') as f:
            json.dump(self.config, f)

    def get(self, prop):
        if prop in self.config.keys():
            return self.config[prop]
        else:
            return None

    def set(self, prop, value):
        self.config[prop] = value
