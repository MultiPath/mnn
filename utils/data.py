import socket
import numpy as np
import math
import os
import scipy.misc
from mnn.utils.im import *

# image1 size + image2 size + radius, azimuth, elevation (int32 each)
MSGLEN = 128 * 128 * 3 + 128 * 128 * 3 + 3 * 4


class SyncDataReceiver():

    def __init__(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self, host, port):
        self.client_socket.connect((host, port))

    def receiveMessage(self):
        chunks = []
        bytes_recd = 0
        while bytes_recd < MSGLEN:
            chunk = self.client_socket.recv(min(MSGLEN - bytes_recd, 2048))
            if chunk == '':
                raise RuntimeError("socket connection broken")
            chunks.append(chunk)
            bytes_recd = bytes_recd + len(chunk)
        return b''.join(chunks)

    def decodeMessage(self, message):
        image1 = np.fromstring(message[0:128*128*3], dtype='uint8')
        image1 = image1.reshape(128, 128, 3)
        image1 = (image1 / 255.0 - 0.5) * 2
        image1 = np.flipud(image1)

        image2 = np.fromstring(message[128*128*3:128*128*3*2], dtype='uint8')
        image2 = image2.reshape(128, 128, 3)
        image2 = (image2 / 255.0 - 0.5) * 2
        image2 = np.flipud(image2)

        tmp_label = np.fromstring(message[128*128*3*2:MSGLEN], dtype='int32')
        el = math.radians(tmp_label[2]*10)
        az = math.radians(tmp_label[1]*10)

        label = np.array([math.sin(el), math.cos(el),
                          math.sin(az), math.cos(az)])
        return image1, image2, label


class DataLoader():

    def __init__(self):
        print("data loader")

    def loadTestBatch(self, folder):
        input_images_dic = {}
        output_images_dic = {}
        labels_dic = {}

        for i in os.listdir(folder):
            im = scipy.misc.imread(os.path.join(folder, i))
            val = i.split(".")[0]
            parts = val.split("_")
            if parts[0] == "input":
                input_images_dic[parts[1]] = im
            else:
                el = math.radians(int(parts[2]))
                az = math.radians(int(parts[3]))
                output_images_dic[parts[1]] = im
                labels_dic[parts[1]] = np.array([math.sin(el),
                                                 math.cos(el),
                                                 math.sin(az),
                                                 math.cos(az)])

        input_images = []
        output_images = []
        labels = []
        for key in sorted(input_images_dic):
            input_images.append(input_images_dic[key])
            output_images.append(output_images_dic[key])
            labels.append(labels_dic[key])

        #input_images = (np.array(input_images) / 255.0 - 0.5) * 2
        #output_images = (np.array(output_images) / 255.0 - 0.5) * 2
        input_images = np.array(input_images) / 255.0
        output_images = np.array(output_images) / 255.0

        save_images(input_images, [8, 8], "./input.png")
        save_images(output_images, [8, 8], "./gt.png")

        return input_images, output_images, labels
