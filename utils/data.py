import socket
import numpy as np
import math
import os
import scipy.misc
import time
from mnn.utils.im import *

MSGLEN = 128 * 128 * 3 + 128 * 128 * 3 + 128 * 128 * 2 + 12 * 4
#         img1             img2            depth2     viewpoint (rad,az,el) and light (rad,az,el) for 2 images, all as int


class SyncDataReceiver():

    def __init__(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self, host, port):
        self.host = host
        self.port = port
        self.client_socket.connect((host, port))

    def reconnect(self):
        print("reconnecting...")
        self.client_socket.close()
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.host, self.port))

    def receiveMessage(self):
        chunks = []
        bytes_recd = 0
        while bytes_recd < MSGLEN:
            #ready = select.select([self.client_socket], [], [], 10)
            #if ready[0]:
            chunk = self.client_socket.recv(min(MSGLEN - bytes_recd, 2048))
            if chunk == b'':
                print("socket error")
                return None
            chunks.append(chunk)
            bytes_recd = bytes_recd + len(chunk)
            #else:
            #    print("server dead")
            #    self.client_socket.close()
            #    self.client_socket.connect((self.host, self.port))
        return b''.join(chunks)

    def decodeMessage(self, message):
        RAD_SCALE = 10. # to make radius in smaller range befor feeding to the net
        image1 = np.fromstring(message[0:49152], dtype='uint8')
        image1 = image1.reshape(128, 128, 3)
        image1 = (image1 / 255.0 - 0.5) * 1.5

        image2 = np.fromstring(message[49152:98304], dtype='uint8')
        image2 = image2.reshape(128, 128, 3)
        image2 = (image2 / 255.0 - 0.5) * 1.5

        depth2 = np.fromstring(message[98304:131072], dtype='uint16')
        #depth2 = depth2.reshape(128, 128, 1)
        #scipy.misc.toimage(depth2, cmin=0, cmax=65535).save("depth2.png")
        depth2 = (depth2 / 65535.0 - 0.5) * 1.5

        tmp_label = np.fromstring(message[131072:131072 + 12*4], dtype='int32')
        rad1 = tmp_label[0] / 100. / RAD_SCALE
        el1 = math.radians(tmp_label[1] - 90)
        az1 = math.radians(tmp_label[2])
        light_rad1 = tmp_label[3] / 100. / RAD_SCALE
        light_el1 = math.radians(tmp_label[4] - 90)
        light_az1 = math.radians(tmp_label[5])
        rad2 = tmp_label[6] / 100. / RAD_SCALE
        el2 = math.radians(tmp_label[7] - 90)
        az2 = math.radians(tmp_label[8])
        light_rad2 = tmp_label[9] / 100. / RAD_SCALE
        light_el2 = math.radians(tmp_label[10] - 90)
        light_az2 = math.radians(tmp_label[11])

        #TODO finish
        label1 = np.array([rad1, math.sin(el1), math.cos(el1),
                           math.sin(az1), math.cos(az1),
                           light_rad1, math.sin(light_el1), math.cos(light_el1),
                           math.sin(light_az1), math.cos(light_az1)])
        label2 = np.array([rad2, math.sin(el2), math.cos(el2),
                           math.sin(az2), math.cos(az2),
                           light_rad2, math.sin(light_el2), math.cos(light_el2),
                           math.sin(light_az2), math.cos(light_az2)])
        return image1, image2, depth2, label1, label2


class SyncDataSender():

    def __init__(self, host, port):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("connecting to server...")
        self.client_socket.connect((host, port))
        print("connected")

    def send(self, msg):
        msglen = MSGLEN
        totalsent = 0

        while totalsent < msglen:
            sent = self.client_socket.send(msg[totalsent:])
            if sent == 0:
                raise RuntimeError("socket connection broken")
            totalsent = totalsent + sent


class DataLoader():

    def __init__(self):
        print("data loader")

    def loadOneImage(self, fname, batch_size):
        im = scipy.misc.imread(fname)
        input_image = []
        labels = []
        el = math.radians(20)
        for i in range(0, batch_size):
            rad = (1 + i / 64.0) / 10.0
            az = math.radians(((i * 10) % 360))
            input_image.append(im)
            labels.append([rad, math.sin(el), math.cos(el),
                          math.sin(az), math.cos(az)])
        input_image = (np.array(input_image) / 255.0 - 0.5) * 2
        return input_image, labels

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
                labels_dic[parts[1]] = np.array([20.0 / 100.0,
                                                 math.sin(el),
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

        input_images = (np.array(input_images) / 255.0 - 0.5) * 2
        output_images = (np.array(output_images) / 255.0 - 0.5) * 2
        #input_images = np.array(input_images) / 255.0
        #output_images = np.array(output_images) / 255.0

        save_images(input_images, [8, 8], "./input.png")
        save_images(output_images, [8, 8], "./gt.png")

        return input_images, output_images, labels
