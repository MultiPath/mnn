#!/usr/bin/python
from panda3d.core import *
from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData
from direct.filter.FilterManager import FilterManager
import time
import random
import socket
import math
import sys

import numpy as np

from mnn.utils.im import *


class Renderer(ShowBase):

    def __init__(self):
        ShowBase.__init__(self)
        base.disableMouse()

        with open("models.txt") as f:
            self.model_names = f.readlines()

        print("Loading models...")
        self.loadModels("/misc/lmbraid17/tatarchm/datasets/ShapeNet/3d_models/egg_cars")
        print("Done.")

        self.scene = NodePath("Scene")
        self.scene.reparentTo(self.render)

        self.scene.setScale(1, 1, 1)
        self.scene.setTwoSided(True)
        self.scene.setPos(0, 0, 0)
        self.scene.setHpr(0, 0, 0)
        self.setUpLight()

        #create depth texture
        base.depth_tex = Texture()
        self.depth_tex.setFormat(Texture.FDepthComponent)
        self.depth_buffer=base.win.makeTextureBuffer('depthmap', 128, 128, self.depth_tex, to_ram=True)
        self.depth_cam = self.makeCamera(self.depth_buffer)
        self.depth_cam.reparentTo(base.render)

        self.generateData()

    def setCameraPosition(self, radius, azimuth, elevation):
        self.camera.setPos(radius*math.cos(elevation)*math.cos(azimuth),
                           radius*math.cos(elevation)*math.sin(azimuth),
                           radius*math.sin(elevation))
        self.camera.lookAt(0, 0, 0)

        self.depth_cam.setPos(radius*math.cos(elevation)*math.cos(azimuth),
                           radius*math.cos(elevation)*math.sin(azimuth),
                           radius*math.sin(elevation))
        self.depth_cam.lookAt(0, 0, 0)

    def textureToNormalizedImage(self, texture):
        im = texture.getRamImageAs("RGB")
        strim = im.getData()
        image = np.fromstring(strim, dtype='uint8')
        image = image.reshape(128, 128, 3)
        image = image / 255.0
        image = np.flipud(image)
        return image

    def generateData(self):
        #self.manager = FilterManager(base.win, base.cam)
        for i in range(0, len(self.model_names)):

            if i > 0:
                self.models[i-1].detachNode()
            self.models[i].reparentTo(self.scene)

            for j in range(0, 35):

                rad = int(2)
                #az1 = random.randint(0, 35) * 10
                #el1 = random.randint(0, 4) * 10
                az1 = 12 * 10
                el1 = 2 * 10
                self.setCameraPosition(rad,
                                   math.radians(az1),
                                   math.radians(el1))

                base.graphicsEngine.renderFrame()
                tex1 = base.win.getScreenshot()
                im1 = self.textureToNormalizedImage(tex1)
                fname = "input_" + self.model_names[i].rstrip() + str(j) + "_" +\
                    str(el1) + "_" + str(az1) + ".png"
                scipy.misc.imsave(fname, im1)

                rad = int(2)
                #az2 = random.randint(0, 35) * 10
                az2 = j * 10
                #el2 = random.randint(0, 4) * 10
                el2 = 2 * 10
                self.setCameraPosition(rad,
                                   math.radians(az2),
                                   math.radians(el2))

                base.graphicsEngine.renderFrame()
                tex2 = base.win.getScreenshot()
                im2 = self.textureToNormalizedImage(tex2)
                fname = "output_" + self.model_names[i].rstrip() + str(j) + "_" +\
                    str(el2) + "_" + str(az2) + ".png"
                scipy.misc.imsave(fname, im2)

            #im = PNMImage()
            #self.depth_tex.store(im)
            #print(im.getBlueVal(64,64))
            #im.write("test.png")

    def loadModels(self, models_path):
        mn = []

        for i in range(0, len(self.model_names)):
            mn.append(models_path + "/" +
                      self.model_names[i].rstrip() + "/model.bam")

        start_time = time.time()
        self.models = self.loader.loadModel(mn)
        print("---loaded in %s seconds ---" % (time.time() - start_time))

    def setUpLight(self):
        plight1 = PointLight('plight1')
        plight1.setColor(VBase4(1, 1, 1, 1))
        plnp1 = self.render.attachNewNode(plight1)
        plnp1.setPos(0, 2, 5)
        self.render.setLight(plnp1)

        #plight2 = PointLight('plight2')
        #plight2.setColor(VBase4(1, 1, 1, 1))
        #plnp2 = self.render.attachNewNode(plight2)
        #plnp2.setPos(-14, -14, 12)
        #self.render.setLight(plnp2)

        alight = AmbientLight('alight')
        alight.setColor(VBase4(10, 10, 10, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)

    def connectToServer(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("connecting to server...")
        self.client_socket.connect(("localhost", 8888))
        print("connected")


def main(argv=None):
    #loadPrcFileData("", "window-type offscreen")
    loadPrcFileData('', 'win-size 128 128')
    app = Renderer()
    app.run()

if __name__ == "__main__":
    sys.exit(main())
