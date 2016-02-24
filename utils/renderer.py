from panda3d.core import *
from direct.showbase.ShowBase import ShowBase
import math
import numpy as np
import time
import scipy
import scipy.misc
import os
import random


class Renderer(ShowBase):

    def __init__(self, depth, background):

        self.generate_depth = depth
        self.replace_background = background

        loadPrcFileData("", "window-type offscreen")
        loadPrcFileData('', 'win-size 128 128')
        ShowBase.__init__(self)
        base.disableMouse()
        base.setBackgroundColor(0.5, 0.5, 0.5)

        # setup scene
        self.scene = NodePath("Scene")
        self.scene.reparentTo(self.render)
        self.scene.setScale(1, 1, 1)
        self.scene.setTwoSided(True)
        self.scene.setPos(0, 0, 0)
        self.scene.setHpr(0, 0, 0)

        # setup light
        self.plight = PointLight('plight')
        self.plight.setColor(VBase4(1, 1, 1, 1))
        self.plnp = self.render.attachNewNode(self.plight)
        self.plnp.setPos(0, 2, 5)
        self.render.setLight(self.plnp)

        self.plight2 = PointLight('plight2')
        self.plight2.setColor(VBase4(1, 1, 1, 1))
        self.plnp2 = self.render.attachNewNode(self.plight2)
        self.plnp2.setPos(-5, -5, -5)
        self.render.setLight(self.plnp2)

        self.alight = AmbientLight('alight')
        self.alight.setColor(VBase4(10, 10, 10, 1))
        self.alnp = self.render.attachNewNode(self.alight)
        self.render.setLight(self.alnp)

        # prepare texture and camera for depth rendering
        if self.generate_depth is True:
            self.depth_tex = Texture()
            self.depth_tex.setFormat(Texture.FDepthComponent)
            self.depth_buffer = base.win.makeTextureBuffer(
                'depthmap', 128, 128, self.depth_tex, to_ram=True)
            self.depth_cam = self.makeCamera(self.depth_buffer)
            self.depth_cam.reparentTo(base.render)

        # list of models in memory
        self.models = []
        self.backgrounds = []

        self.model = None

    def textureToNormalizedImage(self, texture):
        im = texture.getRamImageAs("RGB")
        strim = im.getData()
        image = np.fromstring(strim, dtype='uint8')
        image = image.reshape(128, 128, 3)
        image = image / 255.0
        image = np.flipud(image)
        return image

    def textureToImage(self, texture):
        im = texture.getRamImageAs("RGB")
        strim = im.getData()
        image = np.fromstring(strim, dtype='uint8')
        image = image.reshape(128, 128, 3)
        image = np.flipud(image)
        return image

    def textureToString(self, texture):
        im = texture.getRamImageAs("RGB")
        return im.getData()

    def setCameraPosition(self, rad, el, az):
        self.camera.setPos(rad*math.cos(el)*math.cos(az),
                           rad*math.cos(el)*math.sin(az),
                           rad*math.sin(el))
        self.camera.lookAt(0, 0, 0)

        if self.generate_depth is True:
            self.depth_cam.setPos(rad*math.cos(el)*math.cos(az),
                                  rad*math.cos(el)*math.sin(az),
                                  rad*math.sin(el))
            self.depth_cam.lookAt(0, 0, 0)

    def loadImagenetBackgrounds(self, path, start, count):
        start_time = time.time()
        for i in range(start, start + count):
            fname = "ILSVRC2012_preprocessed_val_" + str(i).zfill(8) + ".JPEG"
            im = scipy.misc.imread(os.path.join(path, fname))
            im = scipy.misc.imresize(im, (128, 128, 3), interp='nearest')
            self.backgrounds.append(im)
        print("loaded backgrounds in %f s" % (time.time() - start_time))

    def selectModel(self, model_ind):
        self.model = self.models[model_ind]
        self.model.reparentTo(self.scene)

    def unselectModel(self, model_ind):
        self.model.detachNode()
        self.model = None

    def loadModels(self, model_names, models_path):
        mn = []
        for i in range(0, len(model_names)):
            mn.append(
                models_path + "/" +
                model_names[i].rstrip() + "/model.bam")

        self.models = self.loader.loadModel(mn)

    def unloadModels(self):
        for model in self.models:
            model.removeNode()

    # camera/light pos as [rad, el, az], el/az in angles
    def renderView(self, camera_pos, light_pos):
        self.setCameraPosition(camera_pos[0],
                               math.radians(camera_pos[1]),
                               math.radians(camera_pos[2]))
        # set light pos
        lp_rad = light_pos[0]
        lp_el = math.radians(light_pos[1])
        lp_az = math.radians(light_pos[2])

        self.plnp.setPos(lp_rad*math.cos(lp_el)*math.cos(lp_az),
                         lp_rad*math.cos(lp_el)*math.sin(lp_az),
                         lp_rad*math.sin(lp_el))

        base.graphicsEngine.renderFrame()
        tex = base.win.getScreenshot()
        im = self.textureToImage(tex)
        data = np.zeros((128,128), dtype=np.uint16)

        if self.generate_depth is True:
            depth_im = PNMImage()
            self.depth_tex.store(depth_im)
            ss = StringStream()
            depth_im.write(ss, 'pgm')
            data = ss.getData()
            data = data[16:32768+16]
            data = np.fromstring(data, dtype=np.uint16).reshape(128, 128)
            data[0, 0] = 65535

        if self.replace_background is True:
            mask = (data >= 60000)
            idx = (mask != 0)
            bg_ind = random.randint(0, len(self.backgrounds)-1)
            im[idx] = self.backgrounds[bg_ind][idx]

        return im, data
