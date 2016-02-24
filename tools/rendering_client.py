from mnn.utils.renderer import *
from mnn.utils.sys import *
from mnn.utils.data import *

import random
import getopt
import sys
import time
import struct


def main(argv=None):

    if argv is None:
        argv = sys.argv
    opts, args = getopt.getopt(argv[1:], "s:bd", ["start", "background", "depth"])
    start = 1
    background = False
    depth = False
    for o, a in opts:
        if o in ("-s", "--start"):
            start = int(a)
        if o in ("-b", "--background"):
            background = True
        if o in ("-d", "--depth"):
            depth = True
    print("Models: " + str(args))

    sender = SyncDataSender("localhost", 8888)

    rend = Renderer(depth, background)
    rend.loadImagenetBackgrounds("/misc/lmbraid12/datasets/public/ImageNet/ILSVRC2012_preprocessed/val",
        start, 100)

    rend.loadModels(
        args,
        "/misc/lmbraid17/tatarchm/datasets/ShapeNet/3d_models/egg_cars")

    start_time = time.time()
    for i in range(0, 1000):
        light_pos1 = [random.random()*3. + 4.,
                      random.randint(10, 90),
                      random.randint(0, 360)]

        #light_pos = [5, 5, 5]

        rend.selectModel(i % 20)

        rad1 = random.random() * 0.6 + 1.7
        el1 = int(random.random() * 50 - 10)
        az1 = int(random.random() * 360)
        im1 = rend.renderView([rad1, el1, az1], light_pos1)
        #scipy.misc.toimage(im1[0], cmin=0, cmax=255).save("output1.png")

        light_pos2 = light_pos1
        rad2 = random.random() * 0.6 + 1.7
        el2 = int(random.random() * 50 - 10)
        az2 = int(random.random() * 360)
        im2 = rend.renderView([rad2, el2, az2], light_pos2)
        #scipy.misc.toimage(im2[0], cmin=0, cmax=255).save("output2.png")
        #scipy.misc.toimage(im2[1], cmin=0, cmax=65535).save("depth2.png")

        message = im1[0].tostring() + im2[0].tostring() + im2[1].tostring() +\
                        struct.pack("<I", int(100 * rad1)) +\
                        struct.pack("<I", el1 + 90) + \
                        struct.pack("<I", az1) + \
                        struct.pack("<I", int(100 * light_pos1[0])) +\
                        struct.pack("<I", light_pos1[1] + 90) + \
                        struct.pack("<I", light_pos1[2]) + \
                        struct.pack("<I", int(100 * rad2)) + \
                        struct.pack("<I", el2 + 90) + \
                        struct.pack("<I", az2) + \
                        struct.pack("<I", int(100 * light_pos2[0])) +\
                        struct.pack("<I", light_pos2[1]+90) + \
                        struct.pack("<I", light_pos2[2])

        sender.send(message)

        rend.unselectModel(i % 20)

        if i % 100 == 0 and i > 0:
            fps = 200.0 / (time.time() - start_time)
            print("%f fps" % fps)
            start_time = time.time()

if __name__ == "__main__":
    sys.exit(main())
