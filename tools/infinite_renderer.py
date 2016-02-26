import os
import sys
import getopt


def main(argv=None):

    if argv is None:
        argv = sys.argv
    opts, args = getopt.getopt(argv[1:], "s:bd", ["start", "background", "depth"])
    start = 0
    background = False
    depth = False
    for o, a in opts:
        if o in ("-s", "--start"):
            start = int(a)
        if o in ("-b", "--background"):
            background = True
        if o in ("-d", "--depth"):
            depth = True


    with open("models.txt") as f:
        model_names = f.readlines()

    model_num = start
    model_count = 20
    bg_num = 1

    while True:
        command = "python rendering_client.py -s " + str(bg_num)
        if background is True:
            command += " -b"
        if depth is True:
            command += " -d"
        print("Model num: %d" % model_num)
        for i in range(model_num, model_num + model_count):
            command += " " + model_names[i].rstrip()

        model_num = model_num + model_count
        if(model_num + model_count > len(model_names)):
            model_num = 0
        if(bg_num + 100 > 50000):
            bg_num = 1
        bg_num += 100

        os.system(command)

if __name__ == "__main__":
    sys.exit(main())
