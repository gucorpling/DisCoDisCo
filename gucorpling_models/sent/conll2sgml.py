import io
import os
from collections import defaultdict

# the following directory should "only" containing data folders:

DATA_DIR = '/home/shabnam/data/codes/data/dstrpt/2019/'
OUT_DIR = '/home/shabnam/data/codes/data/dstrpt/2019-output/'

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

names = os.listdir(DATA_DIR + "/")
partition = ["_train", "_dev", "_test"]

for n in names:
    if not os.path.isdir(DATA_DIR + n + "/"):
        continue
    file_ = DATA_DIR + n + "/" + n
    data = defaultdict(list)
    for p in partition:
        file = file_ + p + ".conll"
        lines = io.open(file, encoding="utf8").read().strip().split("\n")
        path = OUT_DIR + n + "/"
        if not os.path.exists(path):
            os.makedirs(path)
        with io.open(path + "sent" + p + ".tt", "w", encoding="utf8", newline="\n") as f:
            begin = True

            for line in lines:
                if not line:
                    continue
                if line.startswith("1" + '\t'):
                    if not begin:
                        f.write("</s>" + "\n")
                    f.write("<s>" + "\n")
                    begin = False
                if line[0].isdigit():
                    f.write(line.split('\t')[1] + "\n")
            f.write("</s>" + "\n")
