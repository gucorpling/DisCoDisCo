from glob import glob
from argparse import ArgumentParser

if __name__ == "__main__":

    p = ArgumentParser()
    p.add_argument("-f", "--file", help="dir containing subdirectories for each languages, including sent_test.pred "
                                        "file",
                   default='/Users/shabnam/Desktop/GU/Projects/DISRPT/sharedtask2019/correctDATA/data/2019-output'
                           '-testing/')
    p.add_argument("-m", "--mode", help="train/test/dev",
                   default="dev")
    opts = p.parse_args()

    folders = glob(opts.file + '/' + '*/')
    for data_dir in folders:
        sents = []
        begin = True
        with open(data_dir + '/sent_' + opts.mode + '.pred', 'r') as inp:
            for line in inp:
                line = line.rstrip()
                if len(line) == 0:
                    continue
                if begin and line != "<s>":
                    sents.append("<s>")
                begin = False
                sents.append(line)
        new_sents = []
        count = 0
        for j in range(len(sents)):
            if count == 256 and new_sents[j] != "</s>":
                new_sents.append("</s>")
                new_sents.append("<s>")
                count = 0
            new_sents.append(sents[j])
            count += 1

        with open(data_dir + '/sent_' + opts.mode + '.predV2', 'w') as out:
            for s in new_sents:
                out.write(s + '\n')
