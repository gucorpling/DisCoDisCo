import sys

if len(sys.argv) == 1:
    print("Provide an output filename created with `allennlp predict`")
    sys.exit(1)

if len(sys.argv) == 2:
    print("Provide a path for the conll output to be written to`")
    sys.exit(1)


import json

with open(sys.argv[1], encoding='utf8') as f:
    s = f.read()

with open(sys.argv[2], 'w', encoding='utf8') as f:
    for line in s.strip().split("\n"):
        data = json.loads(line)
        for i, (token, pred_label) in enumerate(zip(data['tokens'], data['pred_labels'])):
            f.write(f"{i+1}\t{token}" + ("\t_" * 7) + "\t" + ("BeginSeg=Yes" if pred_label == "B" else "_") + "\n")
        f.write("\n")
