#!/usr/bin/env python
import sys

if len(sys.argv) == 1:
    print("Provide an output filename created with `allennlp predict`")
    sys.exit(1)

POSITIVE_LABEL = "BeginSeg"
NEGATIVE_LABEL = "_"


import json

with open(sys.argv[1]) as f:
    s = f.read()

TP = 0
FP = 0
TN = 0
FN = 0
total = 0
labels_seen = []
for line in s.strip().split("\n"):
    data = json.loads(line)
    for gold, pred in zip(data["gold_labels"], data["pred_labels"]):
        if gold not in [POSITIVE_LABEL, NEGATIVE_LABEL]:
            raise Exception("Unknown label" + gold)
        if pred not in [POSITIVE_LABEL, NEGATIVE_LABEL]:
            raise Exception("Unknown label" + pred)
        if gold not in labels_seen:
            labels_seen.append(gold)
        if pred not in labels_seen:
            labels_seen.append(pred)
        if len(labels_seen) > 2:
            raise Exception("More than two labels encountered!")
        if gold == POSITIVE_LABEL and pred == POSITIVE_LABEL:
            TP += 1
        elif gold == NEGATIVE_LABEL and pred == NEGATIVE_LABEL:
            TN += 1
        elif gold == POSITIVE_LABEL and pred == NEGATIVE_LABEL:
            FN += 1
        elif gold == NEGATIVE_LABEL and pred == POSITIVE_LABEL:
            FP += 1
        total += 1

accuracy = (TP + TN) / total
recall = TP / (TP + FN)
f1 = 2 / ((1 / accuracy) + (1 / recall))

print("Accuracy: ", accuracy)
print("Recall:   ", recall)
print("F1:       ", f1)
