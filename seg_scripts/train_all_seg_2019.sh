#!/bin/bash
set -o errexit
for CORPUS in `ls sharedtask2019/data`; do
	echo ""
	echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
        echo "#### Training on ${CORPUS}"
        echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
	echo ""
	TRAIN_DATA_PATH="sharedtask2019/data/${CORPUS}/${CORPUS}_train.conll" \
		VALIDATION_DATA_PATH="sharedtask2019/data/${CORPUS}/${CORPUS}_dev.conll" \
		allennlp train \
		configs/seg/baseline/bert_baseline.jsonnet \
		-s models/${CORPUS}_seg_bert_baseline
done
