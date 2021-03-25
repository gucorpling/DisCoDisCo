#!/bin/bash
set -o errexit
for CORPUS in `ls sharedtask2019/data`; do
	echo ""
	echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
	echo "#### Testing on ${CORPUS}"
	echo "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
	echo ""
	MODEL_DIR=models/${CORPUS}_seg_bert_baseline
	allennlp predict \
		--cuda-device 0 \
		--output-file ${MODEL_DIR}/output_test.jsonl \
		--use-dataset-reader \
		${MODEL_DIR}/model.tar.gz \
		"sharedtask2019/data/${CORPUS}/${CORPUS}_test.conll"
	python seg_scripts/score_binary_output ${MODEL_DIR}/output_test.jsonl > ${MODEL_DIR}/metrics_test.txt
done
